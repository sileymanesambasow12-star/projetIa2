import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice de confusion - {model_name}')
    plt.ylabel('Vrai')
    plt.xlabel('Preddit')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

def train_for_descriptor(descriptor_name, signature_file):
    try:
        data = np.load(signature_file)
    except:
        print(f"{signature_file} introuvable")
        return None

    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e5, 1e5)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classes = sorted(np.unique(y))

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }

    results = {}

    print("\n" + "="*60)
    print(f"DESCRIPTEUR: {descriptor_name}")
    print("="*60)

    for name, model in models.items():
        print(f"\nEntrainement de {name}...")
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        plot_confusion_matrix(y_test, y_pred, classes, f"{descriptor_name}_{name.replace(' ', '_')}")

    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "="*60)
    print(f"MEILLEUR MODELE pour {descriptor_name}: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("="*60)

    joblib.dump(best_model, f"best_model_{descriptor_name}.pkl")
    joblib.dump(scaler, f"scaler_{descriptor_name}.pkl")
    
    return results

def main():
    descriptors = {
        'concat': 'signatures_concat.npy',
        'glcm': 'signatures_glcm.npy',
        'haralick': 'signatures_haralick.npy',
        'bitdesc': 'signatures_bitdesc.npy'
    }
    
    all_results = {}
    
    for desc_name, signature_file in descriptors.items():
        if os.path.exists(signature_file):
            results = train_for_descriptor(desc_name, signature_file)
            if results:
                all_results[desc_name] = results
        else:
            print(f"{signature_file} non trouve, lancez d'abord extraction.py")
    
    print("\n" + "="*60)
    print("RESUME FINAL")
    print("="*60)
    for desc_name, results in all_results.items():
        best = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"{desc_name}: Meilleur modele = {best[0]} (Accuracy: {best[1]['accuracy']:.4f})")

if __name__ == "__main__":
    main()