import os
import numpy as np
import joblib
from descriteurs import concat_rgb, glcm_RGB, haralick_RGB, bitdesc_RGB

classes = sorted([d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))])
dict_class = {i: c for i, c in enumerate(classes)}

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def canberra_distance(a, b):
    return np.sum(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-10))

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

distance_functions = {
    "euclidean": euclidean_distance,
    "canberra": canberra_distance,
    "cosine": cosine_distance
}

descriptor_config = {
    "GLCM": {"func": glcm_RGB, "model": "best_model_glcm.pkl", "scaler": "scaler_glcm.pkl"},
    "Haralick": {"func": haralick_RGB, "model": "best_model_haralick.pkl", "scaler": "scaler_haralick.pkl"},
    "BitDesc": {"func": bitdesc_RGB, "model": "best_model_bitdesc.pkl", "scaler": "scaler_bitdesc.pkl"},
    "Concaténation": {"func": concat_rgb, "model": "best_model_concat.pkl", "scaler": "scaler_concat.pkl"}
}

def get_descriptor(descriptor_name, image_path):
    config = descriptor_config.get(descriptor_name)
    if config:
        return config["func"](image_path)
    return None

def search(image_path, k=5, distance_name="euclidean", descriptor_name="Concaténation"):
    
    config = descriptor_config.get(descriptor_name)
    if not config:
        return []
    
    model = joblib.load(config["model"])
    scaler = joblib.load(config["scaler"])
    
    features = get_descriptor(descriptor_name, image_path)
    if features is None:
        print(f"Impossible d'extraire les caracteristiques avec {descriptor_name}")
        return []
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)[0]
    
    pred = model.predict([features_scaled])[0]
    classe = dict_class[pred]
    
    print(f"Classe predite: {classe}")
    
    dossier = os.path.join("dataset", classe)
    
    if not os.path.exists(dossier):
        print(f"Dossier {dossier} introuvable")
        return []
    
    results = []
    distance_func = distance_functions.get(distance_name, euclidean_distance)
    
    for file in os.listdir(dossier):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        path = os.path.join(dossier, file)
        
        img_features = get_descriptor(descriptor_name, path)
        if img_features is None:
            continue
        
        img_features = np.array(img_features)
        img_features_scaled = scaler.transform([img_features])[0]
        dist = distance_func(features_scaled, img_features_scaled)
        
        results.append((path, dist))
    
    results = sorted(results, key=lambda x: x[1])
    
    return results[:k]