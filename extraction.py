import os
import numpy as np
import pandas as pd
from descriteurs import concat_rgb, glcm_RGB, haralick_RGB, bitdesc_RGB

def extraction(chemin_dossier, descriptor_name, descriptor_func):
    if not os.path.exists(chemin_dossier):
        print(f"Dataset introuvable: {chemin_dossier}")
        return

    classes = sorted([d for d in os.listdir(chemin_dossier)
                      if os.path.isdir(os.path.join(chemin_dossier, d))])

    if len(classes) == 0:
        print("Aucune classe trouvee")
        return

    print(f"Extraction {descriptor_name}...")
    print(f"Classes: {classes}")

    dict_class = {classe: i for i, classe in enumerate(classes)}
    list_caracteristiques = []
    count = 0

    for root, _, files in os.walk(chemin_dossier):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                count += 1
                path = os.path.join(root, file)
                class_name = os.path.basename(os.path.dirname(path))

                if count % 100 == 0:
                    print(f"Progression: {count} images traitees, {len(list_caracteristiques)} valides")

                if class_name in dict_class:
                    carac = descriptor_func(path)
                    if carac is not None and len(carac) > 0:
                        list_caracteristiques.append(carac + [dict_class[class_name]])

    print(f"Total images: {count}")
    print(f"Images valides: {len(list_caracteristiques)}")

    if len(list_caracteristiques) == 0:
        print("Aucune image traitee")
        return

    signatures = np.array(list_caracteristiques)
    np.save(f'signatures_{descriptor_name}.npy', signatures)
    print(f"Extraction {descriptor_name} terminee: {len(signatures)} images")

def main():
    descriptors = {
        'concat': concat_rgb,
        'glcm': glcm_RGB,
        'haralick': haralick_RGB,
        'bitdesc': bitdesc_RGB
    }
    
    for name, func in descriptors.items():
        extraction('dataset', name, func)
        print()

if __name__ == '__main__':
    main()