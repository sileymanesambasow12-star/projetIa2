import os
import numpy as np
import joblib
from descriteurs import concat_rgb

classes = sorted([d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))])
dict_class = {i: classe for i, classe in enumerate(classes)}

model = joblib.load('best_model_concat.pkl')

path = 'dataset/cat/0b54dde5f5.jpg'

carac = concat_rgb(path)

if carac is None:
    print("Erreur extraction")
else:
    carac = np.array([carac])
    pred = model.predict(carac)[0]   
    print("Classe predite :", dict_class[pred])