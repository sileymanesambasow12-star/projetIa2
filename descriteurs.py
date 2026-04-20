import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo
import cv2

def safe_read(chemin):
    data = cv2.imread(chemin)
    if data is None:
        return None
    return cv2.resize(data, (128, 128))

def glcm_RGB(chemin):
    data = safe_read(chemin)
    if data is None:
        return []

    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    list_cara = []
    for i in range(3):
        canal = data[:,:,i]
        try:
            co = graycomatrix(canal,[1],[np.pi/2],symmetric=True,normed=True)
            features = [
                graycoprops(co,'contrast')[0,0],
                graycoprops(co,'homogeneity')[0,0],
                graycoprops(co,'energy')[0,0],
                graycoprops(co,'correlation')[0,0]
            ]
        except:
            features = [0]*4

        list_cara.extend(features)

    return list_cara

def haralick_RGB(chemin):
    data = safe_read(chemin)
    if data is None:
        return []

    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    list_cara = []
    for i in range(3):
        canal = data[:,:,i]
        try:
            features = haralick(canal).mean(0)
        except:
            features = [0]*13

        list_cara.extend(features)

    return list_cara

def bitdesc_RGB(chemin):
    data = safe_read(chemin)
    if data is None:
        return []

    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    list_cara = []
    for i in range(3):
        canal = data[:,:,i]
        try:
            features = bio_taxo(canal)
        except:
            features = [0]*14

        list_cara.extend(features)

    return list_cara

def concat_rgb(chemin):
    try:
        result = glcm_RGB(chemin) + haralick_RGB(chemin) + bitdesc_RGB(chemin)
        if len(result) == 0:
            return None
        return result
    except Exception as e:
        return None