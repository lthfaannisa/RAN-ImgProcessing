import cv2
import numpy as np

def apply_linear(img_path):

    # Membaca citra
    img = cv2.imread(img_path)
    
    # Ukuran gambar 
    width = 200
    height = 200

    # Melakukan interpolasi linear
    linear = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return linear