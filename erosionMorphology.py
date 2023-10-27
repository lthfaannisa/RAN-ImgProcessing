import cv2
import numpy as np

def apply_erosion(img_path):

    # Membaca citra
    img = cv2.imread(img_path)

    # Mendefinisikan kernel
    kernel = np.ones((5,5),np.uint8)

    # Melakukan operasi erosi
    erosion = cv2.erode(img, kernel, iterations = 1)
    
    return erosion