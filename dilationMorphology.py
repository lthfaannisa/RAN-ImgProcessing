import cv2
import numpy as np

def apply_dilation(img_path):

    # Membaca citra
    img = cv2.imread(img_path)

    # Mendefinisikan kernel
    kernel = np.ones((5,5),np.uint8)

    # Melakukan operasi dilasi
    dilation = cv2.dilate(img, kernel, iterations = 1)

    return dilation
