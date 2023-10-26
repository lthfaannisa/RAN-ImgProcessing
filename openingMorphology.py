import cv2
import numpy as np

def apply_opening(img_path):
    # Baca gambar
    img = cv2.imread(img_path)

    # Definisikan kernel untuk erosi dan dilasi
    kernel = np.ones((5, 5), np.uint8)

    # Erosi
    erosion = cv2.erode(img, kernel, iterations=1)

    # Dilasi
    opening = cv2.dilate(erosion, kernel, iterations=1)

    return opening
