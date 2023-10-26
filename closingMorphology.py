import cv2
import numpy as np

def apply_closing(image_path):
    # Baca gambar
    img = cv2.imread(image_path)

    # Buat kernel untuk operasi closing
    kernel = np.ones((5, 5), np.uint8)

    # Dilasi
    dilasi = cv2.dilate(img, kernel, iterations=1)

    # Erosi
    closing = cv2.erode(dilasi, kernel, iterations=1)
    
    return closing