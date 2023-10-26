import cv2
import numpy as np

def apply_closing(image_path):
    # Baca gambar
    img = cv2.imread(image_path)

    # Buat kernel untuk operasi closing
    kernel = np.ones((5, 5), np.uint8)

    # Lakukan operasi closing
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closing
