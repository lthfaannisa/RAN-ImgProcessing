import cv2
import numpy as np

def apply_closing(image_path, kernel_size=5):
    # Baca gambar
    img = cv2.imread(image_path)

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Buat kernel untuk operasi closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Lakukan operasi closing
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return closing
