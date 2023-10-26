import cv2
import numpy as np

def apply_opening(img_path, kernel_size=5):
    # Baca gambar
    img = cv2.imread(img_path)

    # Konversi ke citra grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Definisikan kernel untuk erosi dan dilasi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Erosi
    erosion = cv2.erode(gray, kernel, iterations=1)

    # Dilasi
    opening = cv2.dilate(erosion, kernel, iterations=1)

    return opening
