import cv2
import numpy as np

def apply_cubic(img_path):

    # Membaca gambar
    img = cv2.imread(img_path)

    # Ukuran gambar asli
    h, w = img.shape[:2]

    # Ukuran gambar yang diinginkan
    new_h, new_w = 2*h, 2*w

    # Melakukan interpolasi cubic
    cubic = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    
    return cubic