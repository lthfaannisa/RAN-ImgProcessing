import cv2

def reduce_noise(img_path):
    # Baca gambar
    img = cv2.imread(img_path)

    # Terapkan filter bilateral untuk mengurangi noise
    img_smoothed = cv2.bilateralFilter(img, 5, 100, 100)

    return img_smoothed
