import cv2

def reduce_noise(img_path):
    # Baca gambar
    img = cv2.imread(img_path)

    # Terapkan filter Gaussian untuk mengurangi noise
    img_smoothed = cv2.GaussianBlur(img, (5, 5), 0)

    return img_smoothed
