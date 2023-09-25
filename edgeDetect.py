import cv2

def detect_edges(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Membaca gambar dalam skala abu-abu
    edges = cv2.Canny(img, 100, 200)  # Deteksi tepi dengan algoritma Canny

    return edges
