import cv2
import numpy as np

def cluster_segmentation(image_path, num_clusters):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ubah ruang warna menjadi RGB

    # Bentuk matriks piksel untuk pengelompokan
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Definisikan kriteria berhenti untuk K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Lakukan pengelompokan dengan K-Means
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Ubah label piksel menjadi tipe data uint8
    labels = labels.flatten()
    segmented_img = centers[labels]

    # Bentuk ulang gambar yang telah digroupkan
    segmented_img = segmented_img.reshape(img.shape)

    return segmented_img
