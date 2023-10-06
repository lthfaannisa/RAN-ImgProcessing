import matplotlib.pyplot as plt
import cv2

def calculate_histogram(img_path):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(img_path)

    # Menghitung histogram untuk masing-masing saluran (R, G, B)
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Normalisasi histogram
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()

    return hist_r, hist_g, hist_b

def save_histogram_image(hist_r, hist_g, hist_b, output_path):
    plt.figure()
    plt.title("RGB Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.plot(hist_r, color='red', label='Red')
    plt.plot(hist_g, color='green', label='Green')
    plt.plot(hist_b, color='blue', label='Blue')
    plt.legend()
    plt.savefig(output_path)
    plt.close('all')

def equalize_image(img_path):
    img = cv2.imread(img_path)
    
    # Hasil equalisasi
    img_equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Ubah ke ruang warna YCrCb
    img_equalized[:, :, 0] = cv2.equalizeHist(img_equalized[:, :, 0])  # Equalisasi komponen Y (luminance)
    img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCrCb2BGR)  # Kembalikan ke ruang warna BGR

    return img_equalized
