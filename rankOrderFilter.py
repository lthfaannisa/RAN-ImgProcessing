import numpy as np
import cv2

def rank_order_filter(img, window_size, rank):
    img_filtered = np.copy(img)
    half_window = window_size // 2

    for c in range(img.shape[2]):  # Loop over color channels (assuming it's a 3-channel image)
        for i in range(half_window, img.shape[0] - half_window):
            for j in range(half_window, img.shape[1] - half_window):
                window = img[i - half_window : i + half_window + 1, j - half_window : j + half_window + 1, c]
                img_filtered[i, j, c] = np.partition(window.flatten(), rank)[rank]

    return img_filtered

def remove_salt_and_pepper_noise_rank_order(img_path, window_size, rank):
    img = cv2.imread(img_path)
    img_filtered = rank_order_filter(img, window_size, rank)
    return img_filtered
