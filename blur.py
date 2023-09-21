import cv2
import os

def apply_blur(img_path, kernel_size=(15, 15)):
    img = cv2.imread(img_path)
    
    # Terapkan efek Gaussian Blur
    img_blurred = cv2.GaussianBlur(img, kernel_size, 0)
    
    return img_blurred
