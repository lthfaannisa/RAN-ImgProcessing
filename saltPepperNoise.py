import numpy as np
import cv2

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Add salt noise
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 1

    # Add pepper noise
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0

    return noisy_image

def apply_salt_and_pepper_noise(image_path, salt_prob, pepper_prob):
    # Read the image
    img = cv2.imread(image_path)

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Add salt and pepper noise
    noisy_img = add_salt_and_pepper_noise(img, salt_prob, pepper_prob)

    # Convert back to the range [0, 255]
    noisy_img = (noisy_img * 255).astype(np.uint8)

    return noisy_img
