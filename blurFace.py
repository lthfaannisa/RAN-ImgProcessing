import cv2

def apply_face_blur(img_path, face_cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam gambar
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Terapkan efek blur pada wajah yang terdeteksi
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30)  # Sesuaikan ukuran kernel dan kekuatan blur
        img[y:y+h, x:x+w] = face

    return img
