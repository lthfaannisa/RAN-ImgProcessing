import cv2

def add_moustache_to_face(image_path):
    # Load the face and mustache images
    face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
    face_image = cv2.imread(image_path)
    mustache_image = cv2.imread('static/mustache.png', -1)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(face_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Adjust the mustache size and position
        mustache_width = w
        mustache_height = int(mustache_width * 0.5)
        mustache_x = x
        mustache_y = y + int(h / 2)

        # Resize the mustache image to fit the face
        mustache = cv2.resize(mustache_image, (mustache_width, mustache_height))

        # Create a mask for the mustache
        mustache_mask = mustache[:, :, 3]

        # Overlay the mustache on the face
        for c in range(0, 3):
            face_image[mustache_y:mustache_y + mustache_height, mustache_x:mustache_x + mustache_width, c] = \
                face_image[mustache_y:mustache_y + mustache_height, mustache_x:mustache_x + mustache_width, c] * \
                (1 - mustache_mask / 255.0) + \
                mustache[:, :, c] * (mustache_mask / 255.0)

    return face_image
