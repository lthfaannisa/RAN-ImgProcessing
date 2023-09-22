from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
from histoEqual import calculate_histogram, save_histogram_image, equalize_image
from faceDetect import detect_faces
from blurFace import apply_face_blur

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)
        face_cascade = cv2.CascadeClassifier('static\haarcascade_frontalface_default.xml')

        if 'equalize' in request.form:  # Tombol "Histogram Equalization" ditekan
            # Menghitung histogram
            hist_r, hist_g, hist_b = calculate_histogram(img_path)

            # Simpan histogram sebagai gambar PNG
            hist_image_path = os.path.join(app.config['UPLOAD'], 'histogram.png')
            save_histogram_image(hist_r, hist_g, hist_b, hist_image_path)

            # Equalisasi histogram
            img_equalized = equalize_image(img_path)

            # Menyimpan gambar hasil equalisasi ke folder "static/uploads"
            equalized_image_path = os.path.join('static', 'uploads', 'img-equalized.jpg')
            cv2.imwrite(equalized_image_path, img_equalized)

            # Menghitung histogram untuk gambar yang sudah diequalisasi
            hist_equalized_r, hist_equalized_g, hist_equalized_b = calculate_histogram(equalized_image_path)

            # Simpan histogram hasil equalisasi sebagai gambar PNG
            hist_equalized_image_path = os.path.join(app.config['UPLOAD'], 'histogram_equalized.png')
            save_histogram_image(hist_equalized_r, hist_equalized_g, hist_equalized_b, hist_equalized_image_path)

            return render_template('index.html', img=img_path, img2=equalized_image_path, histogram=hist_image_path, histogram2=hist_equalized_image_path)

        elif 'detect_face' in request.form:  # Tombol "Detect Face" ditekan
            # Deteksi wajah
            img_with_faces, faces_detected = detect_faces(img_path, face_cascade)

            # Menyimpan gambar hasil deteksi wajah ke folder "static/uploads"
            face_detected_image_path = os.path.join('static', 'uploads', 'img-face-detected.jpg')
            cv2.imwrite(face_detected_image_path, img_with_faces)

            return render_template('index.html', img=img_path, img2=face_detected_image_path, faces_detected=faces_detected)
        
        elif 'blur_face' in request.form:  # Tombol "Blur Face" ditekan
            # Efek blur wajah
            img_blurred = apply_face_blur(img_path, face_cascade)  # Menggunakan cascade classifier yang telah diinisialisasi

            # Menyimpan gambar hasil blur wajah ke folder "static/uploads"
            blurred_image_path = os.path.join('static', 'uploads', 'img-blurred.jpg')
            cv2.imwrite(blurred_image_path, img_blurred)

            return render_template('index.html', img=img_path, img2=blurred_image_path)
        
    return render_template('index.html')

if __name__ == '__main__': 
    app.run(debug=True,port=8001)
