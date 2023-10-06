from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
from histoEqual import calculate_histogram, save_histogram_image, equalize_image
from blurFace import apply_face_blur
from edgeDetect import detect_edges
from imgSegment import cluster_segmentation
from faceMoustache import add_moustache_to_face
from reduceNoise import reduce_noise

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

        elif 'blur_face' in request.form:  # Tombol "Blur Face" ditekan
            face_cascade = cv2.CascadeClassifier('static\haarcascade_frontalface_default.xml')
            
            # Efek blur wajah
            img_blurred = apply_face_blur(img_path, face_cascade)  # Menggunakan cascade classifier yang telah diinisialisasi

            # Menyimpan gambar hasil blur wajah ke folder "static/uploads"
            blurred_image_path = os.path.join('static', 'uploads', 'img-blurred.jpg')
            cv2.imwrite(blurred_image_path, img_blurred)

            return render_template('index.html', img=img_path, img2=blurred_image_path)
        
        elif 'detect_edges' in request.form:  # Tombol "Edge Detection" ditekan
            edges = detect_edges(img_path)

            # Menyimpan gambar hasil deteksi tepi ke folder "static/uploads"
            edges_image_path = os.path.join('static', 'uploads', 'img-edges.jpg')
            cv2.imwrite(edges_image_path, edges)

            return render_template('index.html', img=img_path, img2=edges_image_path)
        
        elif 'segment_image' in request.form:  # Tombol "Segmentasi Citra" ditekan
            num_clusters = 5  # Ganti jumlah klaster sesuai kebutuhan
            segmented_image = cluster_segmentation(img_path, num_clusters)

            # Menyimpan gambar hasil segmentasi ke folder "static/uploads"
            segmented_image_path = os.path.join('static', 'uploads', 'img-segmented.jpg')
            cv2.imwrite(segmented_image_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

            return render_template('index.html', img=img_path, img2=segmented_image_path)
        
        elif 'add_moustache' in request.form:  # Tombol "Tambah Kumis" ditekan
            img_with_moustache = add_moustache_to_face(img_path)  # Menambahkan kumis

            # Menyimpan gambar hasil tambah kumis ke folder "static/uploads"
            img_with_moustache_path = os.path.join('static', 'uploads', 'img-with-moustache.jpg')
            cv2.imwrite(img_with_moustache_path, img_with_moustache)

            return render_template('index.html', img=img_path, img2=img_with_moustache_path)
        
        elif 'reduce_noise' in request.form:  # Tombol "Reduce Noise" ditekan
            # Mengurangi noise pada gambar
            img_smoothed = reduce_noise(img_path)

            # Menyimpan gambar hasil pengurangan noise ke folder "static/uploads"
            smoothed_image_path = os.path.join('static', 'uploads', 'img-smoothed.jpg')
            cv2.imwrite(smoothed_image_path, img_smoothed)

            return render_template('index.html', img=img_path, img2=smoothed_image_path)
        
    return render_template('index.html')

if __name__ == '__main__': 
    app.run(debug=True,port=8001)
