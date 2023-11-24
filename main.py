import os
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from histoEqual import calculate_histogram, save_histogram_image, equalize_image
from blurFace import apply_face_blur
from edgeDetect import detect_edges
from imgSegment import cluster_segmentation
from faceMoustache import add_moustache_to_face
from reduceNoise import reduce_noise
from openingMorphology import apply_opening
from closingMorphology import apply_closing
from dilationMorphology import apply_dilation
from erosionMorphology import apply_erosion
from cubicInterpolation import apply_cubic
from linearInterpolation import apply_linear
from saltPepperNoise import apply_salt_and_pepper_noise
from rankOrderFilter import remove_salt_and_pepper_noise_rank_order

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

        if 'equalize' in request.form:
            # Menghitung histogram
            hist_r, hist_g, hist_b = calculate_histogram(img_path)
            hist_image_path = os.path.join(app.config['UPLOAD'], 'histogram.png')
            save_histogram_image(hist_r, hist_g, hist_b, hist_image_path)

            # Equalisasi histogram
            img_equalized = equalize_image(img_path)
            equalized_image_path = os.path.join('static', 'uploads', 'img-equalized.jpg')
            cv2.imwrite(equalized_image_path, img_equalized)

            # Menghitung histogram untuk gambar yang sudah diequalisasi
            hist_equalized_r, hist_equalized_g, hist_equalized_b = calculate_histogram(equalized_image_path)
            hist_equalized_image_path = os.path.join(app.config['UPLOAD'], 'histogram_equalized.png')
            save_histogram_image(hist_equalized_r, hist_equalized_g, hist_equalized_b, hist_equalized_image_path)

            return render_template('index.html', img=img_path, img2=equalized_image_path, histogram=hist_image_path, histogram2=hist_equalized_image_path)

        elif 'blur_face' in request.form:
            # Inisialisasi cascade classifier
            face_cascade = cv2.CascadeClassifier('static\haarcascade_frontalface_default.xml')
            
            # Efek blur wajah
            img_blurred = apply_face_blur(img_path, face_cascade)
            blurred_image_path = os.path.join('static', 'uploads', 'img-blurred.jpg')
            cv2.imwrite(blurred_image_path, img_blurred)

            return render_template('index.html', img=img_path, img2=blurred_image_path)
        
        elif 'detect_edges' in request.form:
            # Deteksi tepi gambar
            edges = detect_edges(img_path)
            edges_image_path = os.path.join('static', 'uploads', 'img-edges.jpg')
            cv2.imwrite(edges_image_path, edges)

            return render_template('index.html', img=img_path, img2=edges_image_path)
        
        elif 'segment_image' in request.form:
            # Menentukan jumlah cluster
            num_clusters = 3
            
            # Segmentasi Gambar
            segmented_image = cluster_segmentation(img_path, num_clusters)
            segmented_image_path = os.path.join('static', 'uploads', 'img-segmented.jpg')
            cv2.imwrite(segmented_image_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

            return render_template('index.html', img=img_path, img2=segmented_image_path)
        
        elif 'add_moustache' in request.form:
            # Menambahkan kumis
            img_with_moustache = add_moustache_to_face(img_path)
            img_with_moustache_path = os.path.join('static', 'uploads', 'img-with-moustache.jpg')
            cv2.imwrite(img_with_moustache_path, img_with_moustache)

            return render_template('index.html', img=img_path, img2=img_with_moustache_path)
        
        elif 'reduce_noise' in request.form:
            # Mengurangi noise pada gambar
            img_smoothed = reduce_noise(img_path)
            smoothed_image_path = os.path.join('static', 'uploads', 'img-smoothed.jpg')
            cv2.imwrite(smoothed_image_path, img_smoothed)

            return render_template('index.html', img=img_path, img2=smoothed_image_path)
        
        elif 'apply_dilation' in request.form:
            # Menjalankan operasi Dilasi
            img_dilation = apply_dilation(img_path)
            dilation_image_path = os.path.join('static', 'uploads', 'img-dilation.jpg')
            cv2.imwrite(dilation_image_path, img_dilation)

            return render_template('index.html', img=img_path, img2=dilation_image_path)
        
        elif 'apply_erosion' in request.form:
            # Menjalankan operasi Erosi
            img_erosion = apply_erosion(img_path)
            erosion_image_path = os.path.join('static', 'uploads', 'img-erotion.jpg')
            cv2.imwrite(erosion_image_path, img_erosion)

            return render_template('index.html', img=img_path, img2=erosion_image_path)
        
        elif 'apply_opening' in request.form:
            # Menjalankan operasi Opening
            opened_image = apply_opening(img_path)
            opened_image_path = os.path.join('static', 'uploads', 'img-opened.jpg')
            cv2.imwrite(opened_image_path, opened_image)

            return render_template('index.html', img=img_path, img2=opened_image_path)
        
        elif 'apply_closing' in request.form:
            # Menjalankan operasi Closing
            img_closed = apply_closing(img_path)
            closed_image_path = os.path.join('static', 'uploads', 'img-closed.jpg')
            cv2.imwrite(closed_image_path, img_closed)

            return render_template('index.html', img=img_path, img2=closed_image_path)
        
        elif 'apply_linear' in request.form:
            # Menjalankan operasi interpolasi Linear
            img_linear = apply_linear(img_path)
            linear_image_path = os.path.join('static', 'uploads', 'img-linear.jpg')
            cv2.imwrite(linear_image_path, img_linear)

            return render_template('index.html', img=img_path, img2=linear_image_path)
        
        elif 'apply_cubic' in request.form:
            # Menjalankan operasi Cubic
            img_cubic = apply_cubic(img_path)
            cubic_image_path = os.path.join('static', 'uploads', 'img-cubic.jpg')
            cv2.imwrite(cubic_image_path, img_cubic)

            return render_template('index.html', img=img_path, img2=cubic_image_path)
        
        elif 'salt_pepper_noise' in request.form:
            # Menambahkan salt and pepper noise
            salt_prob = 0.01
            pepper_prob = 0.01  
            img_with_noise = apply_salt_and_pepper_noise(img_path, salt_prob, pepper_prob)
            img_with_noise_path = os.path.join('static', 'uploads', 'img-with-noise.jpg')
            cv2.imwrite(img_with_noise_path, img_with_noise)

            return render_template('index.html', img=img_path, img2=img_with_noise_path)
        
        elif 'remove_salt_pepper_noise' in request.form:
            # Applying Rank-Order Filtering to remove salt and pepper noise with default parameters
            window_size = 3  # You can adjust this value if needed
            rank = window_size * window_size // 2  # Median rank for a square window
            img_filtered = remove_salt_and_pepper_noise_rank_order(img_path, window_size, rank)
            img_filtered_path = os.path.join('static', 'uploads', 'img-filtered.jpg')
            cv2.imwrite(img_filtered_path, img_filtered)

            return render_template('index.html', img=img_path, img2=img_filtered_path)



        
    return render_template('index.html')

if __name__ == '__main__': 
    app.run(debug=True,port=8001)
