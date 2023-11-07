import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from net import transform_image, prediction, grad_cam
from flask import Flask, jsonify, render_template, request, send_from_directory

template_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
template_dir = os.path.join(template_dir, 'Covid-19 Classification')
result_dir = os.path.join(template_dir, 'images', 'results')
download_img_path = os.path.join(template_dir, 'images', 'caches')
template_dir = os.path.join(template_dir, 'templates')

app = Flask(__name__, template_folder=template_dir, static_folder=result_dir)

# Only allow image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Page
@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        # Upload image
        file = request.files['x-ray_image']

        if file is None or file.filename == "":
            error_message = ({'Error': 'No file found'})
            return render_template('index.html', message=error_message)
        
        if not allowed_file(file.filename):
            error_message = ({'Error': 'Format not supported'})
            return render_template('index.html', message=error_message)
        
        # Save the upload image
        image_path = "./images/" + file.filename
        file.save(image_path)

        try:
            # Preprocess
            tensor = transform_image(image_path)

            # Prediction
            logit, prob, predicted = prediction(tensor)

            # Print the probability
            table_data = [
                {'Class': 'COVID-19', 'Probability': logit[0]},
                {'Class': 'Normal', 'Probability': logit[1]},
                {'Class': 'Viral Pneumonia', 'Probability': logit[2]}
            ]

            data = ({'Predicted Class': predicted, 'Probability': prob})

            # Apply Grad-Cam to image
            img = grad_cam(image_path)

            # Save current date and time
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save as TIF image
            cache_img = predicted + "_" + current_datetime + ".tif"
            cache_image_path = "./images/caches/" + cache_img
            cv2.imwrite(cache_image_path, img)

            # Save as JPG image
            result_img = predicted + "_" + current_datetime + ".jpg"
            result_image_path = "./images/results/" + result_img
            cv2.imwrite(result_image_path, img)

            return render_template('index.html', image=result_img, download_img=cache_img, table_data=table_data, message=data)
    
        except:
            error_message = ({'Error': 'Error during prediction'})
            return render_template('index.html', message=error_message)

# Download image
@app.route('/download/<filename>')        
def download(filename):
    return send_from_directory(download_img_path, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(port=3000, debug=True)