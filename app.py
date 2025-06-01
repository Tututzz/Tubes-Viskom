import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import ConvNeXtTiny
app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/model_opt-adam_lr-1e-05_bs-32.h5' 
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (224, 224) 
CLASS_NAMES = ['covid', 'normal', 'virus'] 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def build_model(num_classes):
    base_model = ConvNeXtTiny(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

try:
    model = build_model(num_classes=3)  
    model.load_weights('model_opt-adam_lr-1e-05_bs-32.h5')  
    print("* Model loaded successfully")
except Exception as e:
    print(f"* Error loading model: {e}")
    model = None 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Loads and preprocesses an image for model prediction."""
    img = tf_image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

@app.route('/', methods=['GET'])
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if model is None:
        return "Model not loaded. Please check server logs.", 500

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        try:
            processed_image = preprocess_image(img_path)
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(np.max(predictions[0]) * 100) 

            return render_template('result.html',
                                   prediction=predicted_class_name,
                                   confidence=confidence,
                                   image_file=filename)
        except Exception as e:
            print(f"* Error during prediction: {e}")
            return f"Error processing image: {e}", 500
     

    return redirect(url_for('index')) 

if __name__ == '__main__':
    app.run(debug=True) 