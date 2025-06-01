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
import gdown 

app = Flask(__name__)

# --- Konfigurasi Aplikasi ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['covid', 'normal', 'virus']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# --- Konfigurasi Model & Google Drive ---
GOOGLE_DRIVE_FILE_ID = os.environ.get('MODEL_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE')

MODEL_DIR = 'models'
MODEL_FILENAME = 'model_opt-adam_lr-1e-05_bs-32.h5'
LOCAL_MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

model = None

def download_model_if_needed(file_id, save_dir, filename):
    """Downloads the model from Google Drive if it doesn't exist locally."""
    full_path = os.path.join(save_dir, filename)
    if not os.path.exists(full_path):
        print(f"* Model weights not found at {full_path}. Downloading from Google Drive...")
        os.makedirs(save_dir, exist_ok=True)
        try:
            gdown.download(id=file_id, output=full_path, quiet=False)
            print(f"* Model weights downloaded successfully to {full_path}")
            return True
        except Exception as e:
            print(f"* Error downloading model weights: {e}")
            return False
    else:
        print(f"* Model weights found at {full_path}.")
        return True

def build_and_load_model(num_classes, weights_path):
    """Builds the model architecture and loads weights."""
    base_model = ConvNeXtTiny(weights=None, include_top=False, input_shape=(224, 224, 3))

    compiled_model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    print(f"* Attempting to load weights from: {weights_path}")
    compiled_model.load_weights(weights_path)
    print("* Weights loaded into model structure successfully.")
    return compiled_model

print("* Initializing model loading process...")
if GOOGLE_DRIVE_FILE_ID == 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE':
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! WARNING: GOOGLE_DRIVE_FILE_ID is not set. Please set it via         !!!")
    print("!!! environment variable 'MODEL_FILE_ID' or directly in the code.       !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

if download_model_if_needed(GOOGLE_DRIVE_FILE_ID, MODEL_DIR, MODEL_FILENAME):
    try:
        model = build_and_load_model(num_classes=len(CLASS_NAMES), weights_path=LOCAL_MODEL_WEIGHTS_PATH)
        print("* Model built and weights loaded successfully.")
    except Exception as e:
        print(f"* Error building model or loading weights: {e}")
        model = None 
else:
    print("* Model download failed. Model will not be available.")
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
        return "Model not loaded. Please check server logs or wait for download to complete.", 503 # Kode 503 Service Unavailable

    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

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
        finally:

            if os.path.exists(img_path):
                print(f"* Deleting uploaded file: {img_path}")
                os.remove(img_path)
                pass


    return redirect(url_for('index'))

if __name__ == '__main__':
    print("* Starting Flask development server...")
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)