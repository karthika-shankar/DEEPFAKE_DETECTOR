# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)
CORS(app)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model (update path to where your model file is)
#MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deepfake_detection_MNV2_model_2c.h5')
try:
    MODEL = load_model('deepfake_detection_MNV2_model_finetuned.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    MODEL = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize the image to match the model's input size
    img = img.resize((96, 96))  # Ensure this matches your model's input shape
    
    # Convert the image to a NumPy array and normalize pixel values
    img_array = np.array(img) / 255.0
    
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def analyze_image_with_model(image_path):
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)
        
        # Make prediction using the model
        prediction = MODEL.predict(img_array)
        
        # Process prediction result
        confidence_score = float(prediction[0][0])
        is_deepfake = confidence_score > 0.5
        
        return {
            'is_deepfake': is_deepfake,
            'confidence': confidence_score
        }
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return None



@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check if model is loaded
            if MODEL is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            # Analyze the image
            result = analyze_image_with_model(filepath)
            
            # Clean up - remove the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if result is None:
                return jsonify({'error': 'Analysis failed'}), 500
                
            return jsonify({'result': result})
            
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
