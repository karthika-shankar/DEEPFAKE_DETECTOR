from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from werkzeug.utils import secure_filename
from io import BytesIO
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import logging

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model (update path to where your model file is)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deepfake_detection_MNV2_model_finetuned.h5')
try:
    MODEL = load_model(MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    MODEL = None

# Image Preprocessing
def preprocess_image(image_bytes):
    # Open the image
    img = Image.open(BytesIO(image_bytes))
    
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

# Analyze image with model
def analyze_image_with_model(image_bytes):
    try:
        # Preprocess the image
        img_array = preprocess_image(image_bytes)
        
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
        logging.error(f"Error analyzing image: {str(e)}")
        return None

# Endpoint to analyze image from URL
@app.route('/api/analyze/from_url', methods=['POST'])
def analyze_image_from_url():
    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        # Fetch the image from the URL
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image'}), 400
        
        # Process the image
        result = analyze_image_with_model(img_response.content)
        
        if result is None:
            return jsonify({'error': 'Analysis failed'}), 500
        
        return jsonify({'result': result}), 200
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
