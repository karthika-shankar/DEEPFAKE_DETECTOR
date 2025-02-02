import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deepfake_detection_MNV2_model_finetuned.h5')
try:
    MODEL = load_model(MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    MODEL = None

def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Detect faces using OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    
    # Crop the first detected face
    (x, y, w, h) = faces[0]
    face_image = img_array[y:y+h, x:x+w]
    
    # Resize the face image to match the model's input size
    face_image = cv2.resize(face_image, (96, 96))
    
    # Normalize pixel values
    face_image = face_image / 255.0
    
    # Add a batch dimension
    face_image = np.expand_dims(face_image, axis=0)
    
    return face_image

def analyze_image_with_model(image_bytes):
    if MODEL is None:
        raise ValueError("Model is not loaded")
    
    try:
        img_array = preprocess_image(image_bytes)
        prediction = MODEL.predict(img_array)
        confidence_score = float(prediction[0][0])
        is_deepfake = confidence_score > 0.5
        
        return {
            'confidence': confidence_score,
            'is_deepfake': is_deepfake
        }
    except ValueError as e:
        logging.error(f"Error analyzing image: {str(e)}")
        return {'error': str(e)}
    except Exception as e:
        logging.error(f"Error analyzing image: {str(e)}")
        return None

@app.route('/api/analyze/from_url', methods=['POST'])
def analyze_image_from_url():
    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image'}), 400
        
        result = analyze_image_with_model(img_response.content)
        
        if result is None:
            return jsonify({'error': 'Analysis failed'}), 500
        
        return jsonify({'result': result}), 200
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)