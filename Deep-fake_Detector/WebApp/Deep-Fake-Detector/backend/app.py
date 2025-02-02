# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

app = Flask(__name__)
CORS(app)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
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
    
    # Convert PIL Image to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image. Please upload an image containing a clear face.")
    
    # Get the largest face in case of multiple detections
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    # Add padding around the face (20% of face size)
    padding_x = int(w * 0.2)
    padding_y = int(h * 0.2)
    
    # Calculate new coordinates with padding
    start_x = max(x - padding_x, 0)
    start_y = max(y - padding_y, 0)
    end_x = min(x + w + padding_x, img_array.shape[1])
    end_y = min(y + h + padding_y, img_array.shape[0])
    
    # Crop the face region with padding
    face_img = img_array[start_y:end_y, start_x:end_x]
    
    # Convert BGR to RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    face_img = cv2.resize(face_img, (96, 96))
    
    # Convert to float and normalize
    face_img = img_to_array(face_img)
    face_img = face_img / 255.0
    
    # Add batch dimension
    face_img = np.expand_dims(face_img, axis=0)
    
    return face_img

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
    except ValueError as e:
        # Pass along face detection errors
        raise e
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        raise Exception("Failed to analyze the image. Please try again with a different image.")

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
            
            try:
                # Analyze the image
                result = analyze_image_with_model(filepath)
                
                return jsonify({'result': result})
            except ValueError as e:
                # Handle face detection errors
                return jsonify({'error': str(e)}), 400
            except Exception as e:
                # Handle other analysis errors
                return jsonify({'error': str(e)}), 500
            finally:
                # Clean up - remove the uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
            
        return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)