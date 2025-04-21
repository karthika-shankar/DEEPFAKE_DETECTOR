# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

app = Flask(__name__)
CORS(app)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Create necessary folders if they don't exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the models
try:
    # Dynamically construct the path to the models directory
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    image_model_path = os.path.join(base_dir, 'model', 'deepfake_detection_MNV2_model_2c.h5')
    video_model_path = os.path.join(base_dir, 'model', 'deepfake_detector_V.h5')

    # Load the models
    IMAGE_MODEL = tf.keras.models.load_model(image_model_path)
    VIDEO_MODEL = tf.keras.models.load_model(video_model_path)
    # Load the face detection model
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    IMAGE_MODEL = None
    VIDEO_MODEL = None
    FACE_CASCADE = None


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def detect_face(image):
    """Detect faces in an image and return the coordinates of the largest face."""
    if isinstance(image, str):  # If image is a file path
        img = cv2.imread(image)
    else:  # If image is already a numpy array
        img = image
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    return largest_face

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Detect face
    face_rect = detect_face(img)
    
    if face_rect is None:
        # If no face detected, use the original image but still resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (96, 96))
    else:
        # Extract and process the face
        x, y, w, h = face_rect
        
        # Add some margin (20% of the face size)
        margin = int(w * 0.2)
        x_with_margin = max(0, x - margin)
        y_with_margin = max(0, y - margin)
        w_with_margin = min(img.shape[1] - x_with_margin, w + 2 * margin)
        h_with_margin = min(img.shape[0] - y_with_margin, h + 2 * margin)
        
        # Extract face with margin
        face_img = img[y_with_margin:y_with_margin + h_with_margin, 
                       x_with_margin:x_with_margin + w_with_margin]
        
        # Convert to RGB and resize
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(face_img_rgb, (96, 96))
    
    # Convert to numpy array and normalize
    img_array = img_to_array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def analyze_image_with_model(image_path):
    try:
        # Check if face is detected
        img = cv2.imread(image_path)
        face_rect = detect_face(img)
        
        if face_rect is None:
            return {'is_deepfake': -1, 'confidence': 0, 'message': 'No face detected in the image'}
        
        # Preprocess the image with the detected face
        img_array = preprocess_image(image_path)
        
        # Make prediction
        prediction = IMAGE_MODEL.predict(img_array)
        confidence = float(prediction[0][0])
        is_deepfake = int(confidence > 0.5)
        
        return {'is_deepfake': is_deepfake, 'confidence': confidence, 'message': 'Face analyzed successfully'}
    
    except Exception as e:
        raise Exception(f"Failed to analyze the image: {str(e)}")

def preprocess_video_frame(frame):
    # Detect face in the frame
    face_rect = detect_face(frame)
    
    if face_rect is None:
        return None
    
    # Extract and process the face
    x, y, w, h = face_rect
    
    # Add some margin (20% of the face size)
    margin = int(w * 0.2)
    x_with_margin = max(0, x - margin)
    y_with_margin = max(0, y - margin)
    w_with_margin = min(frame.shape[1] - x_with_margin, w + 2 * margin)
    h_with_margin = min(frame.shape[0] - y_with_margin, h + 2 * margin)
    
    # Extract face with margin
    face_img = frame[y_with_margin:y_with_margin + h_with_margin, 
                    x_with_margin:x_with_margin + w_with_margin]
    
    # Convert to RGB and resize
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(face_img_rgb, (224, 224))  # Resize to VIDEO_MODEL input size
    
    # Convert to numpy array and normalize
    img_array = img_to_array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def analyze_video_frame(frame):
    # Preprocess the frame to extract face
    processed_frame = preprocess_video_frame(frame)
    
    # If no face detected, return None
    if processed_frame is None:
        return None
    
    # Make prediction
    prediction = VIDEO_MODEL.predict(processed_frame)
    return float(prediction[0][0])

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    deepfake_scores = []
    frame_count = 0
    processed_frames = 0
    sample_rate = 5  # Process every 5th frame to improve performance
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every nth frame
        if frame_count % sample_rate == 0:
            score = analyze_video_frame(frame)
            if score is not None:  # Only count frames where a face was detected
                deepfake_scores.append(score)
                processed_frames += 1
                
        frame_count += 1
    
    cap.release()
    
    if processed_frames == 0:
        return {'is_deepfake': -1, 'confidence': 0, 'message': 'No faces found in the video'}
    
    confidence = sum(deepfake_scores) / processed_frames
    is_deepfake = int(confidence > 0.5)
    return {
        'is_deepfake': is_deepfake, 
        'confidence': confidence, 
        'message': f'Analyzed {processed_frames} frames with faces out of {frame_count} total frames'
    }

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400
    
    if IMAGE_MODEL is None or FACE_CASCADE is None:
        return jsonify({'error': 'Required models not loaded'}), 500
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        result = analyze_image_with_model(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid file type. Please upload an MP4, AVI, or MOV video.'}), 400
    
    if VIDEO_MODEL is None or FACE_CASCADE is None:
        return jsonify({'error': 'Required models not loaded'}), 500
    
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)
    
    try:
        result = process_video(video_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)