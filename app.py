from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from model_loader import ModelLoader
from video_processor import VideoProcessor
from utils import FileUtils

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model (1).pkl'  # Update with your actual model path

# Initialize components
file_utils = FileUtils(upload_folder=UPLOAD_FOLDER)
model_loader = ModelLoader(MODEL_PATH)
video_processor = VideoProcessor()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.before_first_request
def initialize_model():
    """Load model before first request"""
    if not model_loader.load_model():
        logger.error("Failed to load model")
        raise RuntimeError("Model loading failed")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Violence Detection API is running'
    })

@app.route('/predict', methods=['POST'])
def predict_violence():
    """Main prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        video_path = file_utils.save_uploaded_file(file)
        if not video_path:
            return jsonify({'error': 'Invalid file type'}), 400
        
        try:
            logger.info(f"Processing video: {video_path}")
            frames = video_processor.extract_frames(video_path)
            
            if not frames:
                return jsonify({'error': 'Could not extract frames from video'}), 400
            
            features = video_processor.extract_features(frames)
            
            prediction = model_loader.predict(features)
            
            is_violent = prediction > 0.5
            confidence = float(prediction) if is_violent else float(1 - prediction)
            
            response = {
                'violent': bool(is_violent),
                'confidence': round(confidence, 4),
                'prediction_score': float(prediction),
                'frames_processed': len(frames),
                'message': 'Video contains violent content' if is_violent else 'Video appears to be non-violent'
            }
            
            return jsonify(response)
            
        finally:
            file_utils.cleanup_file(video_path)
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple videos"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        for file in files:
            if file and file_utils.allowed_file(file.filename):
                video_path = file_utils.save_uploaded_file(file)
                if video_path:
                    try:
                        frames = video_processor.extract_frames(video_path)
                        if frames:
                            features = video_processor.extract_features(frames)
                            prediction = model_loader.predict(features)
                            
                            is_violent = prediction > 0.5
                            confidence = float(prediction) if is_violent else float(1 - prediction)
                            
                            results.append({
                                'filename': file.filename,
                                'violent': bool(is_violent),
                                'confidence': round(confidence, 4),
                                'prediction_score': float(prediction)
                            })
                    except Exception as e:
                        results.append({
                            'filename': file.filename,
                            'error': str(e)
                        })
                    finally:
                        file_utils.cleanup_file(video_path)
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    if model_loader.load_model():
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Cannot start server: Model loading failed")