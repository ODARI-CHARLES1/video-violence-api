from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from model_loader import ModelLoader
from video_processor import VideoProcessor
from utils import FileUtils

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_FOLDER = "uploads"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

file_utils = FileUtils(upload_folder=UPLOAD_FOLDER)
model_loader = ModelLoader(MODEL_PATH)
video_processor = VideoProcessor()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Violence Detection API is running"
    })

@app.route('/predict', methods=['POST'])
def predict_violence():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        video_path = file_utils.save_uploaded_file(file)
        if not video_path:
            return jsonify({"error": "Invalid file type"}), 400
        
        try:
            logger.info(f"Processing video: {video_path}")
            frames = video_processor.extract_frames(video_path)

            if not frames:
                return jsonify({"error": "Could not extract frames"}), 400
            
            features = video_processor.extract_features(frames)
            prediction = model_loader.predict(features)

            is_violent = prediction > 0.5
            confidence = float(prediction) if is_violent else float(1 - prediction)

            response = {
                "violent": bool(is_violent),
                "confidence": round(confidence, 4),
                "prediction_score": float(prediction),
                "frames_processed": len(frames),
                "message": "Video contains violent content" if is_violent else "Video appears to be non-violent"
            }

            return jsonify(response)

        finally:
            file_utils.cleanup_file(video_path)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({"error": "No files selected"}), 400
        
        results = []
        for file in files:
            if not file_utils.allowed_file(file.filename):
                results.append({"filename": file.filename, "error": "Invalid file type"})
                continue
            
            video_path = file_utils.save_uploaded_file(file)
            try:
                frames = video_processor.extract_frames(video_path)
                if not frames:
                    results.append({"filename": file.filename, "error": "Could not extract frames"})
                    continue

                features = video_processor.extract_features(frames)
                prediction = model_loader.predict(features)

                is_violent = prediction > 0.5
                confidence = float(prediction) if is_violent else float(1 - prediction)

                results.append({
                    "filename": file.filename,
                    "violent": bool(is_violent),
                    "confidence": round(confidence, 4),
                    "prediction_score": float(prediction)
                })

            except Exception as e:
                results.append({"filename": file.filename, "error": str(e)})

            finally:
                file_utils.cleanup_file(video_path)

        return jsonify({"results": results})

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": f"Batch prediction failed: {e}"}), 500

if __name__ == '__main__':
    if model_loader.load_model():
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Cannot start server: Model loading failed")
