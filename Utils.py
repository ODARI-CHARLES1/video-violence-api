import os
import uuid
from werkzeug.utils import secure_filename
import logging

class FileUtils:
    def __init__(self, upload_folder='uploads', allowed_extensions=None):
        if allowed_extensions is None:
            allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
        
        self.upload_folder = upload_folder
        self.allowed_extensions = allowed_extensions
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(upload_folder, exist_ok=True)
    
    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def save_uploaded_file(self, file):
        """Save uploaded file with secure filename"""
        if file and self.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(self.upload_folder, unique_filename)
            file.save(file_path)
            self.logger.info(f"File saved: {file_path}")
            return file_path
        return None
    
    def cleanup_file(self, file_path):
        """Remove temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"File cleaned up: {file_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning up file: {str(e)}")