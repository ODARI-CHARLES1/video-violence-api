import cv2
import numpy as np
import logging
from typing import List, Tuple

class VideoProcessor:
    def __init__(self, target_size=(128, 128), frame_interval=10):
        self.target_size = target_size
        self.frame_interval = frame_interval
        self.logger = logging.getLogger(__name__)
    
    def extract_frames(self, video_path: str, max_frames: int = 50) -> List[np.ndarray]:
        """Extract frames from video at regular intervals"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Video info: {total_frames} frames, {fps} FPS")
            
            frame_count = 0
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    processed_frame = self.preprocess_frame(frame)
                    frames.append(processed_frame)
                
                frame_count += 1
            
            cap.release()
            self.logger.info(f"Extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess individual frame for model input"""
        frame = cv2.resize(frame, self.target_size)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features from frames for model prediction"""
        if not frames:
            raise ValueError("No frames to process")
        
        # Convert list of frames to numpy array
        features = np.array(frames)
        
        # If we have multiple frames, we can use different strategies:
        # 1. Use all frames (requires 4D input: batch, frames, height, width, channels)
        # 2. Use average of frames
        # 3. Use specific frame selection
        
        # For this implementation, we'll use the average frame
        if len(features) > 1:
            avg_frame = np.mean(features, axis=0)
            features = np.expand_dims(avg_frame, axis=0)
        else:
            features = np.expand_dims(features[0], axis=0)
        
        return features