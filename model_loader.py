import tensorflow as tf
import keras
import pickle
import numpy as np
import logging

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load the pre-trained Keras model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info("Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, features):
        """Make prediction on extracted features"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            if len(features.shape) == 3:
                features = np.expand_dims(features, axis=0)
            
            prediction = self.model.predict(features, verbose=0)
            return prediction[0][0] 
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise