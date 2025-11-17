from tensorflow import keras
from keras.models import load_model
import numpy as np
import logging
import os

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            self.model = load_model(self.model_path)
            self.logger.info("Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False

    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not loaded.")
        if isinstance(features, list):
            features = np.array(features)
        if len(features.shape) == 3:
            features = np.expand_dims(features, axis=0)
        pred = self.model.predict(features, verbose=0)
        return float(pred[0][0])
