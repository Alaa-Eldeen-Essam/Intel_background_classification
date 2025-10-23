# src/model_loader.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from typing import Dict, List
import os

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Model loader and inference class
    """
    
    def __init__(self, model_path: str):
        """
        Initialize model loader
        
        Args:
            model_path: Path to the trained model (.h5 file)
        """
        self.model_path = model_path
        self.model = None
        self.classes = [
            'buildings',
            'forest',
            'glacier',
            'mountain',
            'sea',
            'street'
        ]
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path ,compile=False, safe_mode=False)
            logger.info(f"Model loaded successfully!")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_classes(self) -> List[str]:
        """Get list of class names"""
        return self.classes
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Make prediction on preprocessed image
        
        Args:
            image: Preprocessed image array (batch_size, height, width, channels)
        
        Returns:
            Dictionary with class names and confidence scores
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Make prediction
            predictions = self.model.predict(image, verbose=0)
            
            # Get probabilities for first image in batch
            probs = predictions[0]
            
            # Create dictionary of class: confidence
            results = {
                class_name: float(prob)
                for class_name, prob in zip(self.classes, probs)
            }
            
            # Sort by confidence (descending)
            results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_top_k(self, image: np.ndarray, k: int = 3) -> Dict[str, float]:
        """
        Get top-k predictions
        
        Args:
            image: Preprocessed image array
            k: Number of top predictions to return
        
        Returns:
            Dictionary with top-k class names and scores
        """
        all_predictions = self.predict(image)
        return dict(list(all_predictions.items())[:k])

