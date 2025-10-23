# src/preprocessing.py
import numpy as np
from PIL import Image
import logging
from typing import Tuple
from src.config import settings
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Image preprocessing class
    Handles all image transformations needed for model inference
    """
    
    def __init__(self, target_size: Tuple[int, int] = settings.IMAGE_SIZE):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target size for resizing images (height, width)
        """
        self.target_size = target_size
        logger.info(f"ImagePreprocessor initialized with target size: {target_size}")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model inference
        
        Steps:
        1. Convert to RGB (if needed)
        2. Resize to target size
        3. Convert to array
        4. Normalize pixel values to [0, 1]
        5. Add batch dimension
        
        Args:
            image: PIL Image object
        
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.target_size, Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            
            # Normalize pixel values to [0, 1]
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            logger.debug(f"Preprocessed image shape: {img_array.shape}")
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def preprocess_batch(self, images: list[Image.Image]) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of PIL Image objects
        
        Returns:
            Batch of preprocessed images
        """
        preprocessed = [self.preprocess(img)[0] for img in images]
        return np.array(preprocessed)
    
    def validate_image(self, image: Image.Image) -> bool:
        """
        Validate image properties
        
        Args:
            image: PIL Image object
        
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Check if image is valid
            if image is None:
                return False
            
            # Check minimum size
            if image.size[0] < 32 or image.size[1] < 32:
                logger.warning(f"Image too small: {image.size}")
                return False
            
            # Check maximum size (prevent memory issues)
            if image.size[0] > 4096 or image.size[1] > 4096:
                logger.warning(f"Image too large: {image.size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False

