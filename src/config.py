# src/config.py
import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "Intel Image Classification API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for classifying images into 6 categories"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Model Settings
    MODEL_PATH: str = "src/models/model_1_simple.keras"
    IMAGE_SIZE: tuple = (150, 150)
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "api.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

