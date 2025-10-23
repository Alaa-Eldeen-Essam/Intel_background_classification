# src/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any
import io
from PIL import Image
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


# Import custom modules
from src.model_loader import ModelLoader
from src.preprocessing import ImagePreprocessor
from src.config import settings
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessor
model_loader = None
preprocessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI
    Handles startup and shutdown events
    """
    # Startup: Load model and preprocessor
    global model_loader, preprocessor
    
    logger.info("Starting application...")
    logger.info("Loading model and preprocessor...")
    
    try:
        model_loader = ModelLoader(model_path=settings.MODEL_PATH)
        preprocessor = ImagePreprocessor(target_size=(150, 150))
        logger.info("Model and preprocessor loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(f"Response: {response.status_code} - Time: {process_time:.3f}s")
    
    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# ============================================================================
# ENDPOINTS
# ============================================================================
# Serve frontend static files
import os

frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML file."""
    with open(os.path.join(frontend_dir, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# @app.get("/", tags=["Root"])
# async def root():
#     """Root endpoint - API information"""
#     return {
#         "message": "Intel Image Classification API",
#         "version": "1.0.0",
#         "endpoints": {
#             "health": "/health",
#             "predict": "/predict",
#             "docs": "/docs"
#         }
#     }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns API status and model loading status
    """
    model_loaded = model_loader is not None and model_loader.is_loaded()
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
        "classes": model_loader.get_classes() if model_loaded else []
    }

@app.post("/predict", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Image classification endpoint
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        JSON with predicted class, confidence, and all predictions
    """
    
    # Validate model is loaded
    if model_loader is None or not model_loader.is_loaded():
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type in ["image/jpeg", "image/jpg", "image/png"]:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG are supported."
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Log image info
        logger.info(f"Processing image: {file.filename} - Size: {image.size}, Mode: {image.mode}")
        
        # Preprocess image
        processed_image = preprocessor.preprocess(image)
        
        # Get predictions
        predictions = model_loader.predict(processed_image)
        
        # Get top class
        top_class = max(predictions.items(), key=lambda x: x[1])
        
        # Log prediction
        logger.info(f"Prediction: {top_class[0]} with confidence {top_class[1]:.4f}")
        
        # Prepare response
        response = {
            "filename": file.filename,
            "class": top_class[0],
            "confidence": float(top_class[1]),
            "predictions": {k: float(v) for k, v in predictions.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        await file.close()

@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch image classification endpoint
    
    Args:
        files: List of image files (JPEG, PNG)
    
    Returns:
        JSON with predictions for all images
    """
    
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type in ["image/jpeg", "image/jpg", "image/png"]:
                results.append({
                    "filename": file.filename,
                    "error": f"Invalid file type: {file.content_type}"
                })
                continue
            
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            processed_image = preprocessor.preprocess(image)
            
            # Get predictions
            predictions = model_loader.predict(processed_image)
            top_class = max(predictions.items(), key=lambda x: x[1])
            
            results.append({
                "filename": file.filename,
                "class": top_class[0],
                "confidence": float(top_class[1]),
                "predictions": {k: float(v) for k, v in predictions.items()}
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        
        finally:
            await file.close()
    
    return {
        "total": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes", tags=["Information"])
async def get_classes():
    """Get list of supported classes"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": model_loader.get_classes(),
        "total": len(model_loader.get_classes())
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level="info"
    )
