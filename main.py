"""
FastAPI Main Application for Skin Lesion Classification
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Import your modules
from database import (
    get_db, Prediction, User, TempCapture, DatabaseManager,
    get_db_session
)
from hf_vision_client import hf_vision_client
from utils import save_uploaded_file, generate_session_id
from exceptions import *
from auth import get_current_user, get_optional_user, AuthHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
try:
    DatabaseManager.init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")

# Create FastAPI app
app = FastAPI(
    title="Skin Lesion Classification API",
    description="AI-powered skin lesion analysis using Hugging Face vision models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(SkinLesionException, skin_lesion_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Mount static files (if directory exists)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Auth handler
auth_handler = AuthHandler()

# Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web application"""
    # Try to find the HTML file in different locations
    possible_paths = [
        "static/index.html",
        "templates/index.html", 
        "index.html"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return FileResponse(path)
    
    # Fallback HTML if no file found
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Skin Lesion Classification</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🩺 Skin Lesion Classification API</h1>
            <p>AI-powered skin lesion analysis system</p>
            
            <div class="upload-area">
                <h3>Upload Image for Analysis</h3>
                <p>Select a skin lesion image to analyze</p>
                <input type="file" id="imageFile" accept="image/*">
                <br><br>
                <button class="btn" onclick="uploadImage()">Analyze Image</button>
            </div>
            
            <div id="results"></div>
            
            <hr>
            <p><strong>API Endpoints:</strong></p>
            <ul>
                <li><a href="/docs">📋 API Documentation</a></li>
                <li><a href="/system/info">ℹ️ System Info</a></li>
                <li><strong>POST /predict/temp</strong> - Analyze image (no auth required)</li>
                <li><strong>POST /predict</strong> - Analyze image (requires auth)</li>
            </ul>
        </div>
        
        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    document.getElementById('results').innerHTML = '<p>🔄 Analyzing image...</p>';
                    
                    const response = await fetch('/predict/temp', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('results').innerHTML = `
                            <h3>📊 Analysis Results</h3>
                            <p><strong>Prediction:</strong> ${result.prediction}</p>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Analysis:</strong> ${result.hf_analysis || 'Analysis completed'}</p>
                            <p><strong>Processing Time:</strong> ${result.processing_time?.toFixed(2)}s</p>
                            <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px;">
                                <strong>⚠️ Medical Disclaimer:</strong> This is an AI screening tool for educational purposes only. 
                                Always consult healthcare professionals for proper medical diagnosis and treatment.
                            </div>
                        `;
                    } else {
                        document.getElementById('results').innerHTML = `<p style="color: red;">❌ Error: ${result.detail}</p>`;
                    }
                } catch (error) {
                    document.getElementById('results').innerHTML = `<p style="color: red;">❌ Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """)

@app.get("/app", response_class=HTMLResponse)
async def web_app():
    """Alternative endpoint for web app"""
    return await root()

@app.post("/predict/temp")
async def predict_temp(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    Temporary prediction endpoint - no authentication required
    Perfect for testing and anonymous users
    """
    try:
        # Validate the uploaded file
        validate_image_file(file)
        
        # Generate session ID if not provided
        if not session_id:
            session_id = generate_session_id()
        
        # Save uploaded file to temp directory
        file_path = await save_uploaded_file(file, "temp")
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Analyze image using Hugging Face
        logger.info(f"Starting analysis for temp session: {session_id}")
        hf_result = await hf_vision_client.analyze_skin_lesion(file_path)
        
        processing_time = time.time() - start_time
        
        # Prepare response with all relevant information
        result = {
            "prediction": hf_result["prediction"],
            "confidence": hf_result["confidence"],
            "probabilities": hf_result["probabilities"],
            "hf_analysis": hf_result.get("analysis", "Analysis completed"),
            "model_type": hf_result.get("model_type", "huggingface"),
            "processing_time": processing_time,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "technical_details": hf_result.get("technical_details", {})
        }
        
        # Log the successful analysis
        logger.info(f"Analysis completed: {hf_result['prediction']} ({hf_result['confidence']:.3f} confidence)")
        
        # Optionally save to temp_captures table for record keeping
        try:
            db = get_db_session()
            temp_capture = TempCapture(
                session_id=session_id,
                image_filename=file.filename or "unknown.jpg",
                image_path=file_path,
                prediction=hf_result["prediction"],
                confidence=hf_result["confidence"],
                model_used=hf_result.get("model_type", "huggingface"),
                is_processed=True,
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            db.add(temp_capture)
            db.commit()
            db.close()
        except Exception as e:
            logger.warning(f"Failed to save temp capture record: {e}")
        
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/predict")
async def predict_authenticated(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Authenticated prediction endpoint - saves to user's history
    """
    try:
        # Validate the uploaded file
        validate_image_file(file)
        
        # Save uploaded file to predictions directory
        file_path = await save_uploaded_file(file, "predictions")
        
        # Record start time
        start_time = time.time()
        
        # Analyze image using Hugging Face
        logger.info(f"Starting analysis for user: {current_user.username}")
        hf_result = await hf_vision_client.analyze_skin_lesion(file_path)
        
        processing_time = time.time() - start_time
        
        # Save prediction to database
        prediction_data = {
            "user_id": current_user.id,
            "image_filename": file.filename or "unknown.jpg",
            "image_path": file_path,
            "prediction": hf_result["prediction"],
            "confidence": hf_result["confidence"],
            "model_used": hf_result.get("model_type", "huggingface"),
            "processing_time": processing_time,
            "hf_analysis": hf_result.get("analysis"),
            "probabilities": hf_result["probabilities"],
            "technical_details": hf_result.get("technical_details")
        }
        
        saved_prediction = DatabaseManager.save_prediction(prediction_data)
        
        # Prepare response
        result = {
            "id": saved_prediction.id,
            "prediction": hf_result["prediction"],
            "confidence": hf_result["confidence"],
            "probabilities": hf_result["probabilities"],
            "hf_analysis": hf_result.get("analysis", "Analysis completed"),
            "model_type": hf_result.get("model_type", "huggingface"),
            "processing_time": processing_time,
            "timestamp": saved_prediction.timestamp.isoformat(),
            "technical_details": hf_result.get("technical_details", {})
        }
        
        logger.info(f"Analysis saved for user {current_user.username}: {saved_prediction.id}")
        return result
        
    except Exception as e:
        logger.error(f"Authenticated prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/predictions")
async def get_user_predictions(
    current_user: User = Depends(get_current_user),
    limit: int = 50
):
    """Get user's prediction history"""
    try:
        predictions = DatabaseManager.get_user_predictions(current_user.id, limit)
        
        # Convert to response format
        result = []
        for pred in predictions:
            result.append({
                "id": pred.id,
                "image_filename": pred.image_filename,
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "model_used": pred.model_used,
                "processing_time": pred.processing_time,
                "timestamp": pred.timestamp.isoformat(),
                "hf_analysis": pred.hf_analysis,
                "probabilities": pred.probabilities
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get user predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve prediction history")

@app.get("/system/info")
async def system_info():
    """Get system information and health check"""
    try:
        db_healthy = DatabaseManager.check_connection()
        hf_available = hf_vision_client.is_available()
        
        # Get some basic stats
        db = get_db_session()
        try:
            total_predictions = db.query(Prediction).count()
            total_users = db.query(User).count()
        except:
            total_predictions = 0
            total_users = 0
        finally:
            db.close()
        
        return {
            "status": "running",
            "version": "1.0.0",
            "database_healthy": db_healthy,
            "hf_api_available": hf_available,
            "statistics": {
                "total_predictions": total_predictions,
                "total_users": total_users
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/system/cleanup")
async def cleanup_temp_files():
    """Clean up expired temporary files (admin endpoint)"""
    try:
        cleaned_count = DatabaseManager.cleanup_temp_captures()
        return {
            "status": "success",
            "cleaned_files": cleaned_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "skin_lesion_classification"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🩺 Starting Skin Lesion Classification API")
    
    # Check database connection
    if DatabaseManager.check_connection():
        logger.info("✅ Database connection successful")
    else:
        logger.error("❌ Database connection failed")
    
    # Check Hugging Face API
    if hf_vision_client.is_available():
        logger.info("✅ Hugging Face API available")
    else:
        logger.warning("⚠️ Hugging Face API not configured")
    
    # Create directories
    directories = ["uploads/temp", "uploads/predictions", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Skin Lesion Classification API started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("👋 Shutting down Skin Lesion Classification API")
    
    # Cleanup temp files on shutdown
    try:
        DatabaseManager.cleanup_temp_captures()
        logger.info("✅ Temporary files cleaned up")
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )