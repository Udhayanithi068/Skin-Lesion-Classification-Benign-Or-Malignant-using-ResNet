"""
Utility functions for the Skin Lesion Classification API
"""

import os
import uuid
import aiofiles
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
from PIL import Image
import secrets

from database import get_db_session, SystemLog

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def save_uploaded_file(file: UploadFile, subfolder: str = "uploads") -> str:
    """
    Save uploaded file to disk
    
    Args:
        file: FastAPI UploadFile object
        subfolder: Subfolder within uploads directory
        
    Returns:
        File path where the file was saved
    """
    try:
        # Create directory if it doesn't exist
        upload_dir = Path("uploads") / subfolder
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = upload_dir / unique_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"File saved: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"session_{uuid.uuid4().hex[:16]}_{int(datetime.now().timestamp())}"

def validate_image_file(file: UploadFile) -> bool:
    """
    Validate uploaded image file
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check content type
        if not file.content_type or not file.content_type.startswith('image/'):
            return False
        
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        if file.filename:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                return False
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if hasattr(file, 'size') and file.size > max_size:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return False

def validate_and_resize_image(image_path: str, max_size: tuple = (1024, 1024)) -> bool:
    """
    Validate and optionally resize image
    
    Args:
        image_path: Path to image file
        max_size: Maximum allowed dimensions (width, height)
        
    Returns:
        True if valid/processed, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Check if image is valid
            img.verify()
            
            # Reopen for processing (verify closes the file)
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    img.save(image_path, optimize=True, quality=85)
                    logger.info(f"Resized image: {image_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating/resizing image {image_path}: {e}")
        return False

async def log_system_event(
    level: str,
    message: str,
    component: Optional[str] = None,
    user_id: Optional[str] = None,
    additional_data: Optional[dict] = None
):
    """
    Log system event to database
    
    Args:
        level: Log level (INFO, WARNING, ERROR)
        message: Log message
        component: Component name
        user_id: User ID (optional)
        additional_data: Additional data as dict
    """
    try:
        db = get_db_session()
        
        system_log = SystemLog(
            log_level=level,
            message=message,
            component=component,
            user_id=user_id,
            additional_data=additional_data
        )
        
        db.add(system_log)
        db.commit()
        db.close()
        
    except Exception as e:
        logger.error(f"Failed to log system event: {e}")

def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Clean up old files from directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files in hours
    """
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        files_deleted = 0
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    files_deleted += 1
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
        
        if files_deleted > 0:
            logger.info(f"Cleaned up {files_deleted} old files from {directory}")
            
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")

def get_file_info(file_path: str) -> dict:
    """
    Get information about a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)
        stat = path.stat()
        
        info = {
            'filename': path.name,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': path.suffix.lower()
        }
        
        # If it's an image, get dimensions
        if info['extension'] in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            try:
                with Image.open(file_path) as img:
                    info['width'] = img.width
                    info['height'] = img.height
                    info['mode'] = img.mode
                    info['format'] = img.format
            except Exception:
                pass
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return {}

def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)

def format_confidence_level(confidence: float) -> str:
    """
    Format confidence level as human-readable text
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Human-readable confidence level
    """
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.8:
        return "High" 
    elif confidence >= 0.7:
        return "Moderate-High"
    elif confidence >= 0.6:
        return "Moderate"
    elif confidence >= 0.5:
        return "Moderate-Low"
    else:
        return "Low"

def calculate_risk_level(prediction: str, confidence: float) -> dict:
    """
    Calculate risk level based on prediction and confidence
    
    Args:
        prediction: Model prediction ('benign' or 'malignant')
        confidence: Confidence score
        
    Returns:
        Dictionary with risk information
    """
    if prediction.lower() == 'malignant':
        if confidence >= 0.8:
            risk_level = "High Risk"
            recommendation = "Seek immediate medical attention"
            color = "#FF4444"
        elif confidence >= 0.6:
            risk_level = "Moderate-High Risk" 
            recommendation = "Consult a dermatologist soon"
            color = "#FF7744"
        else:
            risk_level = "Uncertain - Moderate Risk"
            recommendation = "Consider professional evaluation"
            color = "#FFAA44"
    else:  # benign
        if confidence >= 0.8:
            risk_level = "Low Risk"
            recommendation = "Continue regular skin monitoring"
            color = "#44AA44"
        elif confidence >= 0.6:
            risk_level = "Low-Moderate Risk"
            recommendation = "Monitor for changes, consider check-up"
            color = "#77AA44"
        else:
            risk_level = "Uncertain"
            recommendation = "Professional evaluation recommended"
            color = "#AAAA44"
    
    return {
        'risk_level': risk_level,
        'recommendation': recommendation,
        'color': color,
        'confidence_text': format_confidence_level(confidence)
    }

class FileManager:
    """File management utilities"""
    
    def __init__(self, base_dir: str = "uploads"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_file(self, file: UploadFile, subfolder: str = "") -> str:
        """Save uploaded file"""
        return await save_uploaded_file(file, subfolder)
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    def cleanup_old_files(self, subfolder: str = "", max_age_hours: int = 24):
        """Clean up old files"""
        directory = self.base_dir / subfolder if subfolder else self.base_dir
        cleanup_old_files(str(directory), max_age_hours)

# Global file manager instance
file_manager = FileManager()