"""
Utility functions for the Skin Lesion Classification app
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import UploadFile

logger = logging.getLogger(__name__)

async def save_uploaded_file(file: UploadFile, folder: str) -> str:
    """
    Save uploaded file to specified folder
    
    Args:
        file: FastAPI UploadFile object
        folder: Folder name (temp, predictions, etc.)
        
    Returns:
        Path to saved file
    """
    try:
        # Create directory if it doesn't exist
        upload_dir = Path("uploads") / folder
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = upload_dir / unique_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

async def log_system_event(level: str, message: str, category: str, user_id: Optional[int] = None):
    """
    Log system events (simplified version)
    
    Args:
        level: Log level (INFO, WARNING, ERROR)
        message: Log message
        category: Event category
        user_id: Optional user ID
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {level} - {category}: {message}"
        if user_id:
            log_entry += f" (User: {user_id})"
        
        logger.info(log_entry)
        
        # You can extend this to save to database if needed
        
    except Exception as e:
        logger.error(f"Failed to log event: {e}")

def cleanup_temp_files(max_age_hours: int = 24):
    """
    Clean up temporary files older than specified hours
    
    Args:
        max_age_hours: Maximum age in hours before deletion
    """
    try:
        temp_dir = Path("uploads/temp")
        if not temp_dir.exists():
            return
        
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        for file_path in temp_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"Cleaned up temp file: {file_path}")
                
    except Exception as e:
        logger.error(f"Failed to cleanup temp files: {e}")

def validate_image_format(file_path: str) -> bool:
    """
    Validate image file format
    
    Args:
        file_path: Path to image file
        
    Returns:
        True if valid image format
    """
    try:
        from PIL import Image
        
        with Image.open(file_path) as img:
            # Basic validation
            if img.format.lower() in ['jpeg', 'jpg', 'png', 'webp']:
                return True
            else:
                return False
                
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False