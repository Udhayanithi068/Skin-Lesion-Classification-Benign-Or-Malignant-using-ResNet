"""
Custom exceptions and error handlers for Skin Lesion Classification API
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
from typing import Union
from datetime import datetime

logger = logging.getLogger(__name__)

class SkinLesionException(Exception):
    """Base exception class for skin lesion application"""
    def __init__(self, message: str, error_code: str = None, status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(message)

class ModelLoadError(SkinLesionException):
    """Exception raised when ML model fails to load"""
    def __init__(self, message: str = "Failed to load ML model"):
        super().__init__(message, "MODEL_LOAD_ERROR", 503)

class ModelPredictionError(SkinLesionException):
    """Exception raised when ML model prediction fails"""
    def __init__(self, message: str = "Model prediction failed"):
        super().__init__(message, "PREDICTION_ERROR", 500)

class ImageProcessingError(SkinLesionException):
    """Exception raised when image processing fails"""
    def __init__(self, message: str = "Image processing failed"):
        super().__init__(message, "IMAGE_PROCESSING_ERROR", 400)

class InvalidImageFormat(SkinLesionException):
    """Exception raised when image format is not supported"""
    def __init__(self, message: str = "Invalid image format"):
        super().__init__(message, "INVALID_IMAGE_FORMAT", 400)

class ImageTooLarge(SkinLesionException):
    """Exception raised when image file is too large"""
    def __init__(self, message: str = "Image file too large"):
        super().__init__(message, "IMAGE_TOO_LARGE", 413)

class DatabaseError(SkinLesionException):
    """Exception raised when database operations fail"""
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message, "DATABASE_ERROR", 500)

class AuthenticationError(SkinLesionException):
    """Exception raised when authentication fails"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTH_ERROR", 401)

class AuthorizationError(SkinLesionException):
    """Exception raised when authorization fails"""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, "AUTHORIZATION_ERROR", 403)

class OpenRouterError(SkinLesionException):
    """Exception raised when OpenRouter API fails"""
    def __init__(self, message: str = "OpenRouter API error"):
        super().__init__(message, "OPENROUTER_ERROR", 503)

class ValidationError(SkinLesionException):
    """Exception raised when data validation fails"""
    def __init__(self, message: str = "Validation failed"):
        super().__init__(message, "VALIDATION_ERROR", 422)

# Error response models
def create_error_response(
    error_code: str,
    message: str,
    details: Union[str, dict] = None,
    status_code: int = 500
) -> dict:
    """Create standardized error response"""
    response = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": status_code
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    return response

# Exception handlers
async def skin_lesion_exception_handler(request: Request, exc: SkinLesionException):
    """Handler for custom SkinLesion exceptions"""
    logger.error(f"SkinLesion error: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error_code=exc.error_code or "UNKNOWN_ERROR",
            message=exc.message,
            status_code=exc.status_code
        )
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code} error: {exc.detail} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            status_code=exc.status_code
        )
    )

async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handler for Starlette HTTP exceptions"""
    logger.warning(f"Starlette {exc.status_code} error: {exc.detail} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            status_code=exc.status_code
        )
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler for request validation errors"""
    logger.warning(f"Validation error: {exc.errors()} | Path: {request.url.path}")
    
    # Format validation errors
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        errors.append(f"{field}: {message}")
    
    return JSONResponse(
        status_code=422,
        content=create_error_response(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"validation_errors": errors},
            status_code=422
        )
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handler for unexpected exceptions"""
    error_id = f"error_{datetime.utcnow().timestamp()}"
    
    logger.error(
        f"Unexpected error [{error_id}]: {str(exc)} | Path: {request.url.path} | "
        f"Traceback: {traceback.format_exc()}"
    )
    
    # Log to system_logs table if possible
    try:
        from .utils import log_system_event
        await log_system_event(
            "ERROR",
            f"Unexpected error: {str(exc)}",
            "exception_handler",
            additional_data={
                "error_id": error_id,
                "path": str(request.url.path),
                "traceback": traceback.format_exc()
            }
        )
    except:
        pass  # Don't let logging errors crash the error handler
    
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please try again later.",
            details={"error_id": error_id},
            status_code=500
        )
    )

# Validation utilities
def validate_image_file(file):
    """Validate uploaded image file"""
    if not file:
        raise ValidationError("No file provided")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise InvalidImageFormat(f"File type '{file.content_type}' not supported. Please upload an image file.")
    
    # Check file size
    max_size = 10 * 1024 * 1024  # 10MB
    if hasattr(file, 'size') and file.size > max_size:
        raise ImageTooLarge("File size exceeds 10MB limit")
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if file.filename:
        file_ext = file.filename.lower().split('.')[-1]
        if f'.{file_ext}' not in allowed_extensions:
            raise InvalidImageFormat(f"File extension '.{file_ext}' not supported")

def validate_prediction_confidence(confidence: float):
    """Validate prediction confidence score"""
    if not isinstance(confidence, (int, float)):
        raise ValidationError("Confidence must be a number")
    
    if not 0.0 <= confidence <= 1.0:
        raise ValidationError("Confidence must be between 0.0 and 1.0")

def validate_user_input(data: dict, required_fields: list):
    """Validate user input data"""
    missing_fields = []
    invalid_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
        elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Specific field validations
    if 'email' in data:
        email = data['email'].strip()
        if not email or '@' not in email or '.' not in email.split('@')[-1]:
            invalid_fields.append("email: Invalid email format")
    
    if 'username' in data:
        username = data['username'].strip()
        if len(username) < 3:
            invalid_fields.append("username: Must be at least 3 characters long")
        if not username.replace('_', '').replace('-', '').isalnum():
            invalid_fields.append("username: Can only contain letters, numbers, hyphens, and underscores")
    
    if 'password' in data:
        password = data['password']
        if len(password) < 6:
            invalid_fields.append("password: Must be at least 6 characters long")
    
    if invalid_fields:
        raise ValidationError(f"Invalid fields: {'; '.join(invalid_fields)}")

# Rate limiting utilities
class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = {}
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = datetime.utcnow()
    
    def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed within rate limit"""
        now = datetime.utcnow()
        
        # Periodic cleanup
        if (now - self.last_cleanup).seconds > self.cleanup_interval:
            self.cleanup_old_entries()
            self.last_cleanup = now
        
        # Get or create request history for identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        request_history = self.requests[identifier]
        
        # Remove old requests outside the window
        cutoff_time = now.timestamp() - window_seconds
        request_history[:] = [req_time for req_time in request_history if req_time > cutoff_time]
        
        # Check if within limit
        if len(request_history) >= max_requests:
            return False
        
        # Add current request
        request_history.append(now.timestamp())
        return True
    
    def cleanup_old_entries(self):
        """Clean up old entries to prevent memory leak"""
        now = datetime.utcnow().timestamp()
        cutoff = now - 3600  # Keep only last hour
        
        for identifier in list(self.requests.keys()):
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier] 
                if req_time > cutoff
            ]
            
            # Remove empty entries
            if not self.requests[identifier]:
                del self.requests[identifier]

# Global rate limiter instance
rate_limiter = RateLimiter()

# Security utilities
def validate_session_id(session_id: str):
    """Validate session ID format"""
    if not session_id or not isinstance(session_id, str):
        raise ValidationError("Invalid session ID")
    
    if len(session_id) < 10 or len(session_id) > 100:
        raise ValidationError("Session ID length invalid")
    
    # Check for dangerous characters
    if not session_id.replace('-', '').replace('_', '').isalnum():
        raise ValidationError("Session ID contains invalid characters")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    if not filename:
        return "unknown"
    
    # Remove path separators and dangerous characters
    import re
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    
    # Ensure filename is not empty and has reasonable length
    if not filename:
        filename = "unknown"
    elif len(filename) > 255:
        filename = filename[:255]
    
    return filename