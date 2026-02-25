"""
Database models and configuration for Skin Lesion Classification
"""

from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Float, Text, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import uuid
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./skin_lesion_app.db")

# Create engine with proper configuration
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL debugging
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)

# Session configuration
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base model
Base = declarative_base()

# Models
class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"

class Prediction(Base):
    """Prediction results model"""
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)  # Allow anonymous predictions
    session_id = Column(String, nullable=True)  # For temporary sessions
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    prediction = Column(String(50), nullable=False)  # benign, malignant, uncertain, etc.
    confidence = Column(Float, nullable=False)
    model_used = Column(String(50), nullable=False, default="huggingface")
    processing_time = Column(Float, nullable=True)
    hf_analysis = Column(Text, nullable=True)  # AI analysis text
    probabilities = Column(JSON, nullable=True)  # Store probability breakdown
    technical_details = Column(JSON, nullable=True)  # Additional technical info
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, prediction={self.prediction}, confidence={self.confidence})>"

class TempCapture(Base):
    """Temporary captures for anonymous users"""
    __tablename__ = "temp_captures"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), nullable=False, index=True)
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    prediction = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    model_used = Column(String(50), nullable=True)
    is_processed = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<TempCapture(id={self.id}, session_id={self.session_id})>"

class TrainingLog(Base):
    """Training session logs"""
    __tablename__ = "training_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_type = Column(String(50), nullable=False)
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    training_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    epoch_number = Column(Integer, nullable=True)
    total_epochs = Column(Integer, nullable=True)
    training_time = Column(Float, nullable=True)
    model_path = Column(String(500), nullable=True)
    training_params = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<TrainingLog(id={self.id}, model_type={self.model_type}, epoch={self.epoch_number})>"

class SystemLog(Base):
    """System event logs"""
    __tablename__ = "system_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    log_level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    component = Column(String(100), nullable=True)
    user_id = Column(String, nullable=True)
    additional_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level={self.log_level}, component={self.component})>"

# Database utility functions
def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """Get a new database session"""
    return SessionLocal()

class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def init_db():
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    @staticmethod
    def check_connection():
        """Check database connection"""
        try:
            db = SessionLocal()
            # Execute a simple query
            db.execute("SELECT 1")
            db.close()
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    @staticmethod
    def cleanup_temp_captures():
        """Clean up expired temporary captures"""
        try:
            db = SessionLocal()
            expired_captures = db.query(TempCapture).filter(
                TempCapture.expires_at < func.now()
            ).all()
            
            count = len(expired_captures)
            if count > 0:
                # Delete files first
                import os
                for capture in expired_captures:
                    try:
                        if os.path.exists(capture.image_path):
                            os.remove(capture.image_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete file {capture.image_path}: {e}")
                
                # Delete database records
                db.query(TempCapture).filter(
                    TempCapture.expires_at < func.now()
                ).delete()
                db.commit()
                logger.info(f"Cleaned up {count} expired temp captures")
            
            db.close()
            return count
        except Exception as e:
            logger.error(f"Failed to cleanup temp captures: {e}")
            return 0
    
    @staticmethod
    def get_user_predictions(user_id: str, limit: int = 50):
        """Get user's prediction history"""
        try:
            db = SessionLocal()
            predictions = db.query(Prediction).filter(
                Prediction.user_id == user_id
            ).order_by(Prediction.timestamp.desc()).limit(limit).all()
            db.close()
            return predictions
        except Exception as e:
            logger.error(f"Failed to get user predictions: {e}")
            return []
    
    @staticmethod
    def create_user(email: str, username: str, hashed_password: str, full_name: str = None):
        """Create a new user"""
        try:
            db = SessionLocal()
            
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.email == email) | (User.username == username)
            ).first()
            
            if existing_user:
                db.close()
                raise ValueError("User with this email or username already exists")
            
            # Create new user
            user = User(
                email=email,
                username=username,
                hashed_password=hashed_password,
                full_name=full_name
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            db.close()
            
            logger.info(f"Created new user: {username} ({email})")
            return user
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    @staticmethod
    def get_user_by_username(username: str):
        """Get user by username"""
        try:
            db = SessionLocal()
            user = db.query(User).filter(User.username == username).first()
            db.close()
            return user
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None
    
    @staticmethod
    def get_user_by_email(email: str):
        """Get user by email"""
        try:
            db = SessionLocal()
            user = db.query(User).filter(User.email == email).first()
            db.close()
            return user
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            return None
    
    @staticmethod
    def save_prediction(prediction_data: dict):
        """Save a prediction to database"""
        try:
            db = SessionLocal()
            
            prediction = Prediction(
                user_id=prediction_data.get("user_id"),
                session_id=prediction_data.get("session_id"),
                image_filename=prediction_data["image_filename"],
                image_path=prediction_data["image_path"],
                prediction=prediction_data["prediction"],
                confidence=prediction_data["confidence"],
                model_used=prediction_data.get("model_used", "huggingface"),
                processing_time=prediction_data.get("processing_time"),
                hf_analysis=prediction_data.get("hf_analysis"),
                probabilities=prediction_data.get("probabilities"),
                technical_details=prediction_data.get("technical_details")
            )
            
            db.add(prediction)
            db.commit()
            db.refresh(prediction)
            db.close()
            
            logger.info(f"Saved prediction: {prediction.id}")
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            raise
    
    @staticmethod
    def log_system_event(level: str, message: str, component: str = None, 
                        user_id: str = None, additional_data: dict = None):
        """Log system event"""
        try:
            db = SessionLocal()
            
            log_entry = SystemLog(
                log_level=level,
                message=message,
                component=component,
                user_id=user_id,
                additional_data=additional_data
            )
            
            db.add(log_entry)
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")

# Initialize database on import (for testing)
if __name__ == "__main__":
    # Test database connection and initialization
    try:
        DatabaseManager.init_db()
        if DatabaseManager.check_connection():
            print("✅ Database initialized and connection successful")
        else:
            print("❌ Database connection failed")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")