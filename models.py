"""
SQLAlchemy models for the Skin Lesion Classification system
"""

from sqlalchemy import Column, String, Boolean, DateTime, Text, DECIMAL, Integer, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    prediction_result = Column(String(50), nullable=False)  # 'benign' or 'malignant'
    confidence_score = Column(DECIMAL(5, 4), nullable=False)  # 0.0000 to 1.0000
    model_used = Column(String(50), nullable=False)  # 'resnet50' or 'mobilenetv2'
    processing_time = Column(DECIMAL(8, 4))  # in seconds
    additional_notes = Column(Text)
    openrouter_analysis = Column(Text)  # AI analysis from OpenRouter
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="predictions")

class TempCapture(Base):
    __tablename__ = "temp_captures"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), nullable=False, index=True)
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    prediction_result = Column(String(50))
    confidence_score = Column(DECIMAL(5, 4))
    model_used = Column(String(50))
    is_processed = Column(Boolean, default=False)
    expires_at = Column(DateTime(timezone=True), server_default=func.now() + func.interval('24 hours'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TrainingLog(Base):
    __tablename__ = "training_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_type = Column(String(50), nullable=False)
    training_accuracy = Column(DECIMAL(5, 4))
    validation_accuracy = Column(DECIMAL(5, 4))
    training_loss = Column(DECIMAL(8, 6))
    validation_loss = Column(DECIMAL(8, 6))
    epoch_number = Column(Integer)
    total_epochs = Column(Integer)
    training_time = Column(DECIMAL(10, 4))
    model_path = Column(String(500))
    training_params = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SystemLog(Base):
    __tablename__ = "system_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    log_level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    component = Column(String(100))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"))
    additional_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())