#!/usr/bin/env python3
"""
Startup script for Skin Lesion Classification System
"""

import os
import sys
import subprocess
import uvicorn
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import torch
        import fastapi
        import sqlalchemy
        import psycopg2
        logger.info("✅ All main dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False

def check_database():
    """Check database connection"""
    try:
        from database import DatabaseManager
        
        if DatabaseManager.check_connection():
            logger.info("✅ Database connection successful")
            return True
        else:
            logger.error("❌ Database connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ Database check error: {e}")
        return False

def initialize_database():
    """Initialize database tables"""
    try:
        from database import DatabaseManager
        DatabaseManager.init_db()
        logger.info("✅ Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    model_dir = Path("models")
    model_files = list(model_dir.glob("*.pth"))
    
    if model_files:
        logger.info(f"✅ Found {len(model_files)} model file(s)")
        for model_file in model_files:
            logger.info(f"   - {model_file.name}")
        return True
    else:
        logger.warning("⚠️  No trained model files found")
        logger.info("   You can train a model using: python -c \"from models import train_skin_lesion_model; train_skin_lesion_model()\"")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "uploads",
        "uploads/temp", 
        "uploads/predictions",
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory ready: {directory}")

def start_database_services():
    """Start database services using Docker Compose"""
    try:
        # Check if docker-compose is available
        result = subprocess.run(
            ["docker-compose", "--version"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            logger.warning("⚠️  Docker Compose not available")
            return False
            
        # Start database services
        logger.info("🚀 Starting database services...")
        result = subprocess.run(
            ["docker-compose", "up", "-d", "postgres", "pgadmin"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Database services started")
            logger.info("   PostgreSQL: http://localhost:5432")
            logger.info("   pgAdmin: http://localhost:5050 (admin@admin.com / admin)")
            return True
        else:
            logger.error(f"❌ Failed to start database services: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.warning("⚠️  Docker not available, please start database manually")
        return False
    except Exception as e:
        logger.error(f"❌ Error starting database services: {e}")
        return False

def main():
    """Main startup function"""
    print("🩺 Skin Lesion Classification System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    # Try to start database services
    db_started = start_database_services()
    
    if db_started:
        # Wait a moment for services to start
        import time
        logger.info("⏳ Waiting for database services to start...")
        time.sleep(5)
    
    # Check database connection
    if not check_database():
        logger.error("❌ Database not available")
        logger.info("   Please ensure PostgreSQL is running and configured correctly")
        logger.info("   Check your .env file for database settings")
        
        choice = input("\nContinue anyway? (y/N): ").strip().lower()
        if choice != 'y':
            sys.exit(1)
    
    # Initialize database
    try:
        initialize_database()
    except Exception as e:
        logger.warning(f"⚠️  Database initialization skipped: {e}")
    
    # Check model files
    check_model_files()
    
    # Start the application
    print("\n🚀 Starting Skin Lesion Classification System...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   Web App: http://localhost:{port}/app")
    print(f"   API Docs: http://localhost:{port}/docs")
    print("\n" + "=" * 50)
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if debug else "warning"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Skin Lesion Classification System")
        print("Thank you for using our application!")
    except Exception as e:
        logger.error(f"❌ Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()