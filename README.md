# 🩺 Skin Lesion Classification System

A comprehensive AI-powered skin lesion classification system that uses deep learning to help detect whether skin lesions are benign or malignant. Built with FastAPI, React-like frontend, and PostgreSQL database.

![System Architecture](https://img.shields.io/badge/AI-Skin%20Lesion%20Classification-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue)
![Docker](https://img.shields.io/badge/Docker-Supported-blue)

## 🌟 Features

### 🤖 AI-Powered Analysis
- **Deep Learning Models**: ResNet50 and MobileNetV2 for accurate classification
- **Transfer Learning**: Pre-trained models fine-tuned for skin lesion detection
- **High Accuracy**: Trained on medical datasets for reliable predictions
- **Confidence Scoring**: Get confidence levels with each prediction

### 📱 Multi-Platform Access
- **Web Application**: Professional responsive web interface
- **Camera Integration**: Real-time camera capture for immediate analysis
- **Image Upload**: Support for various image formats (JPG, PNG, WebP)
- **Mobile Friendly**: Optimized for mobile devices and tablets

### 🔐 User Management
- **Authentication**: Secure user registration and login
- **Personal History**: Track your analysis history and trends
- **Session Management**: Temporary analysis for quick checks
- **Data Privacy**: HIPAA-compliant data handling

### 🧠 AI Enhancement
- **OpenRouter Integration**: Additional AI analysis and recommendations
- **Medical Insights**: Contextual information about predictions
- **Risk Assessment**: Comprehensive risk level analysis
- **Professional Guidance**: When to seek medical attention

### 💾 Database Management
- **PostgreSQL**: Robust data storage with full ACID compliance
- **pgAdmin**: Web-based database administration
- **Data Tracking**: Complete audit trail of all analyses
- **Performance Optimization**: Indexed queries for fast retrieval

## 🏗️ System Architecture

```
📱 Mobile/Web Client
    ↓
🌐 FastAPI Backend Server
    ↓
🤖 ML Models (ResNet50/MobileNetV2)
    ↓
💾 PostgreSQL Database
    ↓
🔧 pgAdmin (Database Management)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- PostgreSQL 13+ (or use Docker)
- Modern web browser with camera support

### 1. Clone the Repository

```bash
git clone <repository-url>
cd skin-lesion-classification
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit the .env file with your settings
nano .env
```

### 3. Configure Environment Variables

Update the `.env` file with your settings:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/skin_lesion_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=skin_lesion_db
DB_USER=username
DB_PASSWORD=password

# OpenRouter API (for AI analysis)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Security
SECRET_KEY=your_super_secret_key_change_this_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Settings
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### 4. Start Database (Docker Method)

```bash
# Start PostgreSQL and pgAdmin
docker-compose up -d

# Access pgAdmin at http://localhost:5050
# Email: admin@admin.com
# Password: admin
```

### 5. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6. Initialize Database

```bash
# Run database initialization
python -c "from database import DatabaseManager; DatabaseManager.init_db()"
```

### 7. Start the Application

```bash
# Start the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or run directly
python app/main.py
```

### 8. Access the Application

- **Web Application**: http://localhost:8000/app
- **API Documentation**: http://localhost:8000/docs
- **Database Admin**: http://localhost:5050

## 📊 Training Your Model

### Data Preparation

1. **Create Data Structure**:
```
data/
├── train/
│   ├── benign/
│   │   ├── image1.jpg
│   │   └── ...
│   └── malignant/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

2. **Prepare Your Dataset**:
   - Collect skin lesion images
   - Organize by benign/malignant labels
   - Ensure images are high quality
   - Recommended: 224x224 pixels minimum

### Training Process

```bash
# Train ResNet50 model
python -c "from models import train_skin_lesion_model; train_skin_lesion_model('resnet50', './data', batch_size=32, num_epochs=25)"

# Train MobileNetV2 model (lighter, faster)
python -c "from models import train_skin_lesion_model; train_skin_lesion_model('mobilenetv2', './data', batch_size=32, num_epochs=25)"
```

### Model Configuration

Edit model settings in `.env`:
```env
MODEL_TYPE=resnet50  # or mobilenetv2
IMAGE_SIZE=224
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=50
```

## 🔧 API Reference

### Authentication Endpoints

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "username": "testuser",
  "email": "user@example.com",
  "password": "secure_password",
  "full_name": "Test User"
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "testuser",
  "password": "secure_password"
}
```

### Prediction Endpoints

#### Upload and Analyze (Authenticated)
```http
POST /predict
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <image_file>
```

#### Temporary Analysis (No Authentication)
```http
POST /predict/temp
Content-Type: multipart/form-data

file: <image_file>
session_id: optional_session_id
```

#### Get User History
```http
GET /predictions?limit=10&offset=0
Authorization: Bearer <token>
```

### Response Format

```json
{
  "id": "uuid",
  "prediction": "benign",
  "confidence": 0.87,
  "probabilities": {
    "benign": 0.87,
    "malignant": 0.13
  },
  "model_type": "resnet50",
  "processing_time": 1.23,
  "openrouter_analysis": "AI analysis text...",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🖥️ Frontend Usage

### Image Upload
1. Navigate to the Upload section
2. Select an image file (JPG, PNG, WebP)
3. Preview the image
4. Click "Analyze Image"
5. View detailed results with AI insights

### Camera Capture
1. Go to the Camera section
2. Click "Start Camera" and allow permissions
3. Position the lesion in view
4. Click "Capture" to take a photo
5. Review and click "Analyze"
6. Get instant results

### User Features
- **Registration**: Create an account to save history
- **Login**: Access your personal dashboard
- **History**: View past analyses and trends
- **Profile**: Manage account settings

## 🔒 Security Features

### Data Protection
- **Encryption**: All passwords are bcrypt hashed
- **JWT Tokens**: Secure authentication tokens
- **HTTPS Support**: SSL/TLS encryption ready
- **Input Validation**: Comprehensive request validation

### Privacy Compliance
- **Data Minimization**: Only collect necessary information
- **User Control**: Users can delete their data
- **Audit Logging**: Complete activity tracking
- **Temporary Analysis**: No-registration option available

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### Manual Deployment

1. **Server Setup**:
```bash
# Install dependencies
sudo apt update
sudo apt install python3.8 python3-pip postgresql nginx

# Create application user
sudo useradd -m skinlesion
sudo su - skinlesion
```

2. **Application Setup**:
```bash
# Clone and setup application
git clone <repository-url>
cd skin-lesion-classification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Database Setup**:
```bash
# Create database and user
sudo -u postgres psql
CREATE DATABASE skin_lesion_db;
CREATE USER skinlesion_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE skin_lesion_db TO skinlesion_user;
```

4. **Process Manager**:
```bash
# Install and configure Supervisor
sudo apt install supervisor

# Create supervisor configuration
sudo nano /etc/supervisor/conf.d/skinlesion.conf
```

### Environment-Specific Settings

#### Production
```env
DEBUG=False
HOST=0.0.0.0
PORT=8000
SECRET_KEY=<very_secure_production_key>
DATABASE_URL=postgresql://user:pass@localhost:5432/prod_db
```

#### Staging
```env
DEBUG=False
HOST=0.0.0.0
PORT=8001
SECRET_KEY=<secure_staging_key>
DATABASE_URL=postgresql://user:pass@localhost:5432/staging_db
```

## 📈 Monitoring and Analytics

### Performance Monitoring
- **Response Time Tracking**: Monitor API response times
- **Model Performance**: Track prediction accuracy over time
- **User Analytics**: Usage patterns and popular features
- **Error Tracking**: Comprehensive error logging

### Database Monitoring
```sql
-- Check prediction statistics
SELECT 
    prediction_result,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence
FROM predictions 
GROUP BY prediction_result;

-- Monitor user activity
SELECT 
    DATE(created_at) as date,
    COUNT(*) as daily_predictions
FROM predictions 
GROUP BY DATE(created_at)
ORDER BY date DESC;
```

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Model Tests**: ML model accuracy testing
- **Security Tests**: Authentication and authorization
- **Performance Tests**: Load and stress testing

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd skin-lesion-classification

# Create development environment
python -m venv dev-env
source dev-env/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Standards
- **Python**: Follow PEP 8 style guidelines
- **JavaScript**: Use ES6+ features and proper formatting
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Write tests for new features

## 🆘 Troubleshooting

### Common Issues

#### 1. Model Not Loading
```bash
# Check model file exists
ls -la models/

# Retrain if missing
python -c "from models import train_skin_lesion_model; train_skin_lesion_model()"
```

#### 2. Database Connection Failed
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U username -d skin_lesion_db -c "SELECT 1;"
```

#### 3. Camera Not Working
- Check browser permissions
- Ensure HTTPS connection (required for camera)
- Verify camera device availability

#### 4. OpenRouter API Errors
- Verify API key in `.env`
- Check API quota and limits
- Review network connectivity

### Performance Optimization

#### Model Optimization
```python
# Use GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

#### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_user_created 
ON predictions(user_id, created_at DESC);

-- Monitor slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

## 📞 Support

### Getting Help
- **Documentation**: Check this README and code comments
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub Discussions for questions
- **Email**: contact@skinlesionai.com

### Medical Disclaimer

⚠️ **IMPORTANT MEDICAL DISCLAIMER**: This system is for educational and screening purposes only. It is NOT a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for proper medical evaluation and treatment of skin lesions.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **FastAPI**: For the high-performance API framework  
- **Bootstrap**: For the responsive UI components
- **Medical Community**: For datasets and domain expertise
- **Open Source Community**: For the amazing tools and libraries

---

**Built with ❤️ for better healthcare outcomes**