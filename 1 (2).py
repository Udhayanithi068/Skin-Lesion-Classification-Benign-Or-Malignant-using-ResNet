"""
Hugging Face vision model client for skin lesion analysis
"""

import os
import httpx
import asyncio
import logging
import base64
from PIL import Image
import io
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceVisionClient:
    """Hugging Face vision client for skin lesion analysis"""
    
    def __init__(self):
        self.api_token = os.getenv("HF_API_TOKEN", "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")
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Available vision models (free)
        self.vision_models = {
            "vit": "google/vit-base-patch16-224",
            "swin": "microsoft/swin-base-patch4-window7-224", 
            "dino": "facebook/dinov2-base",
            "medical": "microsoft/swin-tiny-patch4-window7-224"  # Smaller, faster
        }
        
        # Default model
        self.current_model = self.vision_models["vit"]
        
        if not self.api_token:
            logger.warning("Hugging Face API token not found")
    
    def is_available(self) -> bool:
        """Check if HF client is available"""
        return bool(self.api_token)
    
    async def analyze_skin_lesion(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze skin lesion using Hugging Face vision model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Analysis results with prediction and confidence
        """
        if not self.api_token:
            logger.warning("Hugging Face API token not available")
            return None
        
        try:
            # Load and preprocess image
            image_data = self._prepare_image(image_path)
            if not image_data:
                return None
            
            # Try vision analysis first
            vision_result = await self._analyze_with_vision_model(image_data)
            
            # If vision analysis fails, use classification model
            if not vision_result:
                vision_result = await self._analyze_with_classification_model(image_data)
            
            # Process and interpret results
            if vision_result:
                processed_result = self._process_vision_results(vision_result)
                return processed_result
            
            return None
            
        except Exception as e:
            logger.error(f"Skin lesion analysis failed: {e}")
            return None
    
    def _prepare_image(self, image_path: str) -> Optional[bytes]:
        """Prepare image for HF API"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (HF has size limits)
                max_size = 1024
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                return img_byte_arr.getvalue()
                
        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            return None
    
    async def _analyze_with_vision_model(self, image_data: bytes) -> Optional[Any]:
        """Analyze with vision transformer model"""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/octet-stream"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try different models in order of preference
                for model_name, model_id in self.vision_models.items():
                    try:
                        logger.info(f"Trying model: {model_id}")
                        
                        response = await client.post(
                            f"{self.base_url}/{model_id}",
                            headers=headers,
                            content=image_data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            logger.info(f"Success with model: {model_id}")
                            return result
                        elif response.status_code == 503:
                            logger.warning(f"Model {model_id} is loading, trying next...")
                            continue
                        else:
                            logger.warning(f"Model {model_id} failed with status {response.status_code}")
                            continue
                            
                    except Exception as e:
                        logger.warning(f"Model {model_id} failed: {e}")
                        continue
                
                logger.error("All vision models failed")
                return None
                
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return None
    
    async def _analyze_with_classification_model(self, image_data: bytes) -> Optional[Any]:
        """Fallback to image classification model"""
        classification_models = [
            "microsoft/resnet-50",
            "google/efficientnet-b0",
            "microsoft/beit-base-patch16-224"
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/octet-stream"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for model_id in classification_models:
                    try:
                        response = await client.post(
                            f"{self.base_url}/{model_id}",
                            headers=headers,
                            content=image_data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            logger.info(f"Classification success with: {model_id}")
                            return result
                            
                    except Exception as e:
                        logger.warning(f"Classification model {model_id} failed: {e}")
                        continue
                
                return None
                
        except Exception as e:
            logger.error(f"Classification analysis failed: {e}")
            return None
    
    def _process_vision_results(self, raw_results: Any) -> Dict[str, Any]:
        """Process and interpret vision model results for medical context"""
        try:
            # Handle different result formats
            if isinstance(raw_results, list) and len(raw_results) > 0:
                results = raw_results
            else:
                results = [raw_results]
            
            # Extract meaningful information
            processed = {
                "prediction": "uncertain",
                "confidence": 0.5,
                "analysis": "AI vision analysis completed",
                "technical_details": {
                    "model": self.current_model,
                    "raw_results": results[:3]  # Top 3 results
                },
                "interpretation": self._interpret_for_medical_context(results)
            }
            
            # Try to extract confidence and prediction
            if results and isinstance(results[0], dict):
                if 'score' in results[0]:
                    processed["confidence"] = float(results[0]['score'])
                
                if 'label' in results[0]:
                    label = results[0]['label'].lower()
                    processed["prediction"] = self._map_label_to_medical(label)
            
            return processed
            
        except Exception as e:
            logger.error(f"Result processing failed: {e}")
            return {
                "prediction": "uncertain",
                "confidence": 0.5,
                "analysis": f"Analysis completed with limitations: {str(e)}",
                "technical_details": {"error": str(e)}
            }
    
    def _map_label_to_medical(self, label: str) -> str:
        """Map model labels to medical categories"""
        # Map common ImageNet labels to medical context
        benign_indicators = [
            'skin', 'tissue', 'normal', 'healthy', 'spot', 'freckle', 
            'mole', 'nevus', 'birthmark'
        ]
        
        malignant_indicators = [
            'lesion', 'tumor', 'cancer', 'melanoma', 'carcinoma',
            'abnormal', 'suspicious', 'irregular'
        ]
        
        label_lower = label.lower()
        
        for indicator in malignant_indicators:
            if indicator in label_lower:
                return "malignant"
        
        for indicator in benign_indicators:
            if indicator in label_lower:
                return "benign"
        
        # If no clear mapping, stay uncertain
        return "uncertain"
    
    def _interpret_for_medical_context(self, results: List[Dict]) -> str:
        """Interpret results in medical context"""
        try:
            if not results or not isinstance(results[0], dict):
                return "Unable to provide detailed analysis"
            
            top_result = results[0]
            confidence = top_result.get('score', 0.5)
            label = top_result.get('label', 'unknown')
            
            interpretation = f"Vision analysis detected: {label} "
            
            if confidence > 0.8:
                interpretation += "(high confidence). "
            elif confidence > 0.6:
                interpretation += "(moderate confidence). "
            else:
                interpretation += "(low confidence). "
            
            interpretation += "This is an AI interpretation of visual features. "
            interpretation += "Professional medical evaluation is essential for accurate diagnosis."
            
            return interpretation
            
        except Exception as e:
            return f"Analysis interpretation limited due to: {str(e)}"

# Global instance
hf_vision_client = HuggingFaceVisionClient()

# Test function
async def test_hf_vision():
    """Test Hugging Face vision connection"""
    if not hf_vision_client.is_available():
        print("❌ Hugging Face API token not configured")
        return False
    
    try:
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='red')
        test_path = 'test_image.jpg'
        test_image.save(test_path)
        
        # Test analysis
        result = await hf_vision_client.analyze_skin_lesion(test_path)
        
        # Clean up
        os.remove(test_path)
        
        if result:
            print("✅ Hugging Face vision analysis working")
            print(f"Test result: {result}")
            return True
        else:
            print("❌ Hugging Face vision analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Hugging Face test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the connection
    asyncio.run(test_hf_vision())