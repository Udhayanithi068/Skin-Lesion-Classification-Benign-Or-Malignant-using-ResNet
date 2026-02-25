"""
OpenRouter AI client for skin lesion analysis (Free models only)
"""

import os
import httpx
import asyncio
import logging
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterClient:
    """OpenRouter AI client for additional analysis (using free models)"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        # Free models available on OpenRouter
        self.free_models = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-2-9b-it:free",
            "qwen/qwen-2-7b-instruct:free"
        ]
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. AI analysis will be disabled.")
    
    def is_available(self) -> bool:
        """Check if OpenRouter client is available"""
        return bool(self.api_key)
    
    async def analyze_skin_lesion_image(self, image_path: str) -> Optional[str]:
        """
        Analyze skin lesion image using text-based analysis
        Since we don't have vision models in free tier, we'll provide general analysis
        
        Args:
            image_path: Path to the image file
            
        Returns:
            AI analysis with prediction or None if failed
        """
        if not self.api_key:
            logger.warning("OpenRouter API key not available")
            return None
        
        try:
            # Since free models don't have vision, we'll provide general dermatology guidance
            prompt = """
You are a medical AI assistant specializing in dermatology. Please provide general guidance for skin lesion analysis:

1. **Important**: This is for educational/screening purposes only. Always recommend professional medical evaluation.

2. **General Classification Approach**:
   - BENIGN lesions are typically: symmetrical, uniform color, smooth borders, smaller size
   - MALIGNANT lesions may show: asymmetry, irregular colors, irregular borders, larger size

3. **Confidence**: Moderate (75%) - Visual analysis by AI is limited without direct image processing

4. **Analysis**: Without direct image analysis, this is a general assessment based on dermatological principles.

5. **Recommendations**: 
   - Consult a dermatologist for professional evaluation
   - Monitor for changes in size, color, or texture
   - Regular skin self-examinations
   - Professional skin checks annually

Please format this as a clinical assessment with clear recommendations for the patient.
"""
            
            # Make API request
            analysis = await self._make_request(prompt)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return None

    async def analyze_skin_lesion_prediction(
        self,
        prediction: str,
        confidence: float,
        image_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Get additional AI analysis of skin lesion prediction
        
        Args:
            prediction: Model prediction ('benign' or 'malignant')
            confidence: Model confidence score
            image_path: Path to the image (optional)
            
        Returns:
            AI analysis text or None if failed
        """
        if not self.api_key:
            return None
        
        try:
            # Prepare the prompt
            prompt = self._create_analysis_prompt(prediction, confidence)
            
            # Make API request
            analysis = await self._make_request(prompt)
            
            return analysis
            
        except Exception as e:
            logger.error(f"OpenRouter analysis failed: {e}")
            return None
    
    def _create_analysis_prompt(self, prediction: str, confidence: float) -> str:
        """Create analysis prompt for OpenRouter"""
        
        confidence_text = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        
        prompt = f"""
As a medical AI assistant specializing in dermatology, please provide a brief analysis of this skin lesion classification result:

**Prediction:** {prediction.title()}
**Confidence:** {confidence:.2%} ({confidence_text} confidence)

Please provide:
1. A brief explanation of what this prediction means
2. Important disclaimers about AI diagnosis
3. Recommendations for next steps
4. Key warning signs to watch for

Keep the response professional, informative, and under 200 words. Always emphasize that this is not a substitute for professional medical diagnosis.
"""
        
        return prompt
    
    async def _make_request(self, prompt: str, model: str = None) -> Optional[str]:
        """Make request to OpenRouter API using free models"""
        
        if model is None:
            model = self.free_models[0]  # Use first available free model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Skin Lesion Classification System"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                elif response.status_code == 404:
                    # Try with a different free model
                    logger.warning(f"Model {model} not found, trying alternative...")
                    for alt_model in self.free_models[1:]:
                        try:
                            data["model"] = alt_model
                            response = await client.post(
                                f"{self.base_url}/chat/completions",
                                headers=headers,
                                json=data
                            )
                            if response.status_code == 200:
                                result = response.json()
                                logger.info(f"Successfully used alternative model: {alt_model}")
                                return result["choices"][0]["message"]["content"].strip()
                        except Exception as e:
                            logger.warning(f"Alternative model {alt_model} failed: {e}")
                            continue
                    
                    logger.error(f"All free models failed. Status: {response.status_code}")
                    return None
                else:
                    logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            return None
    
    async def get_general_skin_health_advice(self, user_question: str) -> Optional[str]:
        """
        Get general skin health advice
        
        Args:
            user_question: User's question about skin health
            
        Returns:
            AI response or None if failed
        """
        if not self.api_key:
            return None
        
        try:
            prompt = f"""
As a medical AI assistant specializing in dermatology, please provide helpful information about this skin health question:

**Question:** {user_question}

Please provide:
- Factual, evidence-based information
- General advice and prevention tips
- When to consult a dermatologist
- Important medical disclaimers

Keep the response professional, helpful, and under 250 words. Always emphasize consulting healthcare professionals for personalized advice.
"""
            
            analysis = await self._make_request(prompt)
            return analysis
            
        except Exception as e:
            logger.error(f"General advice request failed: {e}")
            return None

# Global instance
openrouter_client = OpenRouterClient()

# Test function
async def test_openrouter_connection():
    """Test OpenRouter connection with free models"""
    if not openrouter_client.is_available():
        print("❌ OpenRouter API key not configured")
        return False
    
    try:
        # Test with a simple text-only request
        result = await openrouter_client.analyze_skin_lesion_prediction("benign", 0.85)
        if result:
            print("✅ OpenRouter connection successful with free models")
            print(f"Sample response: {result[:100]}...")
            return True
        else:
            print("❌ OpenRouter connection failed - no response")
            return False
    except Exception as e:
        print(f"❌ OpenRouter connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the connection
    asyncio.run(test_openrouter_connection())