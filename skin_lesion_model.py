"""
Skin Lesion Classification Models - ResNet50 and MobileNetV2 implementations
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionClassifier(nn.Module):
    """Base skin lesion classifier class"""
    
    def __init__(self, model_type: str = "resnet50", num_classes: int = 2, pretrained: bool = True):
        super(SkinLesionClassifier, self).__init__()
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        
        if self.model_type == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            # Replace the final layer
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)
            
        elif self.model_type == "mobilenetv2":
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            # Replace the classifier
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict_proba(self, x):
        """Get prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities
    
    def predict(self, x):
        """Get class predictions"""
        probabilities = self.predict_proba(x)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions

class ImagePreprocessor:
    """Image preprocessing for skin lesion images"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        
        # Training transforms with data augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/Test transforms (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Inference transform for single image prediction
        self.inference_transform = self.val_transform
    
    def preprocess_image(self, image_path: str, for_training: bool = False) -> torch.Tensor:
        """
        Preprocess a single image for model input
        
        Args:
            image_path: Path to the image file
            for_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            transform = self.train_transform if for_training else self.inference_transform
            tensor = transform(image)
            
            # Add batch dimension
            return tensor.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def preprocess_pil_image(self, pil_image: Image.Image, for_training: bool = False) -> torch.Tensor:
        """
        Preprocess a PIL image for model input
        
        Args:
            pil_image: PIL Image object
            for_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Ensure RGB format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Apply transforms
            transform = self.train_transform if for_training else self.inference_transform
            tensor = transform(pil_image)
            
            # Add batch dimension
            return tensor.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error preprocessing PIL image: {e}")
            raise

class ModelInference:
    """Model inference utilities"""
    
    def __init__(self, model_path: str, model_type: str = "resnet50", device: str = None):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Class labels
        self.class_labels = {0: "benign", 1: "malignant"}
    
    def _load_model(self) -> SkinLesionClassifier:
        """Load the trained model"""
        try:
            # Initialize model
            model = SkinLesionClassifier(model_type=self.model_type, num_classes=2)
            
            # Load state dict
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # Move to device
            model.to(self.device)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        Predict skin lesion type from image path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            image_tensor = self.preprocessor.preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
            
            # Extract results
            pred_class = prediction.item()
            confidence = probabilities[0][pred_class].item()
            
            # Prepare result
            result = {
                "prediction": self.class_labels[pred_class],
                "confidence": float(confidence),
                "probabilities": {
                    "benign": float(probabilities[0][0]),
                    "malignant": float(probabilities[0][1])
                },
                "model_type": self.model_type,
                "device": self.device
            }
            
            logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_pil_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Predict skin lesion type from PIL Image
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            image_tensor = self.preprocessor.preprocess_pil_image(pil_image)
            image_tensor = image_tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
            
            # Extract results
            pred_class = prediction.item()
            confidence = probabilities[0][pred_class].item()
            
            # Prepare result
            result = {
                "prediction": self.class_labels[pred_class],
                "confidence": float(confidence),
                "probabilities": {
                    "benign": float(probabilities[0][0]),
                    "malignant": float(probabilities[0][1])
                },
                "model_type": self.model_type,
                "device": self.device
            }
            
            logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

def create_model(model_type: str = "resnet50", pretrained: bool = True) -> SkinLesionClassifier:
    """Factory function to create a model"""
    return SkinLesionClassifier(model_type=model_type, pretrained=pretrained)

def load_inference_model(model_path: str, model_type: str = "resnet50", device: str = None) -> ModelInference:
    """Factory function to create an inference model"""
    return ModelInference(model_path=model_path, model_type=model_type, device=device)