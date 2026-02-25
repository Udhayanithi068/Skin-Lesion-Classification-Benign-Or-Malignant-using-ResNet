"""
Model training script for skin lesion classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import time
import copy
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

from .skin_lesion_model import SkinLesionClassifier, ImagePreprocessor
from database import get_db_session, TrainingLog

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionDataset(Dataset):
    """Custom dataset for skin lesion images"""
    
    def __init__(self, data_dir: str, transform=None, split: str = "train"):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = ["benign", "malignant"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob("*.jpeg"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        logger.info(f"Found {len(self.samples)} images in {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

class ModelTrainer:
    """Model training class"""
    
    def __init__(
        self,
        model_type: str = "resnet50",
        data_dir: str = "./data",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 25,
        device: str = None
    ):
        self.model_type = model_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Initialize model
        self.model = SkinLesionClassifier(model_type=model_type, pretrained=True)
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training, validation, and testing"""
        
        # Create datasets
        train_dataset = SkinLesionDataset(
            self.data_dir, 
            transform=self.preprocessor.train_transform,
            split="train"
        )
        
        val_dataset = SkinLesionDataset(
            self.data_dir,
            transform=self.preprocessor.val_transform,
            split="val"
        )
        
        test_dataset = SkinLesionDataset(
            self.data_dir,
            transform=self.preprocessor.val_transform,
            split="test"
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct_predictions / total_predictions
        
        return epoch_loss, epoch_acc
    
    def train(self) -> Dict[str, Any]:
        """Train the model"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model: {self.model_type}, Epochs: {self.num_epochs}, Batch size: {self.batch_size}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        # Training loop
        best_val_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())
        training_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())
            
            epoch_time = time.time() - epoch_start_time
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Epoch Time: {epoch_time:.2f}s")
            
            # Log to database
            try:
                self._log_training_epoch(epoch + 1, train_acc, val_acc, train_loss, val_loss, epoch_time)
            except Exception as e:
                logger.warning(f"Failed to log training epoch to database: {e}")
        
        # Load best model weights
        self.model.load_state_dict(best_model_weights)
        
        training_time = time.time() - training_start_time
        
        # Test the model
        test_acc = self.test(test_loader)
        
        results = {
            'model_type': self.model_type,
            'best_val_acc': best_val_acc,
            'final_test_acc': test_acc,
            'training_time': training_time,
            'history': self.history
        }
        
        logger.info(f"Training completed! Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return results
    
    def test(self, test_loader: DataLoader) -> float:
        """Test the model"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        test_acc = correct_predictions / total_predictions
        return test_acc
    
    def save_model(self, save_path: str, include_metadata: bool = True):
        """Save the trained model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if include_metadata:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history
            }, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def _log_training_epoch(self, epoch: int, train_acc: float, val_acc: float, 
                           train_loss: float, val_loss: float, epoch_time: float):
        """Log training epoch to database"""
        try:
            db = get_db_session()
            
            training_log = TrainingLog(
                model_type=self.model_type,
                training_accuracy=train_acc,
                validation_accuracy=val_acc,
                training_loss=train_loss,
                validation_loss=val_loss,
                epoch_number=epoch,
                total_epochs=self.num_epochs,
                training_time=epoch_time,
                training_params={
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'device': self.device
                }
            )
            
            db.add(training_log)
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Failed to log training epoch: {e}")

def train_skin_lesion_model(
    model_type: str = "resnet50",
    data_dir: str = "./data",
    save_dir: str = "./models",
    **kwargs
) -> Dict[str, Any]:
    """Main training function"""
    
    # Create trainer
    trainer = ModelTrainer(model_type=model_type, data_dir=data_dir, **kwargs)
    
    # Train model
    results = trainer.train()
    
    # Save model
    model_filename = f"{model_type}_skin_lesion_classifier.pth"
    save_path = os.path.join(save_dir, model_filename)
    trainer.save_model(save_path)
    
    results['model_path'] = save_path
    
    return results

if __name__ == "__main__":
    # Example usage
    results = train_skin_lesion_model(
        model_type="resnet50",
        data_dir="./data",
        batch_size=32,
        learning_rate=0.001,
        num_epochs=25
    )
    
    print("Training Results:")
    for key, value in results.items():
        if key != 'history':
            print(f"{key}: {value}")