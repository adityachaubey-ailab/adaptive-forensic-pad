import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoder import MobileNetEncoder
from .heads import TextureHead, FrequencyHead, GeometryHead

class PADModel(nn.Module):
    """
    Complete PAD Model combining the shared encoder and all heads.
    Architecture:
        Face ROI -> Shared Encoder -> Texture Head
        Face ROI -> Frequency Head
        Landmarks -> Geometry Head
    """
    def __init__(
        self, 
        model_type: str = 'small', 
        pretrained: bool = True, 
        num_classes: int = 2
    ):
        super().__init__()
        
        # Shared Encoder
        self.encoder = MobileNetEncoder(model_type=model_type, pretrained=pretrained)
        encoder_dim = self.encoder.get_feature_dim()
        
        # Heads
        self.texture_head = TextureHead(in_channels=encoder_dim, num_classes=num_classes)
        self.frequency_head = FrequencyHead(in_channels=3, num_classes=num_classes)
        self.geometry_head = GeometryHead(in_features=1404, num_classes=num_classes)
        
    def forward(self, image: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            image (torch.Tensor): Face ROI image (B, 3, 224, 224).
            landmarks (torch.Tensor, optional): Facial landmarks (B, 468, 3).
        Returns:
            Dict containing output scores and features.
        """
        outputs = {}
        
        # 1. Frequency Analysis (from raw image)
        outputs['frequency_score'] = self.frequency_head(image)
        
        # 2. Shared Encoder
        enc_features = self.encoder(image)
        outputs['encoder_features'] = enc_features
        
        # 3. Texture Analysis
        outputs['texture_score'] = self.texture_head(enc_features)
        
        # 4. Geometry Analysis
        if landmarks is not None:
            outputs['geometry_score'] = self.geometry_head(landmarks)
            
        return outputs
        
    def get_model_size_mb(self) -> float:
        """Returns the size of the model in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
        
    def get_param_count(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PADBaselineModel(nn.Module):
    """
    Simple Baseline Model for Phase B.
    Encoder -> AdaptiveAvgPool -> Linear Classifier
    """
    def __init__(self, model_type: str = 'small', pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.encoder = MobileNetEncoder(model_type=model_type, pretrained=pretrained)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.encoder.get_feature_dim(), num_classes)
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        features = self.encoder(image)
        return self.classifier(features)

    def get_model_size_mb(self) -> float:
        """Returns the size of the model in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def get_param_count(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
