import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

class MobileNetEncoder(nn.Module):
    """
    MobileNetV3 Shared Encoder for PAD Engine.
    Uses torchvision MobileNetV3 (Small or Large).
    Removes the classifier head to output raw feature maps.
    """
    def __init__(self, model_type: str = 'small', pretrained: bool = True):
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type == 'small':
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            self.feature_dim = 576
        elif self.model_type == 'large':
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            self.feature_dim = 960
        else:
            raise ValueError("model_type must be 'small' or 'large'")
        
        # Keep only the features extractor (drop classifier and avgpool)
        self.features = backbone.features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input batch of images (B, 3, 224, 224)
        Returns:
            torch.Tensor: Feature maps (B, C, H, W)
        """
        return self.features(x)
        
    def get_feature_dim(self) -> int:
        return self.feature_dim
        
    def freeze(self, num_layers: int = None):
        """Freeze all or first `num_layers` of the encoder."""
        if num_layers is None:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            for idx, child in enumerate(self.features.children()):
                if idx < num_layers:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True
