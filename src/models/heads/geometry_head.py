import torch
import torch.nn as nn

def extract_geometry_features(landmarks: torch.Tensor) -> torch.Tensor:
    """
    Extract geometric features from 468 3D landmarks.
    Computes pose angles, IPD, ratios, etc. (Flattening implemented as mock).
    Args:
        landmarks (torch.Tensor): Tensor of shape (B, 468, 3) or similar.
    Returns:
        torch.Tensor: Flattened normalized features (B, 1404).
    """
    B = landmarks.size(0)
    features = landmarks.reshape(B, -1)
    # Simple normalization
    features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-8)
    return features

class GeometryHead(nn.Module):
    """
    Geometry Analysis Head.
    Input: landmark features (B, 1404)
    Output: classification score.
    """
    def __init__(self, in_features: int = 1404, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(128, num_classes)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, landmark_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            landmark_features (torch.Tensor): Raw or extracted features (B, 1404).
        Returns:
            torch.Tensor: Geometry scores (B, num_classes)
        """
        if landmark_features.dim() > 2:
            landmark_features = extract_geometry_features(landmark_features)
        return self.mlp(landmark_features)
