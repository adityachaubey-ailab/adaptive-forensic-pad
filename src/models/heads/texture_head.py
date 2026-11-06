import torch
import torch.nn as nn

class TextureHead(nn.Module):
    """
    Texture Analysis Head.
    Input: Shared encoder feature maps (B, C, H, W).
    Detects presentation attacks based on texture artifacts.
    """
    def __init__(self, in_channels: int, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Encoder features (B, C, H, W)
        Returns:
            torch.Tensor: Texture scores (B, num_classes)
        """
        x = self.pool(features)
        return self.classifier(x)
