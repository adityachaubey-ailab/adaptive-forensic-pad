import torch
import torch.nn as nn
import torch.fft

def compute_fft_features(image: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D FFT magnitude spectrum of the input image.
    Args:
        image (torch.Tensor): Image tensor (B, C, H, W)
    Returns:
        torch.Tensor: Log-magnitude spectrum (B, C, H, W)
    """
    # 2D FFT, shifting zero-frequency component to center
    fft_img = torch.fft.fft2(image, norm="ortho")
    fft_shifted = torch.fft.fftshift(fft_img, dim=(-2, -1))
    
    # Compute log magnitude
    magnitude = torch.abs(fft_shifted) + 1e-8
    log_magnitude = torch.log(magnitude)
    return log_magnitude

class FrequencyHead(nn.Module):
    """
    Frequency Analysis Head.
    Takes raw image (B, 3, H, W), computes FFT log-magnitude, 
    and applies a small CNN to detect frequency artifacts.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # Small CNN for frequency features
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, roi_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_image (torch.Tensor): Input ROI image (B, 3, H, W)
        Returns:
            torch.Tensor: Frequency scores (B, num_classes)
        """
        # Convert to frequency domain
        freq_features = compute_fft_features(roi_image)
        # Apply CNN
        x = self.conv_net(freq_features)
        # Classification
        return self.fc(x)
