import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class RiskEstimator(nn.Module):
    """
    Feature fusion and risk estimation (Fusion MLP).
    """
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_dict: Dictionary of features to fuse. 
        Returns:
            risk_scores: (B, 3) logits
            confidence: (B,) max confidence from softmax
        """
        features_list = []
        # Maintain consistent ordering
        for key in ['texture_score', 'frequency_score', 'geometry_score', 'temporal_score', 'capture_integrity_score']:
            if key in feature_dict:
                features_list.append(feature_dict[key])
                
        # Concatenate features along feature dimension
        x = torch.cat(features_list, dim=1)
        risk_scores = self.mlp(x)
        probs = torch.softmax(risk_scores, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        return risk_scores, confidence

class EarlyExitDecision:
    """
    Decides whether to pass, spoof, or escalate to uncertain based on thresholds.
    """
    def __init__(self, high_threshold: float = 0.85, low_threshold: float = 0.15):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
    def __call__(self, risk_scores: torch.Tensor) -> List[str]:
        """
        Args:
            risk_scores: (B, 3) logits
        Returns:
            decisions: List of strings ('pass', 'spoof', or 'uncertain')
        """
        probs = torch.softmax(risk_scores, dim=1)
        decisions = []
        for i in range(probs.size(0)):
            p_real, p_spoof, p_uncertain = probs[i]
            
            # Simple thresholding logic
            if p_real > self.high_threshold:
                decisions.append('pass')
            elif p_spoof > self.high_threshold:
                decisions.append('spoof')
            else:
                decisions.append('uncertain')
        return decisions
