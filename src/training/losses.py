import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedCELoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for handling class imbalance.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        """
        Initialize the weighted cross-entropy loss.
        
        Args:
            weight (Optional[torch.Tensor]): A manual rescaling weight given to each class.
            reduction (str): Specifies the reduction to apply to the output.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs (torch.Tensor): Logits from the model (N, C).
            targets (torch.Tensor): Target labels (N,).
            
        Returns:
            torch.Tensor: Computed loss.
        """
        return self.criterion(inputs, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha (float): Weighting factor.
            gamma (float): Focusing parameter.
            reduction (str): Specifies the reduction to apply to the output.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiHeadLoss(nn.Module):
    """
    Combines losses from texture, frequency, and geometry heads, along with fusion loss.
    """
    def __init__(self, w_texture: float = 1.0, w_freq: float = 1.0, 
                 w_geo: float = 1.0, w_fusion: float = 1.0):
        """
        Initialize the multi-head loss.
        
        Args:
            w_texture (float): Weight for texture loss.
            w_freq (float): Weight for frequency loss.
            w_geo (float): Weight for geometry loss.
            w_fusion (float): Weight for fusion loss.
        """
        super().__init__()
        self.w_texture = w_texture
        self.w_freq = w_freq
        self.w_geo = w_geo
        self.w_fusion = w_fusion
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_texture: torch.Tensor, logits_freq: torch.Tensor, 
                logits_geo: torch.Tensor, logits_fusion: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        l_texture = self.criterion(logits_texture, targets)
        l_freq = self.criterion(logits_freq, targets)
        l_geo = self.criterion(logits_geo, targets)
        l_fusion = self.criterion(logits_fusion, targets)
        
        total_loss = (self.w_texture * l_texture + 
                      self.w_freq * l_freq + 
                      self.w_geo * l_geo + 
                      self.w_fusion * l_fusion)
        return total_loss


class DistillationLoss(nn.Module):
    """
    KL divergence between teacher and student outputs + hard label loss.
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        """
        Initialize distillation loss.
        
        Args:
            alpha (float): Balancing parameter between KD loss and CE loss.
            temperature (float): Temperature for softening probabilities.
        """
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        
        kd_loss = self.kl_div(student_soft, teacher_soft) * (self.T ** 2)
        hard_loss = self.ce(student_logits, targets)
        
        return self.alpha * kd_loss + (1 - self.alpha) * hard_loss
