import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time
import logging
from pathlib import Path
from typing import Dict, Tuple

from .losses import DistillationLoss
from .trainer import PADTrainer

logger = logging.getLogger(__name__)

class DistillationTrainer(PADTrainer):
    """
    Knowledge distillation trainer for PAD Engine.
    Trains a student model using a frozen teacher model.
    """
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 config: dict, device: torch.device):
        """
        Initialize the distillation trainer.
        """
        # Initialize base trainer with student model
        super().__init__(student_model, train_loader, val_loader, config, device)
        
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()  # Teacher is always frozen
        
        # Freeze teacher parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Distillation loss
        alpha = config.get('kd_alpha', 0.5)
        temperature = config.get('kd_temperature', 2.0)
        self.distillation_criterion = DistillationLoss(alpha=alpha, temperature=temperature)

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch using distillation.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets = batch[0], batch[1]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Student forward
                student_outputs = self.model(inputs)
                
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs)
                    
                # Calculate distillation loss
                loss = self.distillation_criterion(student_outputs, teacher_outputs, targets)
                
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('clip_grad', 1.0))
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, {}

    # validate() and train() are inherited from PADTrainer
