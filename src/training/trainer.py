import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.metrics import roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)

class PADTrainer:
    """
    Main training loop for the PAD Engine.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 config: dict, device: torch.device):
        """
        Initialize the trainer.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.get('epochs', 50)
        )
        
        # Loss
        class_weights = torch.tensor(config.get('class_weights', [1.0, 1.0])).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Mixed Precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.best_acer = float('inf')
        self.start_epoch = 0
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume
        if config.get('resume_checkpoint'):
            self.load_checkpoint(config['resume_checkpoint'])

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets = batch[0], batch[1]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('clip_grad', 1.0))
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, {}

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        Returns:
            Tuple[float, Dict[str, float]]: validation loss, metrics dict (APCER, BPCER, ACER, AUC)
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch[0], batch[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1] # Assuming class 1 is attack (spoof)
                preds = torch.argmax(outputs, dim=1)
                
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        avg_loss = total_loss / len(self.val_loader)
        
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        # Calculate APCER, BPCER, ACER
        # Assuming 0 = Live/Bona Fide, 1 = Attack/Spoof
        bona_fide_mask = (all_targets == 0)
        attack_mask = (all_targets == 1)
        
        # APCER: Attack Presentation Classification Error Rate (False Acceptance Rate)
        apcer = np.sum(all_preds[attack_mask] == 0) / np.sum(attack_mask) if np.sum(attack_mask) > 0 else 0.0
        
        # BPCER: Bona Fide Presentation Classification Error Rate (False Rejection Rate)
        bpcer = np.sum(all_preds[bona_fide_mask] == 1) / np.sum(bona_fide_mask) if np.sum(bona_fide_mask) > 0 else 0.0
        
        # ACER: Average Classification Error Rate
        acer = (apcer + bpcer) / 2.0
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.5
            
        metrics = {
            'APCER': apcer,
            'BPCER': bpcer,
            'ACER': acer,
            'AUC': auc
        }
        
        return avg_loss, metrics

    def train(self, num_epochs: int) -> Dict[str, list]:
        """
        Run the full training loop.
        """
        history = {'train_loss': [], 'val_loss': [], 'ACER': [], 'AUC': []}
        epochs_without_improvement = 0
        patience = self.config.get('early_stopping_patience', 10)
        
        for epoch in range(self.start_epoch, num_epochs):
            start_time = time.time()
            
            train_loss, _ = self.train_epoch()
            val_loss, val_metrics = self.validate()
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            current_lr = self.scheduler.get_last_lr()[0]
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['ACER'].append(val_metrics['ACER'])
            history['AUC'].append(val_metrics['AUC'])
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.1f}s] - "
                        f"LR: {current_lr:.6f} - "
                        f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                        f"APCER: {val_metrics['APCER']:.4f} - BPCER: {val_metrics['BPCER']:.4f} - "
                        f"ACER: {val_metrics['ACER']:.4f} - AUC: {val_metrics['AUC']:.4f}")
            
            if val_metrics['ACER'] < self.best_acer:
                self.best_acer = val_metrics['ACER']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs due to no improvement in ACER.")
                break
                
        return history

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acer': self.best_acer
        }
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            logger.info(f"Saved new best model to {path}")

    def load_checkpoint(self, path: str):
        """
        Load from a checkpoint.
        """
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_acer = checkpoint['best_acer']
            self.start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from checkpoint: {path} at epoch {self.start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {path}")
