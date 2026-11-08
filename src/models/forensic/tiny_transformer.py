import torch
import torch.nn as nn
from typing import Tuple, List

class PatchEmbedding(nn.Module):
    """
    Patch Embedding for TinyTransformer.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 128, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

class TinyTransformer(nn.Module):
    """
    TinyTransformer for Forensic Escalation.
    Small ViT-like architecture.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 128, num_heads: int = 4, depth: int = 3, num_classes: int = 3):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, embed_dim=embed_dim, img_size=img_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, keyframes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keyframes: (B, num_keyframes, 3, H, W)
        Returns:
            forensic_scores: (B, 3)
        """
        B, N, C, H, W = keyframes.shape
        
        # Flatten batch and temporal dimension to process patches
        x = keyframes.view(B * N, C, H, W)
        x = self.patch_embed(x)
        x = self.transformer(x)
        cls_token_out = x[:, 0]
        
        # Reshape and pool over the temporal dimension (keyframes)
        cls_token_out = cls_token_out.view(B, N, -1)
        temporal_avg = cls_token_out.mean(dim=1)
        
        scores = self.head(temporal_avg)
        return scores

class KeyframeSelector:
    """
    Selects N most informative keyframes from a video.
    """
    def select(self, frames: torch.Tensor, uncertainty_scores: List[float], n: int = 8) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            frames: (num_frames, C, H, W)
            uncertainty_scores: List of length num_frames
            n: number of keyframes to select
        Returns:
            selected_frames: (min(n, num_frames), C, H, W)
            indices: List of selected frame indices
        """
        num_frames = len(frames)
        if num_frames <= n:
            return frames, list(range(num_frames))
        
        # Select evenly spaced frames
        n_spaced = n // 2
        n_uncertain = n - n_spaced
        
        spaced_indices = [int(i * num_frames / n_spaced) for i in range(n_spaced)]
        
        # From the remaining frames, select those with highest uncertainty
        remaining_indices = [i for i in range(num_frames) if i not in spaced_indices]
        remaining_scores = [(i, uncertainty_scores[i]) for i in remaining_indices]
        remaining_scores.sort(key=lambda x: x[1], reverse=True)
        
        uncertain_indices = [x[0] for x in remaining_scores[:n_uncertain]]
        
        final_indices = sorted(spaced_indices + uncertain_indices)
        selected_frames = frames[final_indices]
        
        return selected_frames, final_indices
