import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Any

from .encoder.mobilenet_encoder import MobileNetEncoder
from .heads.texture_head import TextureHead
from .heads.frequency_head import FrequencyHead
from .heads.geometry_head import GeometryHead
from .temporal.tcn import TCN
from .fusion.risk_estimator import RiskEstimator, EarlyExitDecision
from .forensic.tiny_transformer import TinyTransformer, KeyframeSelector

class AdaptiveForensicPAD(nn.Module):
    """
    Complete inference pipeline that combines all components of the PAD Engine.
    """
    def __init__(
        self,
        model_type: str = 'small',
        pretrained: bool = True,
        num_classes: int = 3,
        head_out_dim: int = 64,
        tcn_channels: Optional[List[int]] = None,
        img_size: int = 224
    ):
        super().__init__()
        # Shared Encoder
        self.shared_encoder = MobileNetEncoder(model_type=model_type, pretrained=pretrained)
        encoder_dim = self.shared_encoder.get_feature_dim()
        
        # Feature Heads
        self.texture_head = TextureHead(in_channels=encoder_dim, num_classes=head_out_dim)
        self.frequency_head = FrequencyHead(in_channels=3, num_classes=head_out_dim)
        self.geometry_head = GeometryHead(in_features=1404, num_classes=head_out_dim)
        
        # Temporal Module
        temporal_input_dim = head_out_dim * 3  # tex + freq + geom
        if tcn_channels is None:
            tcn_channels = [64, 64, 64]
        self.tcn = TCN(input_dim=temporal_input_dim, num_channels=tcn_channels)
        
        # Fusion Module — takes last-frame heads + temporal output
        fusion_input_dim = head_out_dim * 3 + tcn_channels[-1]
        self.risk_estimator = RiskEstimator(input_dim=fusion_input_dim, num_classes=num_classes)
        
        # Forensic Escalation Module
        self.tiny_transformer = TinyTransformer(img_size=img_size, num_classes=num_classes)
        self.early_exit = EarlyExitDecision()
        self.keyframe_selector = KeyframeSelector()
        
        self._head_out_dim = head_out_dim
        self._tcn_out_dim = tcn_channels[-1]

    def forward_fast(self, image: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast path for a single image.
        Args:
            image: (B, C, H, W)
            landmarks: Optional landmarks (B, 1404) flattened
        Returns:
            fast_decision: (B, 3) risk scores
            confidence: (B,)
        """
        features = self.shared_encoder(image)
        
        tex_score = self.texture_head(features)
        freq_score = self.frequency_head(image)  # frequency uses raw image
        
        B = image.size(0)
        if landmarks is not None:
            geom_score = self.geometry_head(landmarks)
        else:
            geom_score = torch.zeros(B, self._head_out_dim, device=image.device)
        
        # No temporal context for single frame
        temporal_score = torch.zeros(B, self._tcn_out_dim, device=image.device)
        
        feature_dict = {
            'texture_score': tex_score,
            'frequency_score': freq_score,
            'geometry_score': geom_score,
            'temporal_score': temporal_score
        }
        
        return self.risk_estimator(feature_dict)

    def forward_forensic(self, keyframes: torch.Tensor) -> torch.Tensor:
        """
        Expensive forensic path.
        Args:
            keyframes: (B, N, C, H, W)
        Returns:
            forensic_decision: (B, 3) risk scores
        """
        return self.tiny_transformer(keyframes)

    def forward_full(self, video_frames: torch.Tensor, landmarks_sequence: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full pipeline for a sequence of frames, potentially routing to forensic.
        Args:
            video_frames: (B, T, C, H, W)
            landmarks_sequence: Optional landmarks
        Returns:
            final_decision: (B, 3)
            evidence_dict: Contains all intermediate scores
        """
        B, T, C, H, W = video_frames.shape
        
        # Flatten to process frames individually through the shared encoder
        frames_flat = video_frames.view(B * T, C, H, W)
        
        features = self.shared_encoder(frames_flat)
        tex_score = self.texture_head(features)
        freq_score = self.frequency_head(frames_flat)  # frequency uses raw images
        
        # Geometry from landmarks if available, else zeros
        if landmarks_sequence is not None:
            # landmarks_sequence: (B, T, 1404)
            landmarks_flat = landmarks_sequence.view(B * T, -1)
            geom_score = self.geometry_head(landmarks_flat)
        else:
            geom_score = torch.zeros(B * T, self._head_out_dim, device=video_frames.device)
        
        tex_seq = tex_score.view(B, T, -1)
        freq_seq = freq_score.view(B, T, -1)
        geom_seq = geom_score.view(B, T, -1)
        
        # Concatenate features for temporal input
        temporal_input = torch.cat([tex_seq, freq_seq, geom_seq], dim=-1)
        
        # Get temporal summary
        temporal_score = self.tcn(temporal_input) # (B, 64)
        
        # For fusion, use the spatial features of the last frame alongside temporal summary
        last_tex = tex_seq[:, -1, :]
        last_freq = freq_seq[:, -1, :]
        last_geom = geom_seq[:, -1, :]
        
        feature_dict = {
            'texture_score': last_tex,
            'frequency_score': last_freq,
            'geometry_score': last_geom,
            'temporal_score': temporal_score
        }
        
        risk_scores, confidence = self.risk_estimator(feature_dict)
        decisions = self.early_exit(risk_scores)
        
        final_decision = risk_scores.clone()
        uncertain_mask = [d == 'uncertain' for d in decisions]
        
        if any(uncertain_mask):
            # Process uncertain items with the forensic module
            uncertain_indices = [i for i, d in enumerate(decisions) if d == 'uncertain']
            for i in uncertain_indices:
                frames_i = video_frames[i] # (T, C, H, W)
                # Dummy uncertainty scores to select keyframes
                uncertainty_scores = [0.5] * T
                kf, _ = self.keyframe_selector.select(frames_i, uncertainty_scores, n=8)
                kf_batch = kf.unsqueeze(0) # (1, N, C, H, W)
                forensic_scores = self.forward_forensic(kf_batch)
                final_decision[i] = forensic_scores[0]
                
        evidence_dict = {
            'texture_score': tex_seq,
            'frequency_score': freq_seq,
            'geometry_score': geom_seq,
            'temporal_score': temporal_score,
            'risk_scores': risk_scores,
            'confidence': confidence,
            'decisions': decisions,
            'final_decision': final_decision
        }
        
        return final_decision, evidence_dict

class FrameSkipScheduler:
    """
    Decides which frames get full ML evaluation vs tracking.
    """
    def __init__(self, ml_every_n: int = 4):
        self.ml_every_n = ml_every_n
        
    def should_run_ml(self, frame_idx: int) -> bool:
        """
        Can be extended to be adaptive based on uncertainty.
        """
        return frame_idx % self.ml_every_n == 0
