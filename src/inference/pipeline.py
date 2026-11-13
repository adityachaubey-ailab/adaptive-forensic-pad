import cv2
import numpy as np
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ez = np.exp(x)
    return ez / (1.0 + ez)


def compute_physical_evidence(frame: np.ndarray) -> Dict[str, float]:
    """
    Computes spatial, frequency, and color-space physical evidence from face ROI.
    All scores are continuous 0.0-1.0 via sigmoid mapping (no binary cliff-edges).
    """
    if frame is None or frame.size == 0:
        return {'texture': 0.5, 'frequency': 0.5, 'color': 0.5,
                'is_ir': False, 'liveness_score': 0.5, 'spoof_score': 0.5}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # --- 1. Texture Evidence (Laplacian Variance) ---
    # Live skin in compressed video/webcam: variance ~15-500. Raw photo: 80-800.
    # Blurs / flat spoofs: < 8. Extreme high-freq dither: > 1200.
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    low_score = _sigmoid((lap_var - 12.0) * 0.25)
    high_penalty = _sigmoid((lap_var - 1200.0) * 0.005)
    texture_score = float(low_score * (1.0 - high_penalty * 0.6))

    # --- 2. Frequency Evidence (2D FFT High-Frequency Energy Ratio) ---
    resized_gray = cv2.resize(gray, (128, 128))
    f = np.fft.fft2(resized_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1e-8)

    cy, cx = 64, 64
    r = 20
    h_freq_spectrum = magnitude_spectrum.copy()
    h_freq_spectrum[cy - r:cy + r, cx - r:cx + r] = 0
    high_freq_ratio = float(np.mean(h_freq_spectrum) / (np.mean(magnitude_spectrum) + 1e-8))

    # Real faces: high_freq_ratio ~ 0.82-0.92 (rich natural texture)
    # Flat prints/masks: ratio < 0.70 (unnaturally smooth)
    # Screen pixel grids: ratio > 0.95 (artificial periodic peaks)
    frequency_score = float(_sigmoid((high_freq_ratio - 0.72) * 18.0)
                            * (1.0 - _sigmoid((high_freq_ratio - 0.96) * 25.0) * 0.5))

    # --- 3. Color Space / IR Reflectance ---
    is_ir = False
    if len(frame.shape) == 2:
        is_ir = True
    elif len(frame.shape) == 3:
        b, g, r_ch = frame[:, :, 0].astype(float), frame[:, :, 1].astype(float), frame[:, :, 2].astype(float)
        channel_diff = np.mean(np.abs(b - g)) + np.mean(np.abs(g - r_ch))
        if channel_diff < 15.0:
            is_ir = True

    if is_ir:
        ir_std = float(np.std(gray))
        specular_ratio = float(np.sum(gray > 240)) / (gray.size + 1e-8)
        # Real skin NIR: low specular, moderate std (15-65)
        spec_ok = 1.0 - _sigmoid((specular_ratio - 0.04) * 120.0)
        std_ok = _sigmoid((ir_std - 10.0) * 0.2) * (1.0 - _sigmoid((ir_std - 75.0) * 0.1))
        color_score = float(spec_ok * 0.6 + std_ok * 0.4)
        liveness_score = 0.4 * texture_score + 0.3 * frequency_score + 0.3 * color_score
    else:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cr_std = float(np.std(ycrcb[:, :, 1]))
        cb_std = float(np.std(ycrcb[:, :, 2]))
        chroma_std = (cr_std + cb_std) / 2.0
        # Live skin chrominance: std ~5-25. Screens: very low (<2) or oversaturated (>35)
        color_score = float(_sigmoid((chroma_std - 3.0) * 0.8)
                            * (1.0 - _sigmoid((chroma_std - 35.0) * 0.3) * 0.4))
        liveness_score = 0.35 * texture_score + 0.35 * frequency_score + 0.30 * color_score

    spoof_score = 1.0 - liveness_score

    return {
        'texture': float(texture_score),
        'frequency': float(frequency_score),
        'color': float(color_score),
        'is_ir': is_ir,
        'liveness_score': float(liveness_score),
        'spoof_score': float(spoof_score)
    }


class InferencePipeline:
    """
    Production-grade real-time inference pipeline for PAD.
    Fuses deep feature representations with physical spatial/frequency evidence.
    """
    def __init__(self, model_path: Optional[str] = None, config: Optional[dict] = None, device: str = 'cpu'):
        self.config = config or {}
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path) if model_path else None
        self.is_onnx = self.model_path.suffix == '.onnx' if self.model_path else False
        self.lock = threading.Lock()
        
        # Performance Thresholds & Controls
        self.high_conf_threshold = self.config.get('high_conf_threshold', 0.75)
        self.low_conf_threshold = self.config.get('low_conf_threshold', 0.30)
        self.frame_skip = self.config.get('frame_skip', 2)
        self.img_size = self.config.get('image_size', 224)
        
        # Static transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if available and specified
        if self.model_path and self.model_path.exists():
            self.model = self._load_model()
        else:
            self.model = None
        
    def _load_model(self) -> Any:
        if self.is_onnx:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            return ort.InferenceSession(str(self.model_path), providers=providers)
        else:
            from src.models.pad_model import PADBaselineModel
            model = PADBaselineModel(model_type='small', pretrained=False, num_classes=2)
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded PyTorch checkpoint: {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint into model architecture: {e}")
            
            model.to(self.device)
            model.eval()
            return model

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_frame)
        return tensor.unsqueeze(0)

    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs multimodal passive PAD on a face ROI frame.
        Fuses neural network outputs with physical evidence (spatial, frequency, color integrity).
        """
        start_time = time.perf_counter()
        
        # 1. Compute Physical Spatial/Frequency Evidence
        phys_evidence = compute_physical_evidence(frame)
        phys_liveness = phys_evidence['liveness_score']
        phys_spoof = phys_evidence['spoof_score']
        
        # 2. Run Model Inference if available
        if self.model is not None:
            input_tensor = self.preprocess(frame)
            with self.lock:
                if self.is_onnx:
                    input_data = input_tensor.numpy()
                    input_name = self.model.get_inputs()[0].name
                    outputs = self.model.run(None, {input_name: input_data})[0]
                    exp_scores = np.exp(outputs[0] - np.max(outputs[0]))
                    probs = exp_scores / exp_scores.sum()
                else:
                    input_tensor = input_tensor.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
            model_real = float(probs[0])
            model_spoof = float(probs[1])
            
            # Configurable NN vs physical evidence fusion weight
            # When a trained model is loaded, NN features have 65% weight, physical evidence 35%
            nn_w = self.config.get('nn_weight', 0.65)
            ph_w = 1.0 - nn_w
            final_real_prob = nn_w * model_real + ph_w * phys_liveness
            final_spoof_prob = nn_w * model_spoof + ph_w * phys_spoof
        else:
            final_real_prob = phys_liveness
            final_spoof_prob = phys_spoof
            
        # Normalize
        total = final_real_prob + final_spoof_prob
        final_real_prob /= total
        final_spoof_prob /= total
        
        confidence = float(max(final_real_prob, final_spoof_prob))
        
        # 3-Class Calibrated Operating Point (BONA_FIDE, PRESENTATION_ATTACK, UNCERTAIN)
        prob_margin = abs(final_real_prob - final_spoof_prob)
        
        if confidence < 0.55 or prob_margin < 0.10:
            decision = "UNCERTAIN"
            early_exit = False
        elif final_real_prob > final_spoof_prob:
            decision = "BONA_FIDE"
            early_exit = (confidence >= self.high_conf_threshold)
        else:
            decision = "PRESENTATION_ATTACK"
            early_exit = (confidence >= self.high_conf_threshold)
            
        latency = (time.perf_counter() - start_time) * 1000.0
        
        return {
            'decision': decision,
            'confidence': confidence,
            'early_exit': early_exit,
            'head_scores': [final_real_prob, final_spoof_prob],
            'evidence_breakdown': phys_evidence,
            'latency_ms': latency
        }

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        decisions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % self.frame_skip == 0:
                res = self.predict_frame(frame)
                res['frame_idx'] = frame_idx
                decisions.append(res)
                
            frame_idx += 1
            
        cap.release()
        
        if not decisions:
            return {"error": "No frames processed."}
            
        spoof_count = sum(1 for d in decisions if d['decision'] == 'PRESENTATION_ATTACK')
        real_count = sum(1 for d in decisions if d['decision'] == 'BONA_FIDE')
        
        if spoof_count > real_count:
            final_decision = "PRESENTATION_ATTACK"
        elif real_count > spoof_count:
            final_decision = "BONA_FIDE"
        else:
            final_decision = "UNCERTAIN"
            
        avg_latency = sum(d['latency_ms'] for d in decisions) / len(decisions)
        
        return {
            'final_decision': final_decision,
            'per_frame_decisions': decisions,
            'evidence': f"Spoof frames: {spoof_count}, Bona fide frames: {real_count}",
            'avg_latency_ms': avg_latency
        }
