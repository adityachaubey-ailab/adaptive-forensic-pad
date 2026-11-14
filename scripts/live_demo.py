"""
Live WebCam Presentation Attack Detection (PAD) Demo Application

Connects MediaPipe Face Detection + Landmarks + Real Multimodal Passive PAD Pipeline.
Renders real-time evidence breakdown (Texture, 2D FFT Frequency, Color Integrity, Real/Spoof probabilities),
FPS, and latency stats.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.face_detector import FaceDetector
from src.inference.pipeline import InferencePipeline


class LiveDemoApp:
    def __init__(self, camera_id: int = 0, resolution: tuple = (1280, 720), model_path: Optional[str] = None):
        self.camera_id = camera_id
        self.resolution = resolution
        self.model_path = model_path
        
        # Initialize OpenCV Camera
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Initialize Real Face Detector
        print("[LiveDemo] Initializing Face Detector & Landmark Engine...")
        self.face_detector = FaceDetector(min_detection_confidence=0.5)
        
        # Initialize Inference Pipeline
        if self.model_path and Path(self.model_path).exists():
            print(f"[LiveDemo] Loading model weights from {self.model_path}...")
            self.pipeline = InferencePipeline(model_path=self.model_path)
            self.model_loaded = True
        else:
            default_onnx = PROJECT_ROOT / "outputs" / "models" / "streaming_baseline_model_opt_opt.onnx"
            default_ckpt = PROJECT_ROOT / "outputs" / "models" / "streaming_baseline_model.pth"
            if default_onnx.exists():
                print(f"[LiveDemo] Loading high-speed ONNX model from {default_onnx}...")
                self.pipeline = InferencePipeline(model_path=str(default_onnx))
                self.model_loaded = True
            elif default_ckpt.exists():
                print(f"[LiveDemo] Loading checkpoint from {default_ckpt}...")
                self.pipeline = InferencePipeline(model_path=str(default_ckpt))
                self.model_loaded = True
            else:
                print("[LiveDemo] Initializing pipeline with multimodal physical evidence fusion engine.")
                self.pipeline = InferencePipeline()
                self.model_loaded = False
        
        # UI Toggles
        self.show_benchmark = True
        self.show_evidence = True
        self.show_landmarks = False
        
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

    def draw_bar(self, frame, x, y, width, height, value, name, invert_color=False):
        """Draw score progress bar overlay."""
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        fill_width = int(width * min(max(value, 0.0), 1.0))
        
        if invert_color:
            color = (0, 0, 230) if value > 0.5 else (0, 230, 0)
        else:
            color = (0, 230, 0) if value > 0.5 else (0, 0, 230)
            
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
        cv2.putText(frame, f"{name}: {value:.2f}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        start_process = time.perf_counter()
        
        # 1. Face Detection
        detections = self.face_detector.detect(frame)
        
        if detections:
            best_det = max(detections, key=lambda d: d.score)
            face_roi = self.face_detector.extract_roi(frame, best_det, padding=0.2)
            
            bbox = best_det.bbox
            box_x = max(0, int(bbox[0] * w))
            box_y = max(0, int(bbox[1] * h))
            box_w = int(bbox[2] * w)
            box_h = int(bbox[3] * h)
            
            # 2. Run Multimodal Passive PAD Inference on Face ROI
            if face_roi.size > 0:
                result = self.pipeline.predict_frame(face_roi)
                decision = result['decision']
                confidence = result['confidence']
                probs = result['head_scores']
                phys_breakdown = result.get('evidence_breakdown', {})
                latency_ms = result['latency_ms']
                
                # Decision color: Green = BONA_FIDE, Yellow = UNCERTAIN, Red = PRESENTATION_ATTACK
                if decision == "BONA_FIDE":
                    color = (0, 230, 0)
                elif decision == "UNCERTAIN":
                    color = (0, 230, 230)
                else:
                    color = (0, 0, 230)
                
                # Draw Bounding Box & Label
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
                label_text = f"{decision} ({confidence*100:.1f}%)"
                cv2.putText(frame, label_text, (box_x, max(box_y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 3. Draw Multimodal Evidence Overlay Bars
                if self.show_evidence and len(probs) >= 2:
                    bar_x = min(box_x + box_w + 15, w - 120)
                    bar_y = box_y
                    
                    real_prob = probs[0]
                    spoof_prob = probs[1]
                    
                    tex_score = phys_breakdown.get('texture', 0.8)
                    freq_score = phys_breakdown.get('frequency', 0.8)
                    color_score = phys_breakdown.get('color', 0.8)
                    
                    self.draw_bar(frame, bar_x, bar_y, 100, 10, real_prob, "Real Prob")
                    self.draw_bar(frame, bar_x, bar_y + 25, 100, 10, spoof_prob, "Spoof Prob", invert_color=True)
                    self.draw_bar(frame, bar_x, bar_y + 50, 100, 10, tex_score, "Texture")
                    self.draw_bar(frame, bar_x, bar_y + 75, 100, 10, freq_score, "FFT Freq")
                    self.draw_bar(frame, bar_x, bar_y + 100, 100, 10, color_score, "Color Space")
            
            # Draw Facial Landmarks if toggled
            if self.show_landmarks:
                landmarks = self.face_detector.extract_landmarks(frame)
                if landmarks is not None:
                    for pt in landmarks[::10]:
                        lx, ly = int(pt[0] * w), int(pt[1] * h)
                        cv2.circle(frame, (lx, ly), 1, (255, 255, 0), -1)
        else:
            cv2.putText(frame, "NO FACE DETECTED — PLEASE FACE CAMERA", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            latency_ms = (time.perf_counter() - start_process) * 1000.0

        # Benchmark HUD Metrics
        if self.show_benchmark:
            hud_y = 30
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (15, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Pipeline Latency: {latency_ms:.1f}ms", (15, hud_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Device: {self.pipeline.device.type.upper()}", (15, hud_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, "Controls: 'q'=Quit | 'b'=HUD | 'e'=Evidence | 'l'=Landmarks", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return frame

    def run(self):
        print("\n[LiveDemo] Starting live camera loop. Press 'q' to exit.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[LiveDemo] Failed to grab frame from camera.")
                break
                
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.start_time = time.time()
                self.frame_count = 0
                
            out_frame = self.process_frame(frame)
            
            cv2.imshow("PAD Engine Live SIH Demo", out_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                self.show_benchmark = not self.show_benchmark
            elif key == ord('e'):
                self.show_evidence = not self.show_evidence
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live WebCam PAD Demo")
    parser.add_argument("--model-path", type=str, help="Path to trained PyTorch (.pth) or ONNX (.onnx) model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda)")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID index")
    parser.add_argument("--resolution", type=str, default="1280x720", help="Camera resolution WxH")
    args = parser.parse_args()
    
    res_w, res_h = map(int, args.resolution.split('x'))
    
    app = LiveDemoApp(camera_id=args.camera_id, resolution=(res_w, res_h), model_path=args.model_path)
    app.run()


if __name__ == "__main__":
    main()
