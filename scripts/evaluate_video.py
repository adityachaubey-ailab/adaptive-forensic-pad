"""
Video Evaluation & Visual Annotation Tool

Processes any video file (RGB or Infrared/IR), runs the PAD pipeline,
and saves an annotated output MP4 video with real-time decision overlays,
bounding boxes, and physical evidence breakdowns.

Usage:
    python scripts/evaluate_video.py --video-path dataset/sample_test_video.mp4
    python scripts/evaluate_video.py --video-path dataset/sample_test_video.mp4 --output outputs/annotated_video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.face_detector import FaceDetector
from src.inference.pipeline import InferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate & Annotate Video for PAD")
    parser.add_argument("--video-path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model weights (.pth / .onnx)")
    parser.add_argument("--output", type=str, default="outputs/annotated_video.mp4", help="Path for output annotated MP4")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    
    if not video_path.exists():
        print(f"ERROR: Video file not found at {video_path}")
        sys.exit(1)
        
    print("=" * 60)
    print("  Adaptive Forensic PAD Engine — Video Evaluation Tool")
    print(f"  Input Video:  {video_path}")
    print(f"  Output MP4:   {args.output}")
    print("=" * 60)

    # Initialize Detector & Pipeline
    face_detector = FaceDetector(min_detection_confidence=0.5)
    pipeline = InferencePipeline(model_path=args.model_path, device=args.device)

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"\nProcessing {total_frames} frames ({width}x{height} @ {fps:.1f} FPS)...")

    frame_idx = 0
    decisions = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        t0 = time.perf_counter()
        
        # Face Detection
        detections = face_detector.detect(frame)
        
        if detections:
            best_det = max(detections, key=lambda d: d.score)
            face_roi = face_detector.extract_roi(frame, best_det, padding=0.2)
            
            bbox = best_det.bbox
            box_x = max(0, int(bbox[0] * width))
            box_y = max(0, int(bbox[1] * height))
            box_w = int(bbox[2] * width)
            box_h = int(bbox[3] * height)
            
            if face_roi.size > 0:
                result = pipeline.predict_frame(face_roi)
                decision = result['decision']
                confidence = result['confidence']
                probs = result['head_scores']
                evidence = result.get('evidence_breakdown', {})
                decisions.append(result)
                
                color = (0, 230, 0) if decision == "BONA_FIDE" else (0, 230, 230) if decision == "UNCERTAIN" else (0, 0, 230)
                
                # Bounding Box & Label
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
                cv2.putText(frame, f"{decision} ({confidence*100:.1f}%)", (box_x, max(box_y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Overlay Evidence Summary
                mode_str = "IR/NIR" if evidence.get('is_ir') else "RGB"
                cv2.putText(frame, f"Mode: {mode_str} | Tex: {evidence.get('texture', 0):.2f} | FFT: {evidence.get('frequency', 0):.2f}",
                            (box_x, box_y + box_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames} | Latency: {t_elapsed_ms:.1f}ms",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        writer.write(frame)

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print("  VIDEO EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Processed Frames: {frame_idx}")
    print(f"  Total Execution Time: {total_time:.2f} s ({frame_idx / total_time:.1f} FPS)")
    
    if decisions:
        real_cnt = sum(1 for d in decisions if d['decision'] == 'BONA_FIDE')
        spoof_cnt = sum(1 for d in decisions if d['decision'] == 'PRESENTATION_ATTACK')
        unc_cnt = sum(1 for d in decisions if d['decision'] == 'UNCERTAIN')
        
        print(f"  Bona Fide Frames:      {real_cnt} ({real_cnt/len(decisions)*100:.1f}%)")
        print(f"  Presentation Attacks: {spoof_cnt} ({spoof_cnt/len(decisions)*100:.1f}%)")
        print(f"  Uncertain Frames:     {unc_cnt} ({unc_cnt/len(decisions)*100:.1f}%)")
        
        final_verdict = "BONA_FIDE" if real_cnt > spoof_cnt else "PRESENTATION_ATTACK" if spoof_cnt > real_cnt else "UNCERTAIN"
        print(f"  FINAL VIDEO VERDICT:   {final_verdict}")
        
    print(f"  Annotated Video Saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
