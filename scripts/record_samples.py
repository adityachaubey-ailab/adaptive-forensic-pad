import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import argparse
import csv
from src.capture import CameraCapture, FaceDetector, VideoRecorder

def main():
    parser = argparse.ArgumentParser(description="Record video samples for PAD engine.")
    parser.add_argument("--subject", type=str, default="S001", help="Subject ID (e.g. S001)")
    parser.add_argument("--attack", type=str, default="bona_fide", help="Attack type (e.g. bona_fide, print, replay, screen, mask)")
    parser.add_argument("--lighting", type=str, default="indoor", help="Lighting conditions")
    parser.add_argument("--environment", type=str, default="office", help="Environment type")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, val, test)")
    parser.add_argument("--device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output", type=str, default=None, help="Custom direct output video path (optional)")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    if args.output:
        output_file = Path(args.output)
        output_dir = output_file.parent
    else:
        output_dir = project_root / "dataset" / args.split / args.attack
    
    csv_path = project_root / "dataset" / "metadata.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    camera = CameraCapture(device_id=args.device)
    detector = FaceDetector()
    
    recorder = None
    recording = False
    
    print("=" * 60)
    print(f"  PAD Video Recorder — Subject: {args.subject} | Type: {args.attack}")
    print("  Controls:")
    print("    [R] : Start / Stop Recording")
    print("    [Q] : Quit")
    print("=" * 60)
    
    with camera:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                continue
                
            display_frame = frame.copy()
            h, w = frame.shape[:2]
            detections = detector.detect(frame)
            
            face_detected = len(detections) > 0
            face_w, face_h = 0, 0
            
            if face_detected:
                best_det = detections[0]
                bx, by, bw, bh = best_det.bbox
                x1 = max(0, int(bx * w))
                y1 = max(0, int(by * h))
                face_w = int(bw * w)
                face_h = int(bh * h)
                
                cv2.rectangle(display_frame, (x1, y1), (x1 + face_w, y1 + face_h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face ({best_det.score*100:.0f}%)", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            info = camera.get_info()
            status_text = "● RECORDING" if recording else "READY"
            status_color = (0, 0, 255) if recording else (0, 255, 0)
            
            cv2.putText(display_frame, f"Status: {status_text}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_frame, f"FPS: {info.get('actual_fps', 30.0):.1f} | Cam: {args.device}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press 'r' to Record | 'q' to Quit", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            
            if recording and recorder:
                recorder.add_frame(frame, face_detected, face_w, face_h)
                
            cv2.imshow("PAD Sample Recording", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if recording and recorder:
                    recorder.stop_recording()
                break
            elif key == ord('r'):
                if not recording:
                    recorder = VideoRecorder(
                        output_dir=str(output_dir),
                        device_info=info,
                        subject_id=args.subject,
                        attack_type=args.attack,
                        lighting=args.lighting,
                        environment=args.environment
                    )
                    recorder.start_recording()
                    recording = True
                    print(">> Recording STARTED...")
                else:
                    recorder.stop_recording()
                    recording = False
                    
                    # Update CSV
                    file_exists = csv_path.exists()
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["sample_id", "subject_id", "attack_type", "split"])
                        writer.writerow([recorder.sample_id, args.subject, args.attack, args.split])
                        
                    print(f">> Recording STOPPED. Sample saved: {recorder.sample_id}")
                    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
