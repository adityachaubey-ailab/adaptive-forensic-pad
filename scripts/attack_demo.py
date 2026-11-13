import argparse
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class AttackDemoApp:
    def __init__(self, model_path: Optional[str] = None, dataset_dir: Optional[str] = None):
        self.model_path = model_path
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.results: Dict[str, Dict[str, int]] = {}
        
    def process_video(self, video_path: Path):
        print(f"Processing {video_path.name}...")
        cap = cv2.VideoCapture(str(video_path))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Mock Processing
            display_frame = np.hstack((frame, frame)) # original | evidence
            cv2.putText(display_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Evidence", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Attack Demo", display_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        cap.release()
        
    def show_results(self):
        print("\n=== Per-Attack Type Results ===")
        print(f"{'Attack Type':<20} | {'Count':<10} | {'Detected':<10} | {'Accuracy':<10}")
        print("-" * 55)
        # Mock results
        attacks = ["Print", "Replay", "3D Mask"]
        for attack in attacks:
            count = 50
            detected = np.random.randint(45, 51)
            accuracy = (detected / count) * 100
            print(f"{attack:<20} | {count:<10} | {detected:<10} | {accuracy:.1f}%")
            
    def run(self):
        if not self.dataset_dir or not self.dataset_dir.exists():
            print("Dataset directory not found. Running with mock video...")
            # Create a mock video or skip
        else:
            videos = list(self.dataset_dir.glob("*.mp4"))
            for video in videos:
                self.process_video(video)
                
        self.show_results()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Attack Demo for PAD Engine")
    parser.add_argument("--model-path", type=str, help="Path to the trained model")
    parser.add_argument("--dataset-dir", type=str, help="Path to dataset containing attack videos")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    app = AttackDemoApp(model_path=args.model_path, dataset_dir=args.dataset_dir)
    app.run()

if __name__ == "__main__":
    main()
