"""
Generates a sample Infrared (IR / NIR) video stream file for testing.
"""

import cv2
import numpy as np
from pathlib import Path

def generate_ir_sample(output_path: str = "dataset/sample_ir_test_video.mp4", duration_sec: int = 5, fps: int = 30):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    print(f"Generating synthetic IR video stream ({total_frames} frames)...")

    for i in range(total_frames):
        # Create a single-channel NIR simulation frame (grayscale face structure + uniform diffuse scattering)
        frame_gray = np.full((height, width), 40, dtype=np.uint8)
        
        # Face ellipse
        cv2.ellipse(frame_gray, (320, 240), (120, 160), 0, 0, 360, 180, -1)
        # Eyes
        cv2.circle(frame_gray, (270, 200), 20, 70, -1)
        cv2.circle(frame_gray, (370, 200), 20, 70, -1)
        # Nose
        cv2.ellipse(frame_gray, (320, 250), (15, 30), 0, 0, 360, 210, -1)
        # Mouth
        cv2.ellipse(frame_gray, (320, 320), (40, 15), 0, 0, 360, 100, -1)
        
        # Convert to 3-channel monochrome (typical IR sensor output)
        frame_ir = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        writer.write(frame_ir)

    writer.release()
    print(f"Sample IR video saved to: {output_path}")

if __name__ == "__main__":
    generate_ir_sample()
