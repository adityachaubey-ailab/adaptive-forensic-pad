"""
Converts a real face image into a clean Near-Infrared (NIR / 850nm IR) video stream
simulating NIR active LED diffuse scattering on human skin.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

def create_ir_video_from_image(img_path: str = "dataset/real_test_sample.jpg", output_path: str = "dataset/real_ir_face_video.mp4", duration_sec: int = 4, fps: int = 30):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return
        
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simulate clean NIR 850nm active LED illumination on human skin
    nir_channel = cv2.convertScaleAbs(gray, alpha=1.05, beta=5)
    
    # 3-channel monochrome NIR frame
    nir_bgr = cv2.cvtColor(nir_channel, cv2.COLOR_GRAY2BGR)
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    print(f"Generating Clean NIR 850nm Real Face Video ({total_frames} frames)...")

    for i in range(total_frames):
        # Subtle head motion
        dx = int(2.0 * np.sin(i * 0.1))
        dy = int(1.5 * np.cos(i * 0.15))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(nir_bgr, M, (w, h))
        writer.write(frame)

    writer.release()
    print(f"Real IR Face Video saved to: {out_path}")

if __name__ == "__main__":
    create_ir_video_from_image()
