import cv2
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VideoRecorder:
    """Video recording with metadata generation."""
    
    def __init__(self, output_dir: str, device_info: Dict[str, Any], subject_id: str, 
                 attack_type: str, lighting: str, environment: str):
        """
        Initialize VideoRecorder.
        
        Args:
            output_dir: Directory to save the video and metadata.
            device_info: Camera device info dict.
            subject_id: Identifier for the subject.
            attack_type: 'bona_fide', 'print', 'replay', etc.
            lighting: Description of lighting.
            environment: Description of the environment.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device_info = device_info
        self.subject_id = subject_id
        self.attack_type = attack_type
        self.lighting = lighting
        self.environment = environment
        
        self.sample_id = str(uuid.uuid4())
        self.video_path = self.output_dir / f"{self.sample_id}.mp4"
        self.metadata_path = self.output_dir / f"{self.sample_id}.json"
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.frames_written: int = 0
        self.start_time: float = 0.0
        
        self.face_detected_frames: int = 0
        self.face_sizes: list = []
        
    def start_recording(self) -> None:
        """Initialize the video writer."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        res = self.device_info.get("resolution", (640, 480))
        fps = self.device_info.get("target_fps", 30)
        
        self.writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, res)
        self.frames_written = 0
        self.start_time = time.time()
        
        logger.info(f"Started recording to {self.video_path}")
        
    def add_frame(self, frame: Any, face_detected: bool = False, face_w: int = 0, face_h: int = 0) -> None:
        """
        Write a frame to the video.
        
        Args:
            frame: OpenCV BGR frame.
            face_detected: Boolean indicating if a face was detected in this frame.
            face_w: Width of the face bounding box in pixels.
            face_h: Height of the face bounding box in pixels.
        """
        if self.writer is not None:
            self.writer.write(frame)
            self.frames_written += 1
            if face_detected:
                self.face_detected_frames += 1
                self.face_sizes.append((face_w, face_h))
                
    def stop_recording(self) -> None:
        """Finalize video recording and write metadata."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
        duration = time.time() - self.start_time
        
        avg_w = sum(w for w, h in self.face_sizes) / len(self.face_sizes) if self.face_sizes else 0
        avg_h = sum(h for w, h in self.face_sizes) / len(self.face_sizes) if self.face_sizes else 0
        
        metadata = {
            "sample_id": self.sample_id,
            "device": f"Device_{self.device_info.get('device_id')}",
            "camera": "Unknown",
            "fps": self.device_info.get("target_fps", 30),
            "resolution": f"{self.device_info.get('resolution', (640, 480))[0]}x{self.device_info.get('resolution', (640, 480))[1]}",
            "total_frames": self.frames_written,
            "duration_seconds": round(duration, 2),
            "lighting": self.lighting,
            "environment": self.environment,
            "attack_type": self.attack_type,
            "subject_id": self.subject_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.start_time)),
            "face_detected_frames": self.face_detected_frames,
            "avg_face_size": [int(avg_w), int(avg_h)]
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Finished recording. Metadata saved to {self.metadata_path}")
