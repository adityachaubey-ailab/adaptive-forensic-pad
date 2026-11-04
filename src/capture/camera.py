import cv2
import time
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CameraCapture:
    """Camera capture module using OpenCV."""

    def __init__(self, device_id: int = 0, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
        """
        Initialize CameraCapture.

        Args:
            device_id: Camera device index.
            resolution: Tuple of (width, height).
            fps: Target frames per second.
        """
        self.device_id = device_id
        self.resolution = resolution
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        
        self.frame_count: int = 0
        self.start_time: float = 0.0
        self.actual_fps: float = 0.0
        
    def start(self) -> bool:
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera with device ID {self.device_id}")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Verify resolution
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.resolution = (int(actual_width), int(actual_height))
        
        self.frame_count = 0
        self.start_time = time.time()
        logger.info(f"Camera started: {self.resolution} @ {self.target_fps}fps")
        return True
        
    def stop(self) -> None:
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera stopped.")
        
    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """Read a single frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.actual_fps = self.frame_count / elapsed
        else:
            logger.warning("Failed to grab frame.")
            
        return ret, frame
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about the current capture session."""
        return {
            "device_id": self.device_id,
            "resolution": self.resolution,
            "target_fps": self.target_fps,
            "actual_fps": round(self.actual_fps, 2),
            "frame_count": self.frame_count
        }
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
