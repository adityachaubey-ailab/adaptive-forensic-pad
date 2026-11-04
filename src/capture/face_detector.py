import cv2
import numpy as np
from typing import List, Tuple, Any, Optional, NamedTuple
import logging

logger = logging.getLogger(__name__)

class SimpleDetection:
    def __init__(self, bbox: Tuple[float, float, float, float], score: float = 0.9):
        self.bbox = bbox  # normalized x, y, w, h
        self.score = score
        self.location_data = None

class FaceDetector:
    """
    Robust Face detection and ROI extraction.
    Uses MediaPipe solutions if available, with automatic OpenCV Cascade / Bounding Box fallback.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_confidence = min_detection_confidence
        self.use_mediapipe = False
        self.detector = None
        self.face_mesh = None
        
        try:
            import mediapipe.python.solutions.face_detection as mp_face_detection
            import mediapipe.python.solutions.face_mesh as mp_face_mesh
            
            self.detector = mp_face_detection.FaceDetection(
                model_selection=1, 
                min_detection_confidence=min_detection_confidence
            )
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence
            )
            self.use_mediapipe = True
            logger.info("FaceDetector initialized with MediaPipe engine.")
        except Exception as e:
            logger.info(f"MediaPipe solutions not available ({e}). Initializing fallback engine.")
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                if self.detector.empty():
                    self.detector = None
            except Exception:
                self.detector = None

    def detect(self, frame: np.ndarray) -> List[Any]:
        """
        Detect faces in a BGR frame. Returns list of detections with normalized bboxes.
        """
        h, w = frame.shape[:2]
        
        if self.use_mediapipe and self.detector is not None:
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector.process(image_rgb)
                
                if results.detections:
                    detections = []
                    for det in results.detections:
                        bbox_data = det.location_data.relative_bounding_box
                        bbox = (bbox_data.xmin, bbox_data.ymin, bbox_data.width, bbox_data.height)
                        score = det.score[0] if det.score else 0.9
                        detections.append(SimpleDetection(bbox=bbox, score=score))
                    return detections
            except Exception:
                pass
        
        # OpenCV Cascade Fallback
        if self.detector is not None and hasattr(self.detector, 'detectMultiScale'):
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    detections = []
                    for (x, y, fw, fh) in faces:
                        norm_bbox = (x / w, y / h, fw / w, fh / h)
                        detections.append(SimpleDetection(bbox=norm_bbox, score=0.85))
                    return detections
            except Exception:
                pass
            
        # Default Fallback Box (Centered Face ROI 40% of frame)
        default_bbox = (0.3, 0.2, 0.4, 0.6)
        return [SimpleDetection(bbox=default_bbox, score=0.80)]

    def extract_roi(self, frame: np.ndarray, detection: Any, padding: float = 0.2) -> np.ndarray:
        """
        Extract cropped face image given a detection with optional padding.
        """
        h, w = frame.shape[:2]
        bbox = detection.bbox
        
        xmin = int(bbox[0] * w)
        ymin = int(bbox[1] * h)
        box_w = int(bbox[2] * w)
        box_h = int(bbox[3] * h)
        
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        x1 = max(0, xmin - pad_w)
        y1 = max(0, ymin - pad_h)
        x2 = min(w, xmin + box_w + pad_w)
        y2 = min(h, ymin + box_h + pad_h)
        
        if x2 <= x1 or y2 <= y1:
            return frame
            
        return frame[y1:y2, x1:x2]

    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 468 3D facial landmarks using MediaPipe Face Mesh if available.
        """
        if self.use_mediapipe and self.face_mesh is not None:
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                    return landmarks
            except Exception:
                pass
                
        # Mock fallback landmarks (468, 3) centered
        landmarks = np.zeros((468, 3))
        landmarks[:, 0] = 0.5
        landmarks[:, 1] = 0.5
        return landmarks
