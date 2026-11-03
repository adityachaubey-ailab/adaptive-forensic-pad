import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Any

def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resizes and normalizes a frame to [0, 1] float range."""
    resized = cv2.resize(frame, target_size)
    if resized.dtype == np.uint8:
        normalized = resized.astype(np.float32) / 255.0
    else:
        normalized = resized.astype(np.float32)
    return normalized

def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    """Applies ImageNet normalization to an image (assuming shape H, W, C and range [0, 1])."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (image - mean) / std
    return normalized

def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Computes the 2D FFT log-magnitude spectrum of an image.
    If image is color, converts to grayscale first.
    """
    if len(image.shape) == 3:
        if image.dtype == np.float32 and image.max() <= 1.0:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

def compute_dct_features(image: np.ndarray) -> np.ndarray:
    """Computes 2D DCT features of an image.
    Returns the block-wise DCT coefficients.
    """
    if len(image.shape) == 3:
        if image.dtype == np.float32 and image.max() <= 1.0:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    gray = np.float32(gray)
    dct = cv2.dct(gray)
    return dct

def extract_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """Extracts a color distribution histogram from an image."""
    if image.dtype == np.float32 and image.max() <= 1.0:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)
        
    hist_features = []
    for i in range(3): # Assuming 3 channels
        hist = cv2.calcHist([img_uint8], [i], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        hist_features.append(hist.flatten())
        
    return np.concatenate(hist_features)

def frame_to_tensor(frame: np.ndarray, transforms: Optional[Any] = None) -> torch.Tensor:
    """Converts a numpy array frame to a PyTorch tensor (C, H, W).
    Applies transforms if provided.
    """
    if transforms:
        from PIL import Image
        if frame.dtype == np.float32 and frame.max() <= 1.0:
            frame_img = Image.fromarray((frame * 255).astype(np.uint8))
        else:
            frame_img = Image.fromarray(frame.astype(np.uint8))
        tensor = transforms(frame_img)
    else:
        # Convert HWC to CHW
        if len(frame.shape) == 3:
            frame = np.transpose(frame, (2, 0, 1))
        tensor = torch.from_numpy(frame).float()
    return tensor
