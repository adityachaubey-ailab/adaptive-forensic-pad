"""
PyTorch datasets for PAD Engine.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
from .metadata import MetadataManager

def extract_frames(video_path: str, num_frames: int = 1) -> np.ndarray:
    """Extracts frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1 # Fallback
        
    if num_frames == 1:
        # Just grab the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path}")
        
    while len(frames) < num_frames:
        frames.append(frames[-1])
        
    return np.array(frames)

class PADDataset(Dataset):
    """Dataset for single-image or multi-frame presentation attack detection."""
    
    def __init__(self, metadata_csv: str, split: str, frames_per_video: int = 20, transform: Optional[Callable] = None):
        self.manager = MetadataManager(metadata_csv)
        self.df = self.manager.filter(split=split)
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        # Build index of (row_idx, frame_num)
        self.samples = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            path = str(row["video_path"])
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                for f_idx in range(self.frames_per_video):
                    self.samples.append((idx, f_idx))
            else:
                self.samples.append((idx, 0))
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, sample_idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row_idx, frame_idx = self.samples[sample_idx]
        row = self.df.iloc[row_idx]
        path = str(row["video_path"])
        label = int(row["label"])
        
        if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_num = int((frame_idx + 0.5) * total_frames / self.frames_per_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_num, total_frames - 1))
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.imread(path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                
        if self.transform:
            if hasattr(self.transform, '__call__'):
                # Albumentations format
                try:
                    augmented = self.transform(image=image)
                    image = augmented.get("image", image)
                except Exception:
                    image = self.transform(image)
        else:
            # Default Resize (224x224) + Normalize for MobileNetV3
            image = cv2.resize(image, (224, 224))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # ImageNet mean & std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            
        metadata_dict = row.to_dict()
        return image, label, metadata_dict

class PADVideoDataset(Dataset):
    """Dataset for video-based presentation attack detection."""
    
    def __init__(self, metadata_csv: str, split: str, num_frames: int = 8, transform: Optional[Callable] = None):
        self.manager = MetadataManager(metadata_csv)
        self.df = self.manager.filter(split=split)
        self.num_frames = num_frames
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        path = str(row["video_path"])
        label = int(row["label"])
        
        frames = extract_frames(path, num_frames=self.num_frames)
        
        processed_frames = []
        for frame in frames:
            if self.transform:
                augmented = self.transform(image=frame)
                processed_frames.append(augmented["image"])
            else:
                # Basic conversion if no transform provided
                processed_frames.append(torch.from_numpy(frame).permute(2, 0, 1).float())
                
        frame_sequence_tensor = torch.stack(processed_frames)
            
        metadata_dict = row.to_dict()
        return frame_sequence_tensor, label, metadata_dict

def get_balanced_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Returns a WeightedRandomSampler that handles class imbalance."""
    if hasattr(dataset, "samples") and hasattr(dataset, "df"):
        labels = np.array([int(dataset.df.iloc[r_idx]["label"]) for r_idx, _ in dataset.samples])
    elif hasattr(dataset, "df") and "label" in dataset.df.columns:
        labels = dataset.df["label"].values.astype(int)
    else:
        return None
        
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1) # Avoid division by zero
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def get_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Creates train/val/test dataloaders with proper samplers.
    Expected config keys:
    - metadata_csv: path to CSV
    - batch_size: int
    - num_workers: int
    - train_transform: albumentations transform
    - val_transform: albumentations transform
    - dataset_type: 'image' or 'video'
    - num_frames: int (if video)
    """
    metadata_csv = config.get("metadata_csv")
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    train_transform = config.get("train_transform")
    val_transform = config.get("val_transform")
    dataset_type = config.get("dataset_type", "image")
    
    loaders = {}
    splits = ["train", "validation", "test"]
    transforms = {"train": train_transform, "validation": val_transform, "test": val_transform}
    
    for split in splits:
        if dataset_type == "video":
            dataset = PADVideoDataset(
                metadata_csv, split=split, 
                num_frames=config.get("num_frames", 8), 
                transform=transforms[split]
            )
        else:
            dataset = PADDataset(
                metadata_csv, split=split, 
                transform=transforms[split]
            )
            
        if len(dataset) == 0:
            continue
            
        if split == "train":
            sampler = get_balanced_sampler(dataset)
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler, 
                num_workers=num_workers, drop_last=True
            )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers
            )
            
    return loaders
