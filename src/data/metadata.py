"""
Dataset metadata schema and management for the PAD Engine.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AttackType(str, Enum):
    BONA_FIDE = "bona_fide"
    PRINT = "print"
    REPLAY = "replay"
    SCREEN = "screen"
    MASK = "mask"
    DEEPFAKE = "deepfake"
    INJECTION = "injection"
    UNKNOWN = "unknown"

class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

@dataclass
class SampleMetadata:
    """Dataclass representing metadata for a single dataset sample."""
    sample_id: str
    subject_id: str
    attack_type: str
    device: str
    camera: str
    resolution: str
    lighting: str
    environment: str
    dataset_source: str
    split: str
    label: int  # 0=bona_fide, 1=attack
    video_path: str
    metadata_path: Optional[str] = None

def validate_no_leakage(metadata_df: pd.DataFrame) -> bool:
    """
    Validates that there is no data leakage across splits.
    Checks that subject_id doesn't appear in multiple splits.
    Also ensures no session from the same attack recording appears in both train and test.
    Prints warnings about class imbalance if ratio > 10:1.
    """
    leakage = False
    
    if "subject_id" in metadata_df.columns and "split" in metadata_df.columns:
        subject_splits = metadata_df.groupby("subject_id")["split"].unique()
        leaking_subjects = subject_splits[subject_splits.apply(len) > 1]
        
        if not leaking_subjects.empty:
            logger.warning(f"Data leakage detected! {len(leaking_subjects)} subjects appear in multiple splits.")
            for subject, splits in leaking_subjects.items():
                logger.warning(f"Subject {subject} appears in splits: {list(splits)}")
            leakage = True
            
    if "sample_id" in metadata_df.columns and "split" in metadata_df.columns:
        # Assuming sample_id uniquely identifies a recording session
        sample_splits = metadata_df.groupby("sample_id")["split"].unique()
        leaking_sessions = sample_splits[sample_splits.apply(len) > 1]
        if not leaking_sessions.empty:
            logger.warning(f"Session leakage detected! {len(leaking_sessions)} sessions appear in multiple splits.")
            leakage = True
            
    if "label" in metadata_df.columns:
        label_counts = metadata_df["label"].value_counts()
        if len(label_counts) == 2:
            min_count = label_counts.min()
            max_count = label_counts.max()
            if min_count > 0:
                ratio = max_count / min_count
                if ratio > 10:
                    logger.warning(f"Severe class imbalance detected: ratio is {ratio:.2f}:1")
                    
    return not leakage

class MetadataManager:
    """Manages dataset metadata."""
    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        if self.csv_path.exists():
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.DataFrame(columns=[
                "sample_id", "subject_id", "attack_type", "device", "camera", 
                "resolution", "lighting", "environment", "dataset_source", 
                "split", "label", "video_path", "metadata_path"
            ])
            
    def save(self) -> None:
        """Saves the metadata DataFrame to CSV."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.csv_path, index=False)
        
    def add_sample(self, sample: SampleMetadata) -> None:
        """Adds a new sample to the metadata."""
        new_row = pd.DataFrame([sample.__dict__])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
    def validate(self) -> bool:
        """Validates the metadata for leakage and issues."""
        return validate_no_leakage(self.df)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Generates statistics about the dataset."""
        stats = {}
        if not self.df.empty:
            stats["total_samples"] = len(self.df)
            if "attack_type" in self.df.columns:
                stats["per_attack_type"] = self.df["attack_type"].value_counts().to_dict()
            if "split" in self.df.columns:
                stats["per_split"] = self.df["split"].value_counts().to_dict()
            if "subject_id" in self.df.columns:
                stats["unique_subjects"] = self.df["subject_id"].nunique()
        return stats
        
    def filter(self, attack_type: Optional[str] = None, split: Optional[str] = None, 
               subject_id: Optional[str] = None, device: Optional[str] = None) -> pd.DataFrame:
        """Filters the metadata based on criteria."""
        filtered = self.df.copy()
        if attack_type:
            filtered = filtered[filtered["attack_type"] == attack_type]
        if split:
            filtered = filtered[filtered["split"] == split]
        if subject_id:
            filtered = filtered[filtered["subject_id"] == subject_id]
        if device:
            filtered = filtered[filtered["device"] == device]
        return filtered
