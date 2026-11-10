import time
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class LatencyStats:
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_measurements(cls, measurements: List[float]) -> 'LatencyStats':
        if not measurements:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.array(measurements)
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99))
        )

class LatencyProfiler:
    """
    Profiler for measuring the latency of various pipeline stages.
    """
    def __init__(self):
        self.measurements: Dict[str, List[float]] = {
            'camera_capture_ms': [],
            'decode_ms': [],
            'face_detection_ms': [],
            'preprocessing_ms': [],
            'encoder_ms': [],
            'texture_head_ms': [],
            'frequency_head_ms': [],
            'geometry_head_ms': [],
            'temporal_ms': [],
            'fusion_ms': [],
            'postprocessing_ms': [],
            'total_ms': []
        }
        self.stage_mapping = {
            'camera_capture': 'camera_capture_ms',
            'decode': 'decode_ms',
            'face_detection': 'face_detection_ms',
            'preprocessing': 'preprocessing_ms',
            'encoder': 'encoder_ms',
            'texture_head': 'texture_head_ms',
            'frequency_head': 'frequency_head_ms',
            'geometry_head': 'geometry_head_ms',
            'temporal': 'temporal_ms',
            'fusion': 'fusion_ms',
            'postprocessing': 'postprocessing_ms',
            'total': 'total_ms'
        }

    @contextmanager
    def measure(self, stage: str):
        """
        Context manager to measure the latency of a block of code.
        """
        key = self.stage_mapping.get(stage, f"{stage}_ms")
        if key not in self.measurements:
            self.measurements[key] = []
        
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000.0
        self.measurements[key].append(elapsed_ms)

    def report(self) -> Dict[str, Any]:
        """
        Generate a statistical report of the latencies.
        """
        stats = {}
        total_mean = 0.0
        for stage, data in self.measurements.items():
            if not data:
                continue
            stage_stats = LatencyStats.from_measurements(data)
            stats[stage] = stage_stats
            
            if stage == 'total_ms':
                total_mean = stage_stats.mean
                
        if 'total_ms' in stats and stats['total_ms'].mean > 0:
            effective_fps = 1000.0 / stats['total_ms'].mean
        else:
            effective_fps = 0.0
            
        return {
            'stages': stats,
            'effective_fps': effective_fps,
            'total_pipeline_fps': effective_fps
        }

    def reset(self):
        """Clear all measurements."""
        for key in self.measurements.keys():
            self.measurements[key] = []
