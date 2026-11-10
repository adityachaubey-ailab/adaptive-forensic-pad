import psutil
import os
from typing import Dict, Any, Optional
import time
import threading

class ResourceMonitor:
    """
    Monitors RAM, CPU, and VRAM usage.
    """
    def __init__(self):
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self.history = {
            'cpu': [],
            'ram_mb': [],
            'vram_mb': []
        }
        
    def _monitor_loop(self):
        process = psutil.Process(os.getpid())
        while self._monitoring:
            self.history['cpu'].append(process.cpu_percent(interval=None))
            self.history['ram_mb'].append(process.memory_info().rss / (1024 * 1024))
            
            try:
                import torch
                if torch.cuda.is_available():
                    vram = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.history['vram_mb'].append(vram)
            except ImportError:
                pass
                
            time.sleep(0.1)

    def start(self):
        """Start monitoring in a background thread."""
        self.history = {'cpu': [], 'ram_mb': [], 'vram_mb': []}
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        psutil.Process(os.getpid()).cpu_percent(interval=None)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._monitoring = False
        if self._thread is not None:
            self._thread.join()

    def measure_model_size(self, model_path: str) -> float:
        """Measure model size on disk in MB."""
        if os.path.exists(model_path):
            return os.path.getsize(model_path) / (1024 * 1024)
        return 0.0
        
    def measure_model_parameters(self, model: Any) -> int:
        """Measure total number of parameters in a PyTorch model."""
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            return 0

    def report(self) -> Dict[str, Any]:
        """Generate a resource usage report."""
        import numpy as np
        
        def safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0
            
        def safe_max(lst):
            return float(np.max(lst)) if lst else 0.0

        return {
            'cpu_mean_percent': safe_mean(self.history['cpu']),
            'cpu_max_percent': safe_max(self.history['cpu']),
            'ram_mean_mb': safe_mean(self.history['ram_mb']),
            'ram_max_mb': safe_max(self.history['ram_mb']),
            'vram_mean_mb': safe_mean(self.history['vram_mb']),
            'vram_max_mb': safe_max(self.history['vram_mb'])
        }
