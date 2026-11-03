import logging
import sys
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console."""
    
    COLORS = {
        'DEBUG': '\033[94m',      # Blue
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[1;91m', # Bold Red
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_fmt = f"%(asctime)s - {self.COLORS.get(record.levelname, self.RESET)}%(levelname)s{self.RESET} - %(name)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str, log_dir: str = 'outputs/logs', level: str = 'INFO') -> logging.Logger:
    """Sets up a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
    file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_fmt)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Gets an existing logger."""
    return logging.getLogger(name)
