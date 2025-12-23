"""
Performance and timing utilities
"""
import time
import logging
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FPSCounter:
    """Track and calculate FPS"""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def tick(self) -> float:
        """
        Update FPS counter
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.last_time = current_time
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            return fps
        return 0.0


class PerformanceMonitor:
    """Monitor performance of different components"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.timings: Dict[str, deque] = {}
        self.window_size = 30
    
    def start_timer(self, name: str):
        """Start timing a component"""
        if name not in self.timings:
            self.timings[name] = deque(maxlen=self.window_size)
        
        self.timings[f"_{name}_start"] = time.time()
    
    def stop_timer(self, name: str):
        """Stop timing a component"""
        start_key = f"_{name}_start"
        if start_key in self.timings:
            elapsed = time.time() - self.timings[start_key]
            self.timings[name].append(elapsed)
    
    def get_avg_time(self, name: str) -> float:
        """Get average time for component"""
        if name in self.timings and len(self.timings[name]) > 0:
            return sum(self.timings[name]) / len(self.timings[name])
        return 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics for all components"""
        stats = {}
        for name, times in self.timings.items():
            if not name.startswith('_') and len(times) > 0:
                stats[name] = self.get_avg_time(name)
        return stats
    
    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        logger.info("Performance Statistics:")
        for name, avg_time in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {name}: {avg_time*1000:.2f} ms")


class FrameBuffer:
    """Buffer for storing recent frames"""
    
    def __init__(self, max_size: int = 30):
        """
        Initialize frame buffer
        
        Args:
            max_size: Maximum number of frames to store
        """
        self.max_size = max_size
        self.frames = deque(maxlen=max_size)
    
    def add(self, frame):
        """Add frame to buffer"""
        self.frames.append(frame)
    
    def get_recent(self, n: int = 1):
        """Get n most recent frames"""
        return list(self.frames)[-n:]
    
    def get_all(self):
        """Get all frames in buffer"""
        return list(self.frames)
    
    def clear(self):
        """Clear buffer"""
        self.frames.clear()
