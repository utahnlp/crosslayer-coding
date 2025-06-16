"""Performance profiling utilities for CLT training."""

import time
import torch
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Timer:
    """Simple timer context manager for measuring execution time."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.start_time: Optional[float] = None
        self.cuda_start_event: Optional[torch.cuda.Event] = None
        self.cuda_end_event: Optional[torch.cuda.Event] = None
        
    def __enter__(self):
        if not self.enabled:
            return self
            
        if torch.cuda.is_available():
            # Use CUDA events for more accurate GPU timing
            self.cuda_start_event = torch.cuda.Event(enable_timing=True)
            self.cuda_end_event = torch.cuda.Event(enable_timing=True)
            self.cuda_start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
            
        if torch.cuda.is_available() and self.cuda_start_event and self.cuda_end_event:
            self.cuda_end_event.record()
            torch.cuda.synchronize()
            self.elapsed = self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000.0  # Convert to seconds
        else:
            self.elapsed = time.perf_counter() - self.start_time if self.start_time else 0.0


class TrainingProfiler:
    """Profiler for tracking performance metrics during CLT training."""
    
    def __init__(self, enabled: bool = True, log_interval: int = 100):
        self.enabled = enabled
        self.log_interval = log_interval
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.step_count = 0
        
    def record(self, name: str, duration: float):
        """Record a timing measurement."""
        if not self.enabled:
            return
        self.timings[name].append(duration)
        
    def timer(self, name: str) -> Timer:
        """Create a timer context manager."""
        return Timer(name, enabled=self.enabled)
        
    def step(self):
        """Increment step counter and log if needed."""
        if not self.enabled:
            return
            
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self.log_summary()
            
    def log_summary(self):
        """Log timing summary and clear buffers."""
        if not self.timings:
            return
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Performance Profile (last {self.log_interval} steps)")
        logger.info(f"{'='*60}")
        
        # Calculate statistics
        total_time = 0.0
        timing_stats = {}
        
        for name, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                total_time += avg_time
                timing_stats[name] = {
                    'avg': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'count': len(times)
                }
        
        # Sort by average time
        sorted_stats = sorted(timing_stats.items(), key=lambda x: x[1]['avg'], reverse=True)
        
        # Log each timing
        for name, stats in sorted_stats:
            pct = (stats['avg'] / total_time * 100) if total_time > 0 else 0
            logger.info(
                f"{name:.<30} "
                f"avg: {stats['avg']*1000:>7.2f}ms "
                f"min: {stats['min']*1000:>7.2f}ms "
                f"max: {stats['max']*1000:>7.2f}ms "
                f"({pct:>5.1f}%)"
            )
            
        logger.info(f"{'='*60}")
        logger.info(f"Total average step time: {total_time*1000:.2f}ms")
        logger.info(f"{'='*60}\n")
        
        # Clear timings for next interval
        self.timings.clear()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get timing summary as a dictionary."""
        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times),
                    'count': len(times)
                }
        return summary


class CUDAMemoryProfiler:
    """Profiler for tracking CUDA memory usage."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.peak_memory = 0.0
        self.allocated_history: List[float] = []
        
    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if not self.enabled:
            return
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        self.allocated_history.append(allocated)
        self.peak_memory = max(self.peak_memory, allocated)
        
        if label:
            logger.debug(f"[{label}] CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
    def log_summary(self):
        """Log memory usage summary."""
        if not self.enabled or not self.allocated_history:
            return
            
        avg_allocated = sum(self.allocated_history) / len(self.allocated_history)
        logger.info(f"\nCUDA Memory Summary:")
        logger.info(f"  Peak allocated: {self.peak_memory:.2f}GB")
        logger.info(f"  Average allocated: {avg_allocated:.2f}GB")
        logger.info(f"  Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        logger.info(f"  Current reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")


class DistributedProfiler:
    """Profiler specifically for distributed operations."""
    
    def __init__(self, enabled: bool = True, rank: int = 0):
        self.enabled = enabled
        self.rank = rank
        self.timings: Dict[str, List[float]] = defaultdict(list)
        
    @contextmanager
    def profile_op(self, op_name: str):
        """Context manager to profile a distributed operation."""
        if not self.enabled:
            yield
            return
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        elapsed = time.perf_counter() - start_time
        self.timings[op_name].append(elapsed)
        
    def log_summary(self):
        """Log summary of distributed operations."""
        if not self.enabled or not self.timings:
            return
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Distributed Operations Profile (Rank {self.rank})")
        logger.info(f"{'='*60}")
        
        for op_name, times in sorted(self.timings.items()):
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                total_time = sum(times)
                
                logger.info(
                    f"{op_name:.<30} "
                    f"avg: {avg_time*1000:>7.2f}ms "
                    f"min: {min_time*1000:>7.2f}ms "
                    f"max: {max_time*1000:>7.2f}ms "
                    f"total: {total_time:.2f}s "
                    f"calls: {len(times)}"
                )
                
        logger.info(f"{'='*60}")


@contextmanager
def profile_activation_function(profiler: Optional['TrainingProfiler'], name: str):
    """Context manager to profile activation functions."""
    if profiler is None or not profiler.enabled:
        yield
        return
        
    with profiler.timer(name) as timer:
        yield
        
    if hasattr(timer, 'elapsed'):
        profiler.record(name, timer.elapsed)