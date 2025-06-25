"""
Neural Network Profiling Tools
==============================

Comprehensive profiling and performance analysis tools for neural network implementations.
Provides detailed timing, memory usage, and CPU utilization analysis.
"""

import cProfile
import pstats
import io
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, field
from memory_profiler import profile as memory_profile
from contextlib import contextmanager
import threading
import os
import gc
import pickle
from functools import wraps


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    cpu_percent: float
    memory_usage_mb: float
    peak_memory_mb: float
    cache_misses: Optional[int] = None
    instructions_per_second: Optional[float] = None
    function_calls: Dict[str, int] = field(default_factory=dict)
    hotspots: List[str] = field(default_factory=list)


@dataclass
class ProfilingResult:
    """Container for complete profiling results."""
    name: str
    metrics: PerformanceMetrics
    detailed_stats: Optional[pstats.Stats] = None
    memory_profile: Optional[List[Tuple[float, float]]] = None
    timing_breakdown: Dict[str, float] = field(default_factory=dict)


class SystemMonitor:
    """Monitor system resources during profiling."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Tuple[List[float], List[float], List[float]]:
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.timestamps, self.cpu_samples, self.memory_samples
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        process = psutil.Process()
        start_time = time.time()
        
        while self.monitoring:
            current_time = time.time() - start_time
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.timestamps.append(current_time)
            self.cpu_samples.append(cpu_percent)
            self.memory_samples.append(memory_mb)
            
            time.sleep(self.interval)


class MemoryProfiler:
    """Memory usage profiler for neural network operations."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def profile_memory_usage(func: Callable) -> Callable:
        """Decorator to profile memory usage of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            gc.collect()  # Clean up before measurement
            initial_memory = MemoryProfiler.get_memory_usage()
            
            result = func(*args, **kwargs)
            
            gc.collect()  # Clean up after execution
            final_memory = MemoryProfiler.get_memory_usage()
            peak_memory = max(initial_memory, final_memory)
            
            memory_used = final_memory - initial_memory
            
            if hasattr(wrapper, 'memory_stats'):
                wrapper.memory_stats.append({
                    'initial_memory': initial_memory,
                    'final_memory': final_memory,
                    'peak_memory': peak_memory,
                    'memory_used': memory_used
                })
            else:
                wrapper.memory_stats = [{
                    'initial_memory': initial_memory,
                    'final_memory': final_memory,
                    'peak_memory': peak_memory,
                    'memory_used': memory_used
                }]
            
            return result
        return wrapper


class CPUProfiler:
    """CPU performance profiler with detailed analysis."""
    
    def __init__(self):
        self.profiler = None
        self.stats = None
    
    @contextmanager
    def profile(self):
        """Context manager for CPU profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        try:
            yield self
        finally:
            self.profiler.disable()
            
            # Convert to stats object
            stats_stream = io.StringIO()
            self.stats = pstats.Stats(self.profiler, stream=stats_stream)
            self.stats.sort_stats('cumulative')
    
    def get_stats(self) -> pstats.Stats:
        """Get profiling statistics."""
        return self.stats
    
    def get_hotspots(self, top_n: int = 10) -> List[str]:
        """Get top CPU hotspots."""
        if not self.stats:
            return []
        
        stats_data = self.stats.get_stats_profile()
        sorted_stats = sorted(stats_data.func_profiles.items(), 
                            key=lambda x: x[1].cumulative, reverse=True)
        
        hotspots = []
        for i, (func_name, profile_data) in enumerate(sorted_stats[:top_n]):
            filename, line_no, func = func_name
            hotspots.append(f"{func} ({filename}:{line_no}) - {profile_data.cumulative:.4f}s")
        
        return hotspots
    
    def get_function_calls(self) -> Dict[str, int]:
        """Get function call counts."""
        if not self.stats:
            return {}
        
        stats_data = self.stats.get_stats_profile()
        function_calls = {}
        
        for func_name, profile_data in stats_data.func_profiles.items():
            filename, line_no, func = func_name
            function_calls[func] = profile_data.ncalls
        
        return function_calls


class NeuralNetworkProfiler:
    """Comprehensive profiler for neural network implementations."""
    
    def __init__(self, name: str = "Neural Network"):
        self.name = name
        self.system_monitor = SystemMonitor()
        self.cpu_profiler = CPUProfiler()
        self.results = []
    
    def profile_function(self, func: Callable, *args, **kwargs) -> ProfilingResult:
        """Profile a single function execution."""
        print(f"Profiling {func.__name__}...")
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Measure execution time and CPU profile
        start_time = time.time()
        
        with self.cpu_profiler.profile():
            result = func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        
        # Stop system monitoring
        timestamps, cpu_samples, memory_samples = self.system_monitor.stop_monitoring()
        
        # Compile metrics
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        avg_memory = np.mean(memory_samples) if memory_samples else 0
        peak_memory = max(memory_samples) if memory_samples else 0
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            cpu_percent=avg_cpu,
            memory_usage_mb=avg_memory,
            peak_memory_mb=peak_memory,
            function_calls=self.cpu_profiler.get_function_calls(),
            hotspots=self.cpu_profiler.get_hotspots()
        )
        
        profiling_result = ProfilingResult(
            name=func.__name__,
            metrics=metrics,
            detailed_stats=self.cpu_profiler.get_stats(),
            memory_profile=list(zip(timestamps, memory_samples))
        )
        
        self.results.append(profiling_result)
        return profiling_result
    
    def profile_training_loop(self, model, x_train, y_train, epochs: int = 5) -> ProfilingResult:
        """Profile a complete training loop with detailed breakdown."""
        print(f"Profiling training loop for {epochs} epochs...")
        
        # Start comprehensive monitoring
        self.system_monitor.start_monitoring()
        
        timing_breakdown = {
            'forward_pass': 0,
            'backward_pass': 0,
            'parameter_update': 0,
            'data_loading': 0
        }
        
        total_start_time = time.time()
        
        with self.cpu_profiler.profile():
            for epoch in range(epochs):
                # Forward pass timing
                forward_start = time.time()
                predictions = model.forward(x_train)
                timing_breakdown['forward_pass'] += time.time() - forward_start
                
                # Loss computation
                loss = model.compute_loss(y_train, predictions)
                
                # Backward pass timing
                backward_start = time.time()
                model.backward(x_train, y_train, predictions)
                timing_breakdown['backward_pass'] += time.time() - backward_start
        
        total_execution_time = time.time() - total_start_time
        
        # Stop monitoring
        timestamps, cpu_samples, memory_samples = self.system_monitor.stop_monitoring()
        
        # Compile comprehensive metrics
        metrics = PerformanceMetrics(
            execution_time=total_execution_time,
            cpu_percent=np.mean(cpu_samples) if cpu_samples else 0,
            memory_usage_mb=np.mean(memory_samples) if memory_samples else 0,
            peak_memory_mb=max(memory_samples) if memory_samples else 0,
            function_calls=self.cpu_profiler.get_function_calls(),
            hotspots=self.cpu_profiler.get_hotspots()
        )
        
        profiling_result = ProfilingResult(
            name="training_loop",
            metrics=metrics,
            detailed_stats=self.cpu_profiler.get_stats(),
            memory_profile=list(zip(timestamps, memory_samples)),
            timing_breakdown=timing_breakdown
        )
        
        self.results.append(profiling_result)
        return profiling_result
    
    def benchmark_operations(self, operations: Dict[str, Callable]) -> Dict[str, ProfilingResult]:
        """Benchmark multiple operations for comparison."""
        print("Benchmarking multiple operations...")
        
        results = {}
        for op_name, operation in operations.items():
            print(f"  Benchmarking {op_name}...")
            
            # Run multiple iterations for statistical significance
            times = []
            memory_usage = []
            
            for _ in range(5):  # 5 iterations
                start_time = time.time()
                initial_memory = MemoryProfiler.get_memory_usage()
                
                operation()
                
                execution_time = time.time() - start_time
                final_memory = MemoryProfiler.get_memory_usage()
                
                times.append(execution_time)
                memory_usage.append(final_memory - initial_memory)
            
            # Compute statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_memory = np.mean(memory_usage)
            
            metrics = PerformanceMetrics(
                execution_time=avg_time,
                cpu_percent=0,  # Not measured in benchmark mode
                memory_usage_mb=avg_memory,
                peak_memory_mb=max(memory_usage)
            )
            
            results[op_name] = ProfilingResult(
                name=op_name,
                metrics=metrics
            )
            
            print(f"    Time: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"    Memory: {avg_memory:.2f}MB")
        
        return results
    
    def compare_implementations(self, baseline_func: Callable, optimized_func: Callable,
                              *args, **kwargs) -> Dict[str, Any]:
        """Compare baseline vs optimized implementations."""
        print("Comparing baseline vs optimized implementations...")
        
        # Profile baseline
        baseline_result = self.profile_function(baseline_func, *args, **kwargs)
        
        # Profile optimized
        optimized_result = self.profile_function(optimized_func, *args, **kwargs)
        
        # Compute speedup metrics
        speedup = baseline_result.metrics.execution_time / optimized_result.metrics.execution_time
        memory_improvement = (baseline_result.metrics.peak_memory_mb - 
                            optimized_result.metrics.peak_memory_mb) / baseline_result.metrics.peak_memory_mb * 100
        
        comparison = {
            'baseline': baseline_result,
            'optimized': optimized_result,
            'speedup': speedup,
            'memory_improvement_percent': memory_improvement,
            'execution_time_improvement': baseline_result.metrics.execution_time - optimized_result.metrics.execution_time
        }
        
        print(f"\nComparison Results:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time improvement: {comparison['execution_time_improvement']:.4f}s")
        print(f"  Memory improvement: {memory_improvement:.1f}%")
        
        return comparison
    
    def generate_report(self, output_dir: str = "/home/bc/neural_network_optimization/profiling") -> None:
        """Generate comprehensive profiling report."""
        print(f"Generating profiling report in {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(f"{output_dir}/profiling_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Generate visualizations
        self._create_performance_plots(output_dir)
        
        # Generate text report
        self._create_text_report(output_dir)
        
        print("Profiling report generated successfully!")
    
    def _create_performance_plots(self, output_dir: str) -> None:
        """Create performance visualization plots."""
        if not self.results:
            return
        
        # Set up plot style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time comparison
        names = [result.name for result in self.results]
        times = [result.metrics.execution_time for result in self.results]
        
        axes[0, 0].bar(names, times)
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        memory_usage = [result.metrics.peak_memory_mb for result in self.results]
        
        axes[0, 1].bar(names, memory_usage)
        axes[0, 1].set_title('Peak Memory Usage Comparison')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # CPU utilization
        cpu_usage = [result.metrics.cpu_percent for result in self.results]
        
        axes[1, 0].bar(names, cpu_usage)
        axes[1, 0].set_title('Average CPU Utilization')
        axes[1, 0].set_ylabel('CPU %')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Memory profile over time (if available)
        if self.results and self.results[0].memory_profile:
            timestamps, memory_values = zip(*self.results[0].memory_profile)
            axes[1, 1].plot(timestamps, memory_values)
            axes[1, 1].set_title('Memory Usage Over Time')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_text_report(self, output_dir: str) -> None:
        """Create detailed text report."""
        with open(f"{output_dir}/profiling_report.txt", 'w') as f:
            f.write("Neural Network Performance Profiling Report\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"Function: {result.name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Execution Time: {result.metrics.execution_time:.4f} seconds\n")
                f.write(f"CPU Utilization: {result.metrics.cpu_percent:.1f}%\n")
                f.write(f"Average Memory: {result.metrics.memory_usage_mb:.2f} MB\n")
                f.write(f"Peak Memory: {result.metrics.peak_memory_mb:.2f} MB\n")
                
                if result.metrics.hotspots:
                    f.write("\nTop CPU Hotspots:\n")
                    for i, hotspot in enumerate(result.metrics.hotspots[:5], 1):
                        f.write(f"  {i}. {hotspot}\n")
                
                if result.timing_breakdown:
                    f.write("\nTiming Breakdown:\n")
                    for operation, time_spent in result.timing_breakdown.items():
                        f.write(f"  {operation}: {time_spent:.4f}s\n")
                
                f.write("\n" + "=" * 50 + "\n\n")


def profile_matrix_operations():
    """Profile basic matrix operations for baseline comparison."""
    sizes = [100, 500, 1000, 2000]
    profiler = NeuralNetworkProfiler("Matrix Operations")
    
    operations = {}
    
    for size in sizes:
        # Create random matrices
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)
        
        # Define operations
        operations[f"matmul_{size}x{size}"] = lambda: np.dot(A, B)
        operations[f"add_{size}x{size}"] = lambda: A + B
        operations[f"elementwise_mul_{size}x{size}"] = lambda: A * B
        operations[f"transpose_{size}x{size}"] = lambda: A.T
    
    results = profiler.benchmark_operations(operations)
    profiler.generate_report()
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Running matrix operations profiling...")
    profile_matrix_operations()