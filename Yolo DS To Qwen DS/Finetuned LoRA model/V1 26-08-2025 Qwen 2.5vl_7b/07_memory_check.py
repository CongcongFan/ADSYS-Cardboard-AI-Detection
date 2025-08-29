#!/usr/bin/env python3
"""
Memory monitoring and system check script for LoRA operations.
Provides comprehensive system analysis, memory monitoring, and recommendations
for optimal LoRA merge or direct inference performance.
"""

import os
import sys
import argparse
import json
import time
import platform
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
import psutil
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import torch for GPU info
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU information will be limited")

class SystemAnalyzer:
    """Comprehensive system analysis for LoRA operations."""
    
    def __init__(self):
        self.system_info = {}
        self.memory_info = {}
        self.gpu_info = {}
        self.disk_info = {}
        self.recommendations = []
    
    def collect_system_info(self):
        """Collect basic system information."""
        logger.info("Collecting system information...")
        
        self.system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
        }
        
        # CPU frequency info
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self.system_info['cpu_freq_current'] = cpu_freq.current
                self.system_info['cpu_freq_min'] = cpu_freq.min
                self.system_info['cpu_freq_max'] = cpu_freq.max
        except:
            pass
    
    def collect_memory_info(self):
        """Collect detailed memory information."""
        logger.info("Collecting memory information...")
        
        # RAM information
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        self.memory_info = {
            'ram_total_gb': ram.total / (1024**3),
            'ram_available_gb': ram.available / (1024**3),
            'ram_used_gb': (ram.total - ram.available) / (1024**3),
            'ram_used_percent': ram.percent,
            'ram_free_gb': ram.free / (1024**3),
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_free_gb': swap.free / (1024**3),
            'swap_percent': swap.percent,
        }
    
    def collect_gpu_info(self):
        """Collect GPU information if available."""
        logger.info("Collecting GPU information...")
        
        self.gpu_info = {
            'torch_available': TORCH_AVAILABLE,
            'cuda_available': False,
            'gpu_count': 0,
            'gpus': []
        }
        
        if TORCH_AVAILABLE:
            self.gpu_info['cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                self.gpu_info['gpu_count'] = torch.cuda.device_count()
                self.gpu_info['cuda_version'] = torch.version.cuda
                self.gpu_info['cudnn_version'] = torch.backends.cudnn.version()
                
                # Collect info for each GPU
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    
                    gpu_info = {
                        'index': i,
                        'name': gpu_props.name,
                        'total_memory_gb': gpu_memory / (1024**3),
                        'major': gpu_props.major,
                        'minor': gpu_props.minor,
                        'multi_processor_count': gpu_props.multi_processor_count,
                    }
                    
                    # Try to get current memory usage
                    try:
                        torch.cuda.set_device(i)
                        gpu_info['allocated_memory_gb'] = torch.cuda.memory_allocated(i) / (1024**3)
                        gpu_info['reserved_memory_gb'] = torch.cuda.memory_reserved(i) / (1024**3)
                        gpu_info['free_memory_gb'] = gpu_info['total_memory_gb'] - gpu_info['reserved_memory_gb']
                    except:
                        gpu_info['allocated_memory_gb'] = 0
                        gpu_info['reserved_memory_gb'] = 0
                        gpu_info['free_memory_gb'] = gpu_info['total_memory_gb']
                    
                    self.gpu_info['gpus'].append(gpu_info)
        
        # Alternative GPU detection (for non-CUDA GPUs)
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.strip().split(', ')
                        if len(parts) >= 2:
                            name = parts[0]
                            memory_mb = float(parts[1])
                            # Add to GPU info if not already detected
                            if i >= len(self.gpu_info['gpus']):
                                self.gpu_info['gpus'].append({
                                    'index': i,
                                    'name': name,
                                    'total_memory_gb': memory_mb / 1024,
                                    'source': 'nvidia-smi'
                                })
        except:
            pass
    
    def collect_disk_info(self, path: str = "."):
        """Collect disk space information."""
        logger.info(f"Collecting disk information for path: {path}")
        
        try:
            disk_usage = shutil.disk_usage(path)
            
            self.disk_info = {
                'path': os.path.abspath(path),
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': (disk_usage.total - disk_usage.free) / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'used_percent': ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100,
            }
        except Exception as e:
            logger.error(f"Error collecting disk info: {e}")
            self.disk_info = {'error': str(e)}
    
    def analyze_requirements(self) -> Dict[str, Any]:
        """Analyze system requirements for different LoRA operations."""
        logger.info("Analyzing system requirements...")
        
        analysis = {
            'merge_feasibility': {},
            'direct_inference_feasibility': {},
            'recommendations': []
        }
        
        # Memory requirements (approximate)
        REQUIREMENTS = {
            'merge': {
                'min_ram_gb': 8.0,
                'recommended_ram_gb': 16.0,
                'min_gpu_gb': 4.0,
                'recommended_gpu_gb': 8.0,
                'min_disk_gb': 20.0,
                'recommended_disk_gb': 50.0,
            },
            'direct_inference': {
                'min_ram_gb': 4.0,
                'recommended_ram_gb': 8.0,
                'min_gpu_gb': 2.0,
                'recommended_gpu_gb': 4.0,
                'min_disk_gb': 10.0,
                'recommended_disk_gb': 20.0,
            }
        }
        
        # Analyze merge feasibility
        merge_req = REQUIREMENTS['merge']
        analysis['merge_feasibility'] = {
            'ram_sufficient': self.memory_info['ram_available_gb'] >= merge_req['min_ram_gb'],
            'ram_recommended': self.memory_info['ram_available_gb'] >= merge_req['recommended_ram_gb'],
            'gpu_sufficient': False,
            'gpu_recommended': False,
            'disk_sufficient': self.disk_info.get('free_gb', 0) >= merge_req['min_disk_gb'],
            'disk_recommended': self.disk_info.get('free_gb', 0) >= merge_req['recommended_disk_gb'],
            'overall_feasible': False,
            'use_cpu_recommended': False,
        }
        
        # GPU analysis for merge
        if self.gpu_info['cuda_available'] and self.gpu_info['gpus']:
            best_gpu = max(self.gpu_info['gpus'], key=lambda g: g.get('total_memory_gb', 0))
            gpu_memory = best_gpu.get('free_memory_gb', best_gpu.get('total_memory_gb', 0))
            
            analysis['merge_feasibility']['gpu_sufficient'] = gpu_memory >= merge_req['min_gpu_gb']
            analysis['merge_feasibility']['gpu_recommended'] = gpu_memory >= merge_req['recommended_gpu_gb']
        
        # Overall merge feasibility
        merge_feas = analysis['merge_feasibility']
        if merge_feas['ram_sufficient'] and merge_feas['disk_sufficient']:
            if merge_feas['gpu_sufficient']:
                merge_feas['overall_feasible'] = True
            elif merge_feas['ram_recommended']:  # Can use CPU if enough RAM
                merge_feas['overall_feasible'] = True
                merge_feas['use_cpu_recommended'] = True
        
        # Analyze direct inference feasibility
        infer_req = REQUIREMENTS['direct_inference']
        analysis['direct_inference_feasibility'] = {
            'ram_sufficient': self.memory_info['ram_available_gb'] >= infer_req['min_ram_gb'],
            'ram_recommended': self.memory_info['ram_available_gb'] >= infer_req['recommended_ram_gb'],
            'gpu_sufficient': False,
            'gpu_recommended': False,
            'disk_sufficient': self.disk_info.get('free_gb', 0) >= infer_req['min_disk_gb'],
            'disk_recommended': self.disk_info.get('free_gb', 0) >= infer_req['recommended_disk_gb'],
            'overall_feasible': False,
            'use_cpu_recommended': False,
        }
        
        # GPU analysis for inference
        if self.gpu_info['cuda_available'] and self.gpu_info['gpus']:
            best_gpu = max(self.gpu_info['gpus'], key=lambda g: g.get('total_memory_gb', 0))
            gpu_memory = best_gpu.get('free_memory_gb', best_gpu.get('total_memory_gb', 0))
            
            analysis['direct_inference_feasibility']['gpu_sufficient'] = gpu_memory >= infer_req['min_gpu_gb']
            analysis['direct_inference_feasibility']['gpu_recommended'] = gpu_memory >= infer_req['recommended_gpu_gb']
        
        # Overall inference feasibility
        infer_feas = analysis['direct_inference_feasibility']
        if infer_feas['ram_sufficient'] and infer_feas['disk_sufficient']:
            if infer_feas['gpu_sufficient']:
                infer_feas['overall_feasible'] = True
            else:
                infer_feas['overall_feasible'] = True
                infer_feas['use_cpu_recommended'] = True
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict[str, Any]):
        """Generate specific recommendations based on system analysis."""
        logger.info("Generating recommendations...")
        
        self.recommendations = []
        
        merge_feas = analysis['merge_feasibility']
        infer_feas = analysis['direct_inference_feasibility']
        
        # Memory recommendations
        if not merge_feas['ram_sufficient']:
            self.recommendations.append({
                'type': 'critical',
                'category': 'memory',
                'message': f"Insufficient RAM for LoRA merge. Available: {self.memory_info['ram_available_gb']:.1f}GB, Required: 8GB minimum",
                'action': 'Close other applications or use direct inference instead'
            })
        elif not merge_feas['ram_recommended']:
            self.recommendations.append({
                'type': 'warning',
                'category': 'memory',
                'message': f"RAM below recommended for LoRA merge. Available: {self.memory_info['ram_available_gb']:.1f}GB, Recommended: 16GB",
                'action': 'Consider using --use-cpu flag with memory-efficient merge script'
            })
        
        # GPU recommendations
        if self.gpu_info['cuda_available']:
            if not merge_feas['gpu_sufficient'] and merge_feas['ram_sufficient']:
                self.recommendations.append({
                    'type': 'warning',
                    'category': 'gpu',
                    'message': 'GPU memory insufficient for merge, but RAM is sufficient',
                    'action': 'Use --use-cpu flag with 02_merge_lora_memory_efficient.py'
                })
            elif not merge_feas['gpu_sufficient']:
                self.recommendations.append({
                    'type': 'info',
                    'category': 'gpu',
                    'message': 'Low GPU memory detected',
                    'action': 'Use direct inference (06_direct_lora_inference.py) for better memory efficiency'
                })
        else:
            self.recommendations.append({
                'type': 'info',
                'category': 'gpu',
                'message': 'No CUDA GPU detected',
                'action': 'Use CPU mode for all operations'
            })
        
        # Disk space recommendations
        if not merge_feas['disk_sufficient']:
            self.recommendations.append({
                'type': 'critical',
                'category': 'disk',
                'message': f"Insufficient disk space. Available: {self.disk_info.get('free_gb', 0):.1f}GB, Required: 20GB minimum",
                'action': 'Free up disk space or use a different output directory'
            })
        elif not merge_feas['disk_recommended']:
            self.recommendations.append({
                'type': 'warning',
                'category': 'disk',
                'message': f"Disk space below recommended. Available: {self.disk_info.get('free_gb', 0):.1f}GB, Recommended: 50GB",
                'action': 'Monitor disk usage during operations'
            })
        
        # Overall strategy recommendations
        if merge_feas['overall_feasible'] and infer_feas['overall_feasible']:
            if merge_feas['use_cpu_recommended']:
                self.recommendations.append({
                    'type': 'info',
                    'category': 'strategy',
                    'message': 'System can handle both merge and direct inference',
                    'action': 'Use 02_merge_lora_memory_efficient.py with --use-cpu for merge, or 06_direct_lora_inference.py for immediate use'
                })
            else:
                self.recommendations.append({
                    'type': 'info',
                    'category': 'strategy',
                    'message': 'System well-equipped for LoRA operations',
                    'action': 'Use 02_merge_lora_memory_efficient.py for merge, or 06_direct_lora_inference.py for immediate use'
                })
        elif infer_feas['overall_feasible']:
            self.recommendations.append({
                'type': 'info',
                'category': 'strategy',
                'message': 'System suitable for direct inference but not merge',
                'action': 'Use 06_direct_lora_inference.py for immediate model usage without merge'
            })
        else:
            self.recommendations.append({
                'type': 'critical',
                'category': 'strategy',
                'message': 'System may not meet minimum requirements',
                'action': 'Close other applications and try direct inference with --use-cpu flag'
            })
    
    def run_memory_stress_test(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Run a memory stress test to check system stability."""
        logger.info(f"Running memory stress test for {duration_seconds} seconds...")
        
        results = {
            'start_memory': self.get_current_memory(),
            'peak_memory': None,
            'end_memory': None,
            'memory_samples': [],
            'stable': True,
            'warnings': []
        }
        
        start_time = time.time()
        peak_ram_used = results['start_memory']['ram_used_gb']
        
        try:
            # Allocate some memory to test system response
            test_data = []
            chunk_size = 100 * 1024 * 1024  # 100MB chunks
            
            while time.time() - start_time < duration_seconds:
                current_memory = self.get_current_memory()
                results['memory_samples'].append({
                    'time': time.time() - start_time,
                    'memory': current_memory
                })
                
                # Track peak memory
                if current_memory['ram_used_gb'] > peak_ram_used:
                    peak_ram_used = current_memory['ram_used_gb']
                
                # Check for memory pressure
                if current_memory['ram_used_percent'] > 90:
                    results['warnings'].append(f"High RAM usage detected: {current_memory['ram_used_percent']:.1f}%")
                    results['stable'] = False
                
                # Allocate memory gradually
                if len(test_data) < 5 and current_memory['ram_used_percent'] < 80:
                    test_data.append(bytearray(chunk_size))
                
                time.sleep(1)
            
            # Clean up test data
            del test_data
            
        except MemoryError:
            results['warnings'].append("Memory allocation failed during stress test")
            results['stable'] = False
        except Exception as e:
            results['warnings'].append(f"Stress test error: {e}")
            results['stable'] = False
        
        results['end_memory'] = self.get_current_memory()
        results['peak_memory'] = {'ram_used_gb': peak_ram_used}
        
        logger.info("Memory stress test completed")
        return results
    
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage snapshot."""
        ram = psutil.virtual_memory()
        result = {
            'ram_total_gb': ram.total / (1024**3),
            'ram_available_gb': ram.available / (1024**3),
            'ram_used_gb': (ram.total - ram.available) / (1024**3),
            'ram_used_percent': ram.percent,
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            result['gpu_allocated_gb'] = torch.cuda.memory_allocated(0) / (1024**3)
            result['gpu_reserved_gb'] = torch.cuda.memory_reserved(0) / (1024**3)
        
        return result
    
    def run_full_analysis(self, path: str = ".", stress_test: bool = False) -> Dict[str, Any]:
        """Run complete system analysis."""
        logger.info("Starting comprehensive system analysis...")
        
        # Collect all system information
        self.collect_system_info()
        self.collect_memory_info()
        self.collect_gpu_info()
        self.collect_disk_info(path)
        
        # Analyze requirements
        analysis = self.analyze_requirements()
        
        # Generate recommendations
        self.generate_recommendations(analysis)
        
        # Run stress test if requested
        stress_results = None
        if stress_test:
            stress_results = self.run_memory_stress_test()
        
        # Compile full report
        report = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'memory_info': self.memory_info,
            'gpu_info': self.gpu_info,
            'disk_info': self.disk_info,
            'analysis': analysis,
            'recommendations': self.recommendations,
            'stress_test': stress_results
        }
        
        logger.info("System analysis completed")
        return report

class MemoryMonitor:
    """Real-time memory monitoring for LoRA operations."""
    
    def __init__(self, log_interval: int = 5):
        self.log_interval = log_interval
        self.monitoring = False
        self.start_time = None
        self.samples = []
    
    def start_monitoring(self, output_file: Optional[str] = None):
        """Start real-time memory monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.samples = []
        
        logger.info(f"Starting memory monitoring (interval: {self.log_interval}s)")
        if output_file:
            logger.info(f"Logging to file: {output_file}")
        
        try:
            while self.monitoring:
                sample = self.collect_sample()
                self.samples.append(sample)
                
                # Log sample
                self.log_sample(sample)
                
                # Write to file if specified
                if output_file:
                    self.write_sample_to_file(sample, output_file)
                
                time.sleep(self.log_interval)
                
        except KeyboardInterrupt:
            logger.info("Memory monitoring stopped by user")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        logger.info("Memory monitoring stopped")
    
    def collect_sample(self) -> Dict[str, Any]:
        """Collect a memory usage sample."""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        # RAM info
        ram = psutil.virtual_memory()
        sample = {
            'timestamp': current_time,
            'elapsed_seconds': elapsed,
            'ram_total_gb': ram.total / (1024**3),
            'ram_used_gb': (ram.total - ram.available) / (1024**3),
            'ram_used_percent': ram.percent,
            'ram_available_gb': ram.available / (1024**3)
        }
        
        # GPU info if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            sample['gpu_allocated_gb'] = torch.cuda.memory_allocated(0) / (1024**3)
            sample['gpu_reserved_gb'] = torch.cuda.memory_reserved(0) / (1024**3)
            sample['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Process info
        process = psutil.Process()
        sample['process_memory_gb'] = process.memory_info().rss / (1024**3)
        sample['process_cpu_percent'] = process.cpu_percent()
        
        return sample
    
    def log_sample(self, sample: Dict[str, Any]):
        """Log a memory sample."""
        elapsed = sample['elapsed_seconds']
        ram_used = sample['ram_used_percent']
        ram_gb = sample['ram_used_gb']
        process_gb = sample['process_memory_gb']
        
        log_msg = f"[{elapsed:6.1f}s] RAM: {ram_used:5.1f}% ({ram_gb:5.1f}GB), Process: {process_gb:5.2f}GB"
        
        if 'gpu_allocated_gb' in sample:
            gpu_gb = sample['gpu_allocated_gb']
            log_msg += f", GPU: {gpu_gb:5.1f}GB"
        
        logger.info(log_msg)
    
    def write_sample_to_file(self, sample: Dict[str, Any], filename: str):
        """Write sample to CSV file."""
        import csv
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = list(sample.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(sample)

def print_report(report: Dict[str, Any], format_type: str = "console"):
    """Print system analysis report."""
    
    if format_type == "json":
        print(json.dumps(report, indent=2, default=str))
        return
    
    # Console format
    print("=" * 80)
    print("SYSTEM ANALYSIS REPORT")
    print("=" * 80)
    
    # System info
    sys_info = report['system_info']
    print(f"System: {sys_info['platform']} {sys_info['architecture']}")
    print(f"Processor: {sys_info['processor']}")
    print(f"CPU Cores: {sys_info['cpu_count_physical']} physical, {sys_info['cpu_count_logical']} logical")
    print(f"Python: {sys_info['python_version']}")
    print()
    
    # Memory info
    mem_info = report['memory_info']
    print("MEMORY:")
    print(f"  RAM: {mem_info['ram_used_gb']:.1f}GB used / {mem_info['ram_total_gb']:.1f}GB total ({mem_info['ram_used_percent']:.1f}%)")
    print(f"  Available: {mem_info['ram_available_gb']:.1f}GB")
    if mem_info['swap_total_gb'] > 0:
        print(f"  Swap: {mem_info['swap_used_gb']:.1f}GB used / {mem_info['swap_total_gb']:.1f}GB total")
    print()
    
    # GPU info
    gpu_info = report['gpu_info']
    print("GPU:")
    if gpu_info['cuda_available']:
        print(f"  CUDA Available: Yes ({gpu_info['gpu_count']} GPU(s))")
        print(f"  CUDA Version: {gpu_info.get('cuda_version', 'Unknown')}")
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f"  GPU {i}: {gpu['name']}")
            print(f"    Memory: {gpu.get('free_memory_gb', gpu.get('total_memory_gb', 0)):.1f}GB available / {gpu.get('total_memory_gb', 0):.1f}GB total")
    else:
        print("  CUDA Available: No")
    print()
    
    # Disk info
    disk_info = report['disk_info']
    print("DISK SPACE:")
    if 'error' in disk_info:
        print(f"  Error: {disk_info['error']}")
    else:
        print(f"  Path: {disk_info['path']}")
        print(f"  Available: {disk_info['free_gb']:.1f}GB / {disk_info['total_gb']:.1f}GB ({disk_info['used_percent']:.1f}% used)")
    print()
    
    # Analysis
    analysis = report['analysis']
    print("OPERATION FEASIBILITY:")
    print("  LoRA Merge:")
    merge_feas = analysis['merge_feasibility']
    overall_status = "[FEASIBLE]" if merge_feas['overall_feasible'] else "[NOT FEASIBLE]"
    print(f"    Overall: {overall_status}")
    
    if merge_feas['use_cpu_recommended']:
        print("    Recommendation: Use CPU mode (--use-cpu flag)")
    
    print("  Direct Inference:")
    infer_feas = analysis['direct_inference_feasibility']
    overall_status = "[FEASIBLE]" if infer_feas['overall_feasible'] else "[NOT FEASIBLE]"
    print(f"    Overall: {overall_status}")
    
    if infer_feas['use_cpu_recommended']:
        print("    Recommendation: Use CPU mode (--use-cpu flag)")
    print()
    
    # Recommendations
    recommendations = report['recommendations']
    if recommendations:
        print("RECOMMENDATIONS:")
        for rec in recommendations:
            icon = {"critical": "[CRITICAL]", "warning": "[WARNING]", "info": "[INFO]"}.get(rec['type'], "[NOTE]")
            print(f"  {icon} {rec['message']}")
            print(f"    Action: {rec['action']}")
        print()
    
    # Stress test results
    if report.get('stress_test'):
        stress = report['stress_test']
        print("STRESS TEST RESULTS:")
        print(f"  Stability: {'[STABLE]' if stress['stable'] else '[UNSTABLE]'}")
        if stress['warnings']:
            print("  Warnings:")
            for warning in stress['warnings']:
                print(f"    - {warning}")
        print()
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Memory and System Analysis for LoRA Operations")
    
    # Analysis options
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Path to check for disk space analysis (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for analysis report (JSON format)"
    )
    parser.add_argument(
        "--format",
        choices=["console", "json"],
        default="console",
        help="Output format (default: console)"
    )
    
    # Testing options
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run memory stress test"
    )
    parser.add_argument(
        "--stress-duration",
        type=int,
        default=30,
        help="Duration of stress test in seconds (default: 30)"
    )
    
    # Monitoring options
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Start real-time memory monitoring"
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=5,
        help="Monitoring interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--monitor-output",
        type=str,
        help="Output CSV file for monitoring data"
    )
    
    # Quick checks
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Run quick system check without detailed analysis"
    )
    
    args = parser.parse_args()
    
    if args.monitor:
        # Real-time monitoring mode
        monitor = MemoryMonitor(args.monitor_interval)
        try:
            monitor.start_monitoring(args.monitor_output)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        return
    
    # System analysis mode
    analyzer = SystemAnalyzer()
    
    if args.quick_check:
        # Quick check mode
        analyzer.collect_memory_info()
        analyzer.collect_gpu_info()
        analyzer.collect_disk_info(args.path)
        
        mem_info = analyzer.memory_info
        gpu_info = analyzer.gpu_info
        disk_info = analyzer.disk_info
        
        print("QUICK SYSTEM CHECK")
        print("=" * 40)
        print(f"RAM Available: {mem_info['ram_available_gb']:.1f}GB")
        print(f"GPU Available: {'Yes' if gpu_info['cuda_available'] else 'No'}")
        if gpu_info['cuda_available'] and gpu_info['gpus']:
            best_gpu = max(gpu_info['gpus'], key=lambda g: g.get('total_memory_gb', 0))
            print(f"GPU Memory: {best_gpu.get('total_memory_gb', 0):.1f}GB")
        print(f"Disk Free: {disk_info.get('free_gb', 0):.1f}GB")
        print()
        
        # Simple recommendations
        if mem_info['ram_available_gb'] >= 8:
            print("[OK] Sufficient RAM for LoRA merge")
        else:
            print("[WARNING] Low RAM - consider direct inference")
        
        if gpu_info['cuda_available'] and gpu_info['gpus']:
            best_gpu = max(gpu_info['gpus'], key=lambda g: g.get('total_memory_gb', 0))
            if best_gpu.get('total_memory_gb', 0) >= 6:
                print("[OK] Sufficient GPU memory for operations")
            else:
                print("[WARNING] Low GPU memory - consider CPU mode")
        else:
            print("[INFO] Use CPU mode for all operations")
        
        return
    
    # Full analysis
    try:
        report = analyzer.run_full_analysis(args.path, args.stress_test)
        
        # Output report
        if args.format == "json" or args.output:
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Report saved to: {args.output}")
            else:
                print_report(report, "json")
        else:
            print_report(report, "console")
        
    except Exception as e:
        logger.error(f"Error during system analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()