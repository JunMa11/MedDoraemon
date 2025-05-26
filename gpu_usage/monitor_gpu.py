#!/usr/bin/env python3
"""
GPU monitoring script for fine-tuning diagnostics.

This script provides utilities to monitor GPU utilization, memory usage, temperature, and power draw. It is intended for use during model training or other GPU-intensive tasks through logging to help diagnose performance bottlenecks and resource issues.

Functions:
    get_gpu_stats(): Get GPU utilization and memory stats using nvidia-smi.
    monitor_gpu(interval=2, duration=None, log_file=None): Monitor GPU usage continuously and optionally log to a file.

Example usage:
    python monitor_gpu.py --interval 2 --duration 60 --log gpu_log.csv
"""
import subprocess
import time
import datetime
import argparse
import os

def get_gpu_stats():
    """
    Get GPU utilization and memory stats using nvidia-smi.

    Returns:
        list[dict]: A list of dictionaries, each containing the following keys for each GPU:
            - 'index' (int): GPU index.
            - 'name' (str): GPU name.
            - 'gpu_util' (float): GPU utilization percentage.
            - 'mem_used' (float): Used memory in MB.
            - 'mem_total' (float): Total memory in MB.
            - 'temp' (float): GPU temperature in Celsius.
            - 'power' (float): Power draw in Watts.
    Raises:
        Exception: If nvidia-smi is not available or fails to execute.
    """
    try:
        # Run nvidia-smi command
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            stats = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 7:
                    stats.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'gpu_util': float(parts[2]),
                        'mem_used': float(parts[3]),
                        'mem_total': float(parts[4]),
                        'temp': float(parts[5]),
                        'power': float(parts[6])
                    })
            return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    return []

def monitor_gpu(interval=2, duration=None, log_file=None):
    """
    Monitor GPU usage continuously and display stats in the console. Optionally log stats to a file.

    Args:
        interval (float, optional): Update interval in seconds. Defaults to 2.
        duration (int, optional): Monitor duration in seconds. If None, runs indefinitely. Defaults to None.
        log_file (str, optional): Path to a log file to save GPU stats as CSV. Defaults to None.

    Returns:
        None

    Raises:
        KeyboardInterrupt: If monitoring is stopped by the user.
    """
    start_time = time.time()
    
    print("Starting GPU monitoring...")
    print("Press Ctrl+C to stop")
    print("-" * 100)
    
    # Open log file if specified
    f = None
    if log_file:
        f = open(log_file, 'w')
        f.write("timestamp,gpu_index,gpu_name,gpu_util_%,mem_used_mb,mem_total_mb,temp_c,power_w\n")
    
    try:
        while True:
            current_time = time.time()
            if duration and (current_time - start_time) > duration:
                break
                
            stats = get_gpu_stats()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Clear screen for better display
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"GPU Monitoring - {timestamp}")
            print("-" * 100)
            
            for gpu in stats:
                if gpu['index'] == 0:  # Only monitor GPU 0
                    mem_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
                    
                    # Console output
                    print(f"GPU {gpu['index']}: {gpu['name']}")
                    print(f"  Utilization: {gpu['gpu_util']:5.1f}% {'█' * int(gpu['gpu_util']/2)}")
                    print(f"  Memory:      {gpu['mem_used']:6.0f}/{gpu['mem_total']:.0f} MB ({mem_percent:5.1f}%) {'█' * int(mem_percent/2)}")
                    print(f"  Temperature: {gpu['temp']:5.1f}°C")
                    print(f"  Power:       {gpu['power']:5.1f}W")
                    
                    # Performance indicators
                    if gpu['gpu_util'] < 50:
                        print("\n  ⚠️  WARNING: Low GPU utilization! Check data loading or batch size.")
                    elif gpu['gpu_util'] > 90:
                        print("\n  ✅ Good GPU utilization!")
                    
                    if mem_percent > 90:
                        print("  ⚠️  WARNING: High memory usage! Risk of OOM.")
                    
                    # Log to file if specified
                    if f:
                        f.write(f"{timestamp},{gpu['index']},{gpu['name']},{gpu['gpu_util']},{gpu['mem_used']},{gpu['mem_total']},{gpu['temp']},{gpu['power']}\n")
                        f.flush()
            
            print("\n" + "-" * 100)
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        if f:
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU usage during training")
    parser.add_argument("--interval", type=float, default=2, help="Update interval in seconds")
    parser.add_argument("--duration", type=int, help="Monitor duration in seconds (default: infinite)")
    parser.add_argument("--log", type=str, help="Log file path to save GPU stats")
    
    args = parser.parse_args()
    
    monitor_gpu(interval=args.interval, duration=args.duration, log_file=args.log) 