#!/usr/bin/env python3
"""
run.py - Simple multi-process training launcher
"""

import argparse
import os
import subprocess
import sys
import time


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-process training launcher",
        epilog="Example: python run.py -n 4 -d 8 multi_cpu.py"
    )
    
    parser.add_argument(
        "-n", "--ntask",
        type=int,
        default=2,
        help="Number of tasks (default: 2)"
    )
    
    parser.add_argument(
        "-d", "--ndevice", 
        type=int,
        default=2,
        help="Number of devices per host (default: 2)"
    )
    
    parser.add_argument(
        "script",
        help="Python script to run"
    )
    
    return parser.parse_args()


def launch_processes(ntask, ndevice, script):
    """Launch multiple processes for distributed training."""
    # Set environment variables
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    
    print(f"Starting {ntask} processes for distributed training...")
    print(f"Script: {script}")
    print(f"Tasks: {ntask}, Devices per host: {ndevice}")
    print()
    
    processes = []
    
    for i in range(ntask):
        print(f"Starting process {i} (JAX_PROCESS_ID={i})...")
        
        # Prepare environment for this process
        env = os.environ.copy()
        env["JAX_PROCESS_ID"] = str(i)
        
        # Start the process
        process = subprocess.Popen([
            sys.executable, script,
            f"--ntask={ntask}",
            f"--ndevice={ndevice}"
        ], env=env)
        
        processes.append(process)
        print(f"  Process {i} started with PID: {process.pid}")
        time.sleep(0.1)  # Small delay to avoid race conditions
    
    print(f"\nAll {ntask} processes started!")
    print(f"PIDs: {[p.pid for p in processes]}")
    print()
    
    return processes


def wait_for_processes(processes):
    """Wait for all processes to complete."""
    try:
        for process in processes:
            process.wait()
        print("All processes completed successfully!")
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating processes...")
        for process in processes:
            process.terminate()
        # Wait for termination
        for process in processes:
            process.wait()
        print("All processes terminated.")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Check if script exists
    if not os.path.isfile(args.script):
        print(f"Error: Python script '{args.script}' not found")
        sys.exit(1)
    
    # Launch processes
    processes = launch_processes(args.ntask, args.ndevice, args.script)
    
    # Wait for completion
    wait_for_processes(processes)


if __name__ == "__main__":
    main()