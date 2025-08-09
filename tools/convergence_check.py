#!/usr/bin/env python3
"""
Neural network parameter convergence checking script
For analyzing parameter convergence during FermiNet training

Usage Examples:
    python convergence_check.py \
        --train_dir ~/experiments/h2_molecule/train \
        --output_dir ./convergence_analysis \
        --threshold 1e-5 \
        --window 10 \
        --sample_rate 2

Expected Output:
    - convergence_analysis/parameter_convergence.png: Parameter relative change plots
    - convergence_analysis/parameter_norms.png: Parameter norm change plots  
    - convergence_analysis/convergence_results.txt: Detailed analysis results text file
    - Terminal output convergence status summary
"""

import numpy as np
import jax.numpy as jnp
from ferminet import networks
import matplotlib.pyplot as plt
import os
import glob
import re
from typing import List, Dict, Tuple
import argparse


def extract_step_number(filename: str) -> int:
    """Extract step number from filename"""
    match = re.search(r'ckpt_(\d+)', filename)
    return int(match.group(1)) if match else 0


def load_checkpoint(filename: str) -> Tuple[int, dict]:
    """Load checkpoint file and return step number and parameters"""
    try:
        with open(filename, 'rb') as f:
            ckpt_data = np.load(f, allow_pickle=True)
            step = ckpt_data['t'].tolist() + 1
            params = ckpt_data['params'].tolist()
            return step, params
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None


def extract_layer_parameters(params: dict) -> Dict[str, np.ndarray]:
    """Extract weight and bias parameters from layers.streams"""
    layer_params = {}
    
    if 'layers' in params and 'streams' in params['layers']:
        for i, stream in enumerate(params['layers']['streams']):
            # Process double layer parameters
            if 'double' in stream:
                double_dict = stream['double']
                if 'w' in double_dict:
                    layer_params[f'stream_{i}_double_w'] = np.array(double_dict['w'])
                if 'b' in double_dict:
                    layer_params[f'stream_{i}_double_b'] = np.array(double_dict['b'])
            
            # Process single layer parameters
            if 'single' in stream:
                single_dict = stream['single']
                if 'w' in single_dict:
                    layer_params[f'stream_{i}_single_w'] = np.array(single_dict['w'])
                if 'b' in single_dict:
                    layer_params[f'stream_{i}_single_b'] = np.array(single_dict['b'])
    
    return layer_params


def calculate_parameter_statistics(param_history: Dict[str, List[np.ndarray]]) -> Dict[str, Dict]:
    """Calculate parameter statistics"""
    stats = {}
    
    for param_name, param_values in param_history.items():
        if len(param_values) < 2:
            continue
            
        # Calculate parameter change standard deviation
        param_changes = []
        param_norms = []
        
        for i in range(1, len(param_values)):
            diff = param_values[i] - param_values[i-1]
            param_changes.append(np.linalg.norm(diff.flatten()))
            param_norms.append(np.linalg.norm(param_values[i].flatten()))
        
        # Calculate relative changes
        relative_changes = []
        for i in range(len(param_changes)):
            if param_norms[i] > 1e-10:
                relative_changes.append(param_changes[i] / param_norms[i])
            else:
                relative_changes.append(0.0)
        
        # Calculate recent average change (last 20% of steps or at least last 10 steps)
        recent_window = max(10, len(relative_changes) // 5)
        recent_changes = relative_changes[-recent_window:] if len(relative_changes) >= recent_window else relative_changes
        
        stats[param_name] = {
            'absolute_changes': np.array(param_changes),
            'relative_changes': np.array(relative_changes),
            'param_norms': np.array(param_norms),
            'recent_mean_change': np.mean(recent_changes),
            'final_relative_change': relative_changes[-1] if relative_changes else 0.0
        }
    
    return stats


def assess_convergence(stats: Dict[str, Dict], convergence_threshold: float = 1e-5, 
                      stability_window: int = 10) -> Dict[str, bool]:
    """Assess whether parameters have converged"""
    convergence_status = {}
    
    for param_name, param_stats in stats.items():
        relative_changes = param_stats['relative_changes']
        
        if len(relative_changes) < stability_window:
            convergence_status[param_name] = False
            continue
        
        # Check if recent relative changes are all below threshold
        recent_changes = relative_changes[-stability_window:]
        is_converged = np.all(recent_changes < convergence_threshold)
        
        convergence_status[param_name] = is_converged
    
    return convergence_status


def plot_convergence(steps: List[int], stats: Dict[str, Dict], output_dir: str):
    """Plot convergence charts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group parameters by stream
    stream_groups = {}
    for param_name in stats.keys():
        # Extract stream number from parameter name (e.g., "stream_0_double_w" -> "stream_0")
        if param_name.startswith('stream_'):
            stream_id = '_'.join(param_name.split('_')[:2])  # "stream_0"
            if stream_id not in stream_groups:
                stream_groups[stream_id] = []
            stream_groups[stream_id].append(param_name)
    
    # Plot relative changes grouped by stream
    num_streams = len(stream_groups)
    cols = 2
    rows = (num_streams + cols - 1) // cols  # Calculate rows needed
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (stream_id, param_names) in enumerate(stream_groups.items()):
        plt.subplot(rows, cols, i + 1)
        
        for param_name in param_names:
            param_stats = stats[param_name]
            if len(param_stats['relative_changes']) == 0:
                continue
                
            plot_steps = steps[1:len(param_stats['relative_changes'])+1]
            # Create cleaner labels (remove "stream_X_" prefix)
            label = param_name.replace(f'{stream_id}_', '')
            plt.semilogy(plot_steps, param_stats['relative_changes'], alpha=0.7, label=label)
        
        plt.title(f'{stream_id} - Relative Changes')
        plt.xlabel('Training Steps')
        plt.ylabel('Relative Change (log scale)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add convergence threshold line
        plt.axhline(y=1e-5, color='r', linestyle='--', alpha=0.5, label='Threshold' if i == 0 else "")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot parameter norm changes
    plt.figure(figsize=(15, 8))
    for i, (param_name, param_stats) in enumerate(stats.items()):
        if len(param_stats['param_norms']) == 0:
            continue
            
        plot_steps = steps[1:len(param_stats['param_norms'])+1]
        plt.semilogy(plot_steps, param_stats['param_norms'], label=param_name, alpha=0.7)
    
    plt.title('Parameter Norm Changes')
    plt.xlabel('Training Steps')
    plt.ylabel('Parameter Norm (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'parameter_norms.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Check FermiNet parameter convergence')
    parser.add_argument('--train_dir', type=str, default='./',
                        help='Directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, default='./convergence_analysis',
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=1e-5,
                        help='Convergence threshold')
    parser.add_argument('--window', type=int, default=10,
                        help='Stability window size')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Sample rate (take every N checkpoints)')
    
    args = parser.parse_args()
    
    # Expand path
    train_dir = os.path.expanduser(args.train_dir)
    
    # Get all checkpoint files
    ckpt_files = glob.glob(os.path.join(train_dir, 'qmcjax_ckpt_*.npz'))
    ckpt_files.sort(key=extract_step_number)
    
    # Filter files by sample rate
    ckpt_files = ckpt_files[::args.sample_rate]
    
    print(f"Found {len(ckpt_files)} checkpoint files")
    print(f"Analyzing convergence, threshold: {args.threshold}, stability window: {args.window}")
    
    # Collect parameter history
    param_history = {}
    steps = []
    
    for i, ckpt_file in enumerate(ckpt_files):
        print(f"Processing file {i+1}/{len(ckpt_files)}: {os.path.basename(ckpt_file)}")
        
        step, params = load_checkpoint(ckpt_file)
        if params is None:
            continue
            
        steps.append(step)
        layer_params = extract_layer_parameters(params)
        
        # Initialize parameter history
        if not param_history:
            for param_name in layer_params.keys():
                param_history[param_name] = []
        
        # Add current parameters to history
        for param_name, param_value in layer_params.items():
            if param_name in param_history:
                param_history[param_name].append(param_value)
    
    if not param_history:
        print("Error: No valid parameter data found")
        return
    
    # Calculate statistics
    print("Calculating parameter statistics...")
    stats = calculate_parameter_statistics(param_history)
    
    # Assess convergence
    print("Assessing convergence...")
    convergence_status = assess_convergence(stats, args.threshold, args.window)
    
    # Output results
    print("\n" + "="*60)
    print("Convergence Analysis Results")
    print("="*60)
    
    converged_count = 0
    total_count = len(convergence_status)
    
    for param_name, is_converged in convergence_status.items():
        status = "✓ Converged" if is_converged else "✗ Not converged"
        recent_mean_change = stats[param_name]['recent_mean_change']
        final_change = stats[param_name]['final_relative_change']
        
        print(f"{param_name:30} {status:15} (recent avg: {recent_mean_change:.2e}, final change: {final_change:.2e})")
        
        if is_converged:
            converged_count += 1
    
    print("\n" + "-"*60)
    print(f"Overall convergence status: {converged_count}/{total_count} parameters converged")
    
    convergence_ratio = converged_count / total_count if total_count > 0 else 0
    if convergence_ratio >= 0.8:
        print("✓ Model basically converged (≥80% parameters converged)")
    elif convergence_ratio >= 0.5:
        print("⚠ Model partially converged (50-80% parameters converged)")
    else:
        print("✗ Model not converged (<50% parameters converged)")
    
    # Generate charts
    print(f"\nGenerating convergence charts to {args.output_dir}...")
    plot_convergence(steps, stats, args.output_dir)
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, 'convergence_results.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("FermiNet Parameter Convergence Analysis Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis Configuration:\n")
        f.write(f"- Training directory: {train_dir}\n")
        f.write(f"- Convergence threshold: {args.threshold}\n")
        f.write(f"- Stability window: {args.window}\n")
        f.write(f"- Analyzed files: {len(ckpt_files)}\n")
        f.write(f"- Step range: {min(steps)} - {max(steps)}\n\n")
        
        f.write("Parameter convergence status:\n")
        f.write("-"*50 + "\n")
        for param_name, is_converged in convergence_status.items():
            status = "Converged" if is_converged else "Not converged"
            recent_mean_change = stats[param_name]['recent_mean_change']
            final_change = stats[param_name]['final_relative_change']
            f.write(f"{param_name}: {status} (recent avg: {recent_mean_change:.2e}, final change: {final_change:.2e})\n")
        
        f.write(f"\nOverall convergence rate: {convergence_ratio:.1%}\n")
    
    print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
