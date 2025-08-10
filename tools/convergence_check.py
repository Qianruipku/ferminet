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


def calculate_parameter_stability_metrics(param_history: Dict[str, List[np.ndarray]]) -> Dict[str, Dict]:
    """Calculate parameter stability metrics - detect rotational or oscillatory changes"""
    stability_metrics = {}
    
    for param_name, param_values in param_history.items():
        if len(param_values) < 10:  # Need enough data points
            continue
        
        # Use recent points for analysis
        recent_params = param_values[-min(20, len(param_values)):]  # Reduced window size
        
        try:
            # Calculate change magnitudes (norm-based analysis)
            change_magnitudes = []
            relative_changes = []
            
            for i in range(1, len(recent_params)):
                diff = recent_params[i] - recent_params[i-1]
                change_mag = np.linalg.norm(diff.flatten())
                param_norm = np.linalg.norm(recent_params[i].flatten())
                
                change_magnitudes.append(change_mag)
                if param_norm > 1e-10:
                    relative_changes.append(change_mag / param_norm)
                else:
                    relative_changes.append(0.0)
            
            change_magnitudes = np.array(change_magnitudes)
            relative_changes = np.array(relative_changes)
            
            # Autocorrelation analysis for oscillatory behavior
            if len(change_magnitudes) >= 4:
                n = len(change_magnitudes)
                if n > 3:
                    autocorr_1 = np.corrcoef(change_magnitudes[:-1], change_magnitudes[1:])[0, 1]
                    if n > 4:
                        autocorr_2 = np.corrcoef(change_magnitudes[:-2], change_magnitudes[2:])[0, 1]
                    else:
                        autocorr_2 = 0
                else:
                    autocorr_1 = autocorr_2 = 0
            else:
                autocorr_1 = autocorr_2 = 0
            
            # Trend analysis instead of full covariance
            # Check if changes are getting smaller (converging) or stable
            if len(change_magnitudes) >= 6:
                first_half_mean = np.mean(change_magnitudes[:len(change_magnitudes)//2])
                second_half_mean = np.mean(change_magnitudes[len(change_magnitudes)//2:])
                trend_ratio = second_half_mean / (first_half_mean + 1e-10)
            else:
                trend_ratio = 1.0
            
            # Stability of change magnitude
            change_magnitude_std = np.std(change_magnitudes)
            change_magnitude_mean = np.mean(change_magnitudes)
            change_stability = change_magnitude_std / (change_magnitude_mean + 1e-10)
            
            # Simplified directional consistency using smaller samples
            if len(recent_params) >= 4:
                # Sample a few parameter subsets to check directional consistency
                param_size = recent_params[0].size
                max_sample_size = min(1000, param_size)  # Limit sample size
                
                if param_size > max_sample_size:
                    # Random sampling for large parameters
                    np.random.seed(42)  # For reproducibility
                    sample_indices = np.random.choice(param_size, max_sample_size, replace=False)
                    sampled_params = [p.flatten()[sample_indices] for p in recent_params[-4:]]
                else:
                    sampled_params = [p.flatten() for p in recent_params[-4:]]
                
                # Calculate consecutive change directions
                diffs = [sampled_params[i+1] - sampled_params[i] for i in range(len(sampled_params)-1)]
                
                if len(diffs) >= 2:
                    normalized_diffs = [d / (np.linalg.norm(d) + 1e-10) for d in diffs]
                    cosine_similarities = []
                    for i in range(len(normalized_diffs) - 1):
                        cos_sim = np.dot(normalized_diffs[i], normalized_diffs[i+1])
                        cosine_similarities.append(cos_sim)
                    avg_directional_consistency = np.mean(cosine_similarities) if cosine_similarities else 0
                else:
                    avg_directional_consistency = 0
            else:
                avg_directional_consistency = 0
            
            # Simple variance concentration estimate
            recent_rel_changes = relative_changes[-min(10, len(relative_changes)):]
            if len(recent_rel_changes) > 0:
                change_variance = np.var(recent_rel_changes)
                mean_change = np.mean(recent_rel_changes)
                if mean_change > 1e-10:
                    normalized_variance = change_variance / (mean_change**2)
                else:
                    normalized_variance = 0
            else:
                normalized_variance = 0
            
            stability_metrics[param_name] = {
                'top3_variance_ratio': min(1.0, 1.0 - normalized_variance),  # Approximation
                'top1_variance_ratio': min(1.0, 1.0 - normalized_variance * 0.5),
                'autocorr_lag1': autocorr_1 if not np.isnan(autocorr_1) else 0,
                'autocorr_lag2': autocorr_2 if not np.isnan(autocorr_2) else 0,
                'directional_consistency': avg_directional_consistency,
                'change_stability': change_stability,
                'trend_ratio': trend_ratio,
                'mean_relative_change': np.mean(relative_changes),
            }
            
        except (np.linalg.LinAlgError, ValueError, MemoryError) as e:
            print(f"Warning: Could not calculate stability metrics for {param_name}: {e}")
            # Fallback metrics
            stability_metrics[param_name] = {
                'top3_variance_ratio': 0,
                'top1_variance_ratio': 0,
                'autocorr_lag1': 0,
                'autocorr_lag2': 0,
                'directional_consistency': 0,
                'change_stability': float('inf'),
                'trend_ratio': 1.0,
                'mean_relative_change': 0,
            }
    
    return stability_metrics


def assess_convergence_with_stability(stats: Dict[str, Dict], 
                                    stability_metrics: Dict[str, Dict],
                                    convergence_threshold: float = 1e-5, 
                                    stability_window: int = 10) -> Dict[str, str]:
    """Enhanced convergence assessment considering parameter stability"""
    convergence_status = {}
    
    for param_name, param_stats in stats.items():
        relative_changes = param_stats['relative_changes']
        
        if len(relative_changes) < stability_window:
            convergence_status[param_name] = 'insufficient_data'
            continue
        
        # Traditional convergence check
        recent_changes = relative_changes[-stability_window:]
        is_traditionally_converged = np.all(recent_changes < convergence_threshold)
        recent_mean = np.mean(recent_changes)
        
        # Stability analysis if available
        if param_name in stability_metrics:
            metrics = stability_metrics[param_name]
            
            # Criteria for different convergence states
            is_low_dimensional_change = metrics['top3_variance_ratio'] > 0.85  # Changes concentrated in few directions
            is_directionally_consistent = abs(metrics['directional_consistency']) > 0.3  # Consistent direction changes
            is_stable_magnitude = metrics['change_stability'] < 2.0  # Stable change magnitudes
            is_oscillatory = abs(metrics['autocorr_lag1']) > 0.4 or abs(metrics['autocorr_lag2']) > 0.4
            
            # Enhanced convergence classification
            if is_traditionally_converged:
                convergence_status[param_name] = 'converged'
            elif (recent_mean < convergence_threshold * 20 and 
                  is_low_dimensional_change and is_stable_magnitude):
                convergence_status[param_name] = 'functional_converged'  # Likely rotating but functionally stable
            elif (recent_mean < convergence_threshold * 50 and 
                  is_oscillatory and is_stable_magnitude):
                convergence_status[param_name] = 'stable_oscillation'  # Stable oscillatory pattern
            elif recent_mean < convergence_threshold * 10:
                convergence_status[param_name] = 'slowly_converging'  # Making progress but slowly
            else:
                convergence_status[param_name] = 'not_converged'
        else:
            # Fallback to traditional assessment
            if is_traditionally_converged:
                convergence_status[param_name] = 'converged'
            elif recent_mean < convergence_threshold * 10:
                convergence_status[param_name] = 'slowly_converging'
            else:
                convergence_status[param_name] = 'not_converged'
    
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
    
    # Calculate stability metrics
    print("Calculating stability metrics...")
    stability_metrics = calculate_parameter_stability_metrics(param_history)
    
    # Enhanced convergence assessment
    print("Assessing convergence with stability analysis...")
    convergence_status = assess_convergence_with_stability(stats, stability_metrics, args.threshold, args.window)
    
    # Output results
    print("\n" + "="*80)
    print("Enhanced Convergence Analysis Results")
    print("="*80)
    
    # Count different status types
    status_counts = {
        'converged': 0,
        'functional_converged': 0, 
        'stable_oscillation': 0,
        'slowly_converging': 0,
        'not_converged': 0,
        'insufficient_data': 0
    }
    
    # Status descriptions
    status_descriptions = {
        'converged': '✓ Fully Converged',
        'functional_converged': '◐ Functionally Converged (rotating/low-dim)',
        'stable_oscillation': '~ Stable Oscillation', 
        'slowly_converging': '⚠ Slowly Converging',
        'not_converged': '✗ Not Converged',
        'insufficient_data': '? Insufficient Data'
    }
    
    for param_name, status in convergence_status.items():
        if status in status_counts:
            status_counts[status] += 1
            
        recent_mean_change = stats[param_name]['recent_mean_change'] if param_name in stats else 0
        final_change = stats[param_name]['final_relative_change'] if param_name in stats else 0
        
        desc = status_descriptions.get(status, status)
        print(f"{param_name:35} {desc:35} (avg: {recent_mean_change:.2e}, final: {final_change:.2e})")
        
        # Show stability metrics for interesting cases
        if param_name in stability_metrics and status in ['functional_converged', 'stable_oscillation']:
            metrics = stability_metrics[param_name]
            print(f"{'':37} → Top3 variance: {metrics['top3_variance_ratio']:.3f}, "
                  f"Direction consistency: {metrics['directional_consistency']:.3f}, "
                  f"Change stability: {metrics['change_stability']:.3f}")
    
    print("\n" + "-"*80)
    total_analyzed = sum(status_counts.values())
    print(f"Convergence Summary:")
    for status, count in status_counts.items():
        if count > 0:
            percentage = count / total_analyzed * 100 if total_analyzed > 0 else 0
            print(f"  {status_descriptions[status]:35} {count:3d}/{total_analyzed} ({percentage:5.1f}%)")
    
    # Overall assessment
    effectively_converged = status_counts['converged'] + status_counts['functional_converged']
    convergence_ratio = effectively_converged / total_analyzed if total_analyzed > 0 else 0
    
    print(f"\nOverall Assessment:")
    if convergence_ratio >= 0.8:
        print("✓ Model has effectively converged! (≥80% parameters stable)")
    elif convergence_ratio >= 0.6:
        print("◐ Model shows good convergence (60-80% parameters stable)")
    elif convergence_ratio >= 0.4:
        print("⚠ Model shows partial convergence (40-60% parameters stable)")
    else:
        print("✗ Model has not converged sufficiently (<40% parameters stable)")
    
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
        for param_name, status in convergence_status.items():
            recent_mean_change = stats[param_name]['recent_mean_change'] if param_name in stats else 0
            final_change = stats[param_name]['final_relative_change'] if param_name in stats else 0
            f.write(f"{param_name}: {status} (recent avg: {recent_mean_change:.2e}, final change: {final_change:.2e})\n")
            
            # Add stability metrics for interesting cases
            if param_name in stability_metrics and status in ['functional_converged', 'stable_oscillation']:
                metrics = stability_metrics[param_name]
                f.write(f"  → Stability metrics: top3_var={metrics['top3_variance_ratio']:.3f}, "
                       f"dir_consistency={metrics['directional_consistency']:.3f}, "
                       f"change_stability={metrics['change_stability']:.3f}\n")
        
        # Calculate convergence ratio
        status_counts = {}
        for status in convergence_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        effectively_converged = status_counts.get('converged', 0) + status_counts.get('functional_converged', 0)
        total_count = len(convergence_status)
        convergence_ratio = effectively_converged / total_count if total_count > 0 else 0
        
        f.write(f"\nOverall effective convergence rate: {convergence_ratio:.1%}\n")
        f.write(f"Status breakdown:\n")
        for status, count in status_counts.items():
            f.write(f"  {status}: {count}\n")
    
    print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
