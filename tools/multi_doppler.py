#!/usr/bin/env python3
"""
Main program to compute APMD 1D from multiple input files with t-distribution confidence interval.

Usage:
    python multi_doppler.py <base_name> <n_files> [crystal_direction] [ecut] [dq]

Arguments:
    base_name: Base name of input files (e.g., 'apmd' for apmd_1.npy, apmd_2.npy, ...)
    n_files: Number of files to process (e.g., 3 for apmd_1.npy, apmd_2.npy, apmd_3.npy)
             - apmd_n.npy contains cumulative average of first n groups
             - Individual group n data is derived from apmd_n.npy and apmd_{n-1}.npy
    crystal_direction: Crystal direction ('100', '110', or '111') (default: '100')
    ecut: Energy cutoff in Hartree (default: 10.0)
    dq: Momentum step size (default: 0.1)

Output:
    apmd_1d_<crystal_direction>.txt with 99% confidence interval using t-distribution
"""

import sys
import os
import glob
import numpy as np
import jax.numpy as jnp
from scipy import stats
from ferminet.observable.apmd import write_apmd_1d


def extract_individual_group_data(apmd_n, apmd_n_minus_1, n):
    """
    Extract individual group n data from cumulative averages.
    
    apmd_n contains average of first n groups
    apmd_n_minus_1 contains average of first (n-1) groups
    
    Individual group n data = n * apmd_n - (n-1) * apmd_n_minus_1
    """
    if n == 1:
        return apmd_n
    else:
        return n * apmd_n - (n - 1) * apmd_n_minus_1


def compute_apmd_with_std(crystal_direction, ecut, dq, all_grid_points, all_densities, ckpt_save_path):
    """
    Compute APMD 1D for each individual group and calculate 99% confidence interval using t-distribution.
    """
    n_groups = len(all_densities)
    apmd_results = []
    
    # Compute APMD for each individual group
    for i in range(n_groups):
        print(f"Computing APMD for group {i+1}...")
        
        # Create a temporary directory for this group
        temp_path = f"{ckpt_save_path}/temp_group_{i+1}"
        os.makedirs(temp_path, exist_ok=True)
        
        # Compute APMD for this group
        write_apmd_1d(
            crystal_direction=crystal_direction,
            ecut=ecut,
            dq=dq,
            grid_points=all_grid_points[i],
            density=all_densities[i],
            ckpt_save_path=temp_path
        )
        
        # Read the generated APMD file
        apmd_file = f"{temp_path}/apmd_1d_{crystal_direction}.txt"
        if os.path.exists(apmd_file):
            apmd_data = np.loadtxt(apmd_file)
            apmd_results.append(apmd_data)
            
            # Clean up temporary file
            os.remove(apmd_file)
            os.rmdir(temp_path)
    
    # Convert to numpy array for easier manipulation
    apmd_array = np.array(apmd_results)  # Shape: (n_groups, n_points, 2)
    
    # Calculate mean and standard error with t-distribution correction (99% confidence)
    q_values = apmd_array[0, :, 0]  # q values are the same for all groups
    apmd_values = apmd_array[:, :, 1]  # APMD values for all groups
    
    mean_apmd = np.mean(apmd_values, axis=0)
    std_apmd = np.std(apmd_values, axis=0, ddof=1)  # Use sample standard deviation
    std_error = std_apmd / np.sqrt(n_groups)  # Standard error of the mean
    
    # Apply t-distribution correction for 99% confidence interval
    confidence_level = 0.99
    alpha = 1 - confidence_level
    degrees_of_freedom = n_groups - 1
    t_critical = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
    confidence_interval = t_critical * std_error
    
    # Write results with t-distribution corrected confidence interval
    output_filename = f"{ckpt_save_path}/apmd_1d_{crystal_direction}.txt"
    with open(output_filename, 'w') as f:
        f.write(f"# q (a.u.^-1)    APMD (a.u.)    CI_99% (a.u.)    [n={n_groups}, t={t_critical:.3f}]\n")
        for i in range(len(q_values)):
            f.write(f"{q_values[i]:.6f}    {mean_apmd[i]:.6f}    {confidence_interval[i]:.6f}\n")
    
    return output_filename


def main():
    """Main function to process multiple APMD data files."""
    # Check command line arguments
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python multi_doppler.py <base_name> <n_files> [crystal_direction] [ecut] [dq]")
        print("Example: python multi_doppler.py apmd 3")
        print("Example: python multi_doppler.py apmd 3 110")
        print("Example: python multi_doppler.py apmd 3 110 20.0 0.2")
        sys.exit(1)
    
    # Parse arguments
    base_name = sys.argv[1]
    
    try:
        n_files = int(sys.argv[2])
    except ValueError:
        print("Error: n_files must be an integer")
        sys.exit(1)
    
    if n_files < 1:
        print("Error: n_files must be at least 1")
        sys.exit(1)
    
    # Set default values
    crystal_direction = "100"  # Default crystal direction
    ecut = 10.0                # Default energy cutoff
    dq = 0.1                   # Default momentum step
    
    # Override with command line arguments if provided
    if len(sys.argv) >= 4:
        crystal_direction = sys.argv[3]
    
    if len(sys.argv) >= 5:
        try:
            ecut = float(sys.argv[4])
        except ValueError:
            print("Error: ecut must be a numeric value")
            sys.exit(1)
    
    if len(sys.argv) >= 6:
        try:
            dq = float(sys.argv[5])
        except ValueError:
            print("Error: dq must be a numeric value")
            sys.exit(1)
    
    # Generate file names and check existence
    input_files = []
    for i in range(1, n_files + 1):
        filename = f"{base_name}_{i}.npy"
        if not os.path.exists(filename):
            print(f"Error: Input file '{filename}' not found")
            sys.exit(1)
        input_files.append(filename)
    
    print(f"Processing {n_files} files: {input_files}")
    
    # Read and process data from all files
    try:
        all_cumulative_data = []
        all_grid_points = []
        all_densities = []
        
        # Load all cumulative average files
        for i, input_file in enumerate(input_files):
            print(f"Reading data from {input_file}...")
            
            # Load npy file with shape (ntwist, n_planewave, 5)
            data = np.load(input_file)
            print(f"Loaded {input_file} data shape: {data.shape}")
            
            if len(data.shape) != 3 or data.shape[2] != 5:
                print(f"Error: Expected shape (ntwist, n_planewave, 5), but got {data.shape}")
                sys.exit(1)
            
            all_cumulative_data.append(data)
        
        # Extract individual group data
        print("Extracting individual group data...")
        for i in range(n_files):
            current_cumulative = all_cumulative_data[i]
            
            if i == 0:
                # First group is the same as first cumulative
                individual_data = current_cumulative
            else:
                # Extract individual group data using the formula
                prev_cumulative = all_cumulative_data[i-1]
                n = i + 1  # Group number (1-indexed)
                
                # Apply the formula: individual = n * cumulative_n - (n-1) * cumulative_{n-1}
                individual_data = n * current_cumulative - (n - 1) * prev_cumulative
            
            # Extract grid points and density from individual data
            grid_points = individual_data[:, :, :3]  # (ntwist, n_planewave, 3)
            density = individual_data[:, :, 4]       # (ntwist, n_planewave)
            
            # Convert to JAX arrays
            grid_points_jax = jnp.array(grid_points)
            density_jax = jnp.array(density)
            
            all_grid_points.append(grid_points_jax)
            all_densities.append(density_jax)
            
            ntwist, n_planewave, _ = individual_data.shape
            print(f"Group {i+1}: Number of twists: {ntwist}, Number of planewaves: {n_planewave}")
        
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)
    
    # Set output directory to current directory
    ckpt_save_path = "."
    
    # Compute APMD with standard deviation
    try:
        print(f"Computing APMD 1D with parameters:")
        print(f"  Crystal direction: {crystal_direction}")
        print(f"  Energy cutoff (ecut): {ecut} Ha")
        print(f"  Momentum step (dq): {dq}")
        print(f"  Number of groups: {n_files}")
        
        output_filename = compute_apmd_with_std(
            crystal_direction=crystal_direction,
            ecut=ecut,
            dq=dq,
            all_grid_points=all_grid_points,
            all_densities=all_densities,
            ckpt_save_path=ckpt_save_path
        )
        
        print(f"Successfully generated: {output_filename}")
        print(f"Output includes mean APMD and 99% confidence interval (t-distribution) for {n_files} groups.")
        
    except Exception as e:
        print(f"Error computing APMD: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()