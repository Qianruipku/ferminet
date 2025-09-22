#!/usr/bin/env python3
"""
Main program to compute APMD 1D from input file.

Usage:
    python doppler.py <input_file> [crystal_direction] [ecut] [dq]

Arguments:
    input_file: Text file with 5 columns (grid_points x,y,z, pw_g, density)
    crystal_direction: Crystal direction ('100', '110', or '111') (default: '100')
    ecut: Energy cutoff in Hartree (default: 10.0)
    dq: Momentum step size (default: 0.1)
"""

import sys
import os
import numpy as np
import jax.numpy as jnp
from ferminet.observable.apmd import write_apmd_1d


def main():
    """Main function to process APMD data."""
    # Check command line arguments
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage: python doppler.py <input_file> [crystal_direction] [ecut] [dq]")
        print("Example: python doppler.py apmd_final.txt")
        print("Example: python doppler.py apmd_final.txt 110")
        print("Example: python doppler.py apmd_final.txt 110 20.0 0.2")
        sys.exit(1)
    
    # Parse arguments
    input_file = sys.argv[1]
    
    # Set default values
    crystal_direction = "100"  # Default crystal direction
    ecut = 10.0                # Default energy cutoff
    dq = 0.1                   # Default momentum step
    
    # Override with command line arguments if provided
    if len(sys.argv) >= 3:
        crystal_direction = sys.argv[2]
    
    if len(sys.argv) >= 4:
        try:
            ecut = float(sys.argv[3])
        except ValueError:
            print("Error: ecut must be a numeric value")
            sys.exit(1)
    
    if len(sys.argv) >= 5:
        try:
            dq = float(sys.argv[4])
        except ValueError:
            print("Error: dq must be a numeric value")
            sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Read data from file
    try:
        print(f"Reading data from {input_file}...")
        data = np.loadtxt(input_file)
        print(f"Loaded data shape: {data.shape}")
        
        if data.shape[1] != 5:
            print(f"Error: Expected 5 columns, but got {data.shape[1]} columns")
            sys.exit(1)
        
        # Extract grid points (first 3 columns), pw_g (4th column) and density (5th column)
        grid_points = data[:, :3]
        pw_g = data[:, 3]
        density = data[:, 4]
        
        # Process data: average density for equal magnitudes
        print("Processing data: averaging density for equal magnitudes...")
        unique_magnitudes, indices = np.unique(pw_g, return_inverse=True)
        avg_density = np.zeros_like(unique_magnitudes)
        
        for i in range(len(unique_magnitudes)):
            mask = indices == i
            avg_density[i] = np.mean(density[mask])
        
        # Save averaged data to new file
        avg_filename = "avg_" + input_file
        avg_data = np.column_stack((unique_magnitudes, avg_density))
        
        try:
            np.savetxt(avg_filename, avg_data, fmt='%.6f')
            print(f"Averaged data saved to: {avg_filename}")
            print(f"Original data points: {len(pw_g)}")
            print(f"Unique magnitudes: {len(unique_magnitudes)}")
        except Exception as e:
            print(f"Error saving averaged data: {e}")
        
        # Convert to JAX arrays
        grid_points_jax = jnp.array(grid_points)
        density_jax = jnp.array(density)
        
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Set output directory to current directory
    ckpt_save_path = "."
    
    # Call write_apmd_1d function
    try:
        print(f"Computing APMD 1D with parameters:")
        print(f"  Crystal direction: {crystal_direction}")
        print(f"  Energy cutoff (ecut): {ecut} Ha")
        print(f"  Momentum step (dq): {dq}")
        
        write_apmd_1d(
            crystal_direction=crystal_direction,
            ecut=ecut,
            dq=dq,
            grid_points=grid_points_jax,
            density=density_jax,
            ckpt_save_path=ckpt_save_path
        )
        
        output_filename = f'apmd_1d_{crystal_direction}.txt'
        print(f"Successfully generated: {output_filename}")
        
    except Exception as e:
        print(f"Error computing APMD: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()