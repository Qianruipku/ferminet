#!/usr/bin/env python3
"""
Main program to compute APMD 1D from ABINIT DOPPLER binary file.

Usage:
    python doppler_abinit.py <doppler_file> [crystal_direction] [ecut] [dq] [merge_kpoints]

Arguments:
    doppler_file: ABINIT DOPPLER binary file (e.g., Sio_DS1_DOPPLER)
    crystal_direction: Crystal direction ('100', '110', or '111') (default: '100')
    ecut: Energy cutoff in Hartree (default: 10.0)
    dq: Momentum step size (default: 0.1)
    merge_kpoints: True/False - merge all k-points before calculation (default: False)
                   False: calculate each k-point separately then average (original method)
                   True: merge all k-points data and calculate once (new method)
"""

import sys
import os
import numpy as np
import jax.numpy as jnp
import struct
from ferminet.observable.apmd import cal_apmd_1d
from ferminet.utils.tetrahedron import delaunay_tetrahedralization


def read_doppler_file(filename):
    """
    Read ABINIT DOPPLER unformatted binary file
    
    File structure (Fortran unformatted):
    - Record 1: nfft, nkpt, ucvol, rprim(3,3)
    - Record 2-N: For each k-point: pcart(3,nfft), rho_moment(nfft)
    
    Returns:
    --------
    nfft : int
        Number of FFT points
    nkpt : int  
        Number of k-points
    ucvol : float
        Unit cell volume
    rprim : ndarray(3,3)
        Primitive lattice vectors
    pcart : ndarray(3,nfft,nkpt)
        Cartesian momentum for each FFT point and k-point
    rho_moment : ndarray(nfft,nkpt)
        Density moment for each FFT point and k-point
    """
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    
    with open(filename, 'rb') as f:
        # Read header record
        header_size = struct.unpack('<I', f.read(4))[0]
        header_data = f.read(header_size)
        header_size_end = struct.unpack('<I', f.read(4))[0]
        
        if header_size != header_size_end:
            raise ValueError("Corrupted header record")
        
        # Unpack header: 2 ints + 10 doubles
        header = struct.unpack('<2i10d', header_data)
        nfft, nkpt = header[0], header[1]
        ucvol = header[2]
        rprim = np.array(header[3:12]).reshape(3, 3)
        
        # Initialize arrays
        pcart = np.zeros((3, nfft, nkpt), dtype=np.float64)
        rho_moment = np.zeros((nfft, nkpt), dtype=np.float64)
        
        # Read k-point data
        for ikpt in range(nkpt):
            # Read k-point record
            kpt_size = struct.unpack('<I', f.read(4))[0]
            kpt_data = f.read(kpt_size)
            kpt_size_end = struct.unpack('<I', f.read(4))[0]
            
            if kpt_size != kpt_size_end:
                raise ValueError(f"Corrupted k-point {ikpt} record")
            
            # Unpack k-point data: 4*nfft doubles
            kpt_values = struct.unpack(f'<{4*nfft}d', kpt_data)
            
            # Split data: first 3*nfft are pcart, last nfft are rho_moment
            pcart_flat = np.array(kpt_values[:3*nfft]) * np.pi * 2
            rho_flat = np.array(kpt_values[3*nfft:])
            
            # Reshape pcart (Fortran column-major order)
            pcart[:, :, ikpt] = pcart_flat.reshape(3, nfft, order='F')
            rho_moment[:, ikpt] = rho_flat
    
    return nfft, nkpt, ucvol, rprim, pcart, rho_moment


def process_abinit_data(pcart, rho_moment, ucvol, ecut=10.0):
    """
    Process ABINIT DOPPLER data for each k-point separately
    Apply energy cutoff to filter plane waves outside the energy sphere
    
    Parameters:
    -----------
    pcart : ndarray(3,nfft,nkpt)
        Cartesian momentum
    rho_moment : ndarray(nfft,nkpt)
        Density moment
    ucvol : float
        Unit cell volume
    ecut : float
        Energy cutoff in Hartree (default: 10.0 Ha)
        
    Returns:
    --------
    grid_points_list : list of ndarray
        List of grid points for each k-point
    density_list : list of ndarray
        List of density values for each k-point
    """
    
    print(f"Processing ABINIT data for APMD calculation with ecut = {ecut:.2f} Ha...")
    
    # Process data for each k-point separately
    nfft, nkpt = rho_moment.shape
    
    grid_points_list = []
    density_list = []
    filtered_count = 0
    total_count = 0
    
    for ikpt in range(nkpt):
        # Calculate momentum magnitudes for this k-point
        pcart_kpt = pcart[:, :, ikpt]  # Shape: (3, nfft)
        pw_g_kpt = np.linalg.norm(pcart_kpt, axis=0)  # Shape: (nfft,)
        
        # Get density for this k-point
        rho_kpt = rho_moment[:, ikpt]  # Shape: (nfft,)
        
        # Apply energy cutoff: E = |p|^2 (in Hartree atomic units)
        energy_kpt = pw_g_kpt**2
        energy_mask = energy_kpt <= ecut
        
        # Filter based on energy cutoff
        pcart_filtered = pcart_kpt[:, energy_mask].T  # Shape: (n_filtered, 3)
        rho_filtered = rho_kpt[energy_mask]
        
        if len(pcart_filtered) > 0:
            grid_points_list.append(pcart_filtered)
            density_list.append(rho_filtered)
        
        # Update statistics
        filtered_count += np.sum(energy_mask)
        total_count += len(pw_g_kpt)
        
        if ikpt == 0:  # Print details for first k-point
            print(f"  k-point 1: {np.sum(energy_mask)}/{len(pw_g_kpt)} plane waves within ecut")
            print(f"  Energy range: [{energy_kpt.min():.3f}, {energy_kpt.max():.3f}] Ha")
            print(f"  Momentum range: [{pw_g_kpt.min():.6f}, {pw_g_kpt.max():.6f}]")
    
    # Print filtering statistics
    print(f"Energy cutoff filtering results:")
    print(f"  Original data points: {total_count}")
    print(f"  Points within ecut: {filtered_count}")
    print(f"  Filtering ratio: {filtered_count/total_count:.3f}")
    print(f"  Number of k-points with valid data: {len(grid_points_list)}")
    
    return grid_points_list, density_list


def print_results(nfft, nkpt, ucvol, rprim):
    """Print ABINIT file information"""
    print('=' * 50)
    print('ABINIT DOPPLER File Information:')
    print('=' * 50)
    print(f'nfft = {nfft}')
    print(f'nkpt = {nkpt}')
    print(f'ucvol = {ucvol:.6f}')
    print('rprim 矩阵:')
    for i in range(3):
        print(f'{rprim[i, 0]:12.6f}{rprim[i, 1]:12.6f}{rprim[i, 2]:12.6f}')
    print('=' * 50)


def save_data(filename_base, nfft, nkpt, ucvol, rprim, pcart, rho_moment):
    """Save data in numpy format"""
    try:
        np.savez_compressed(f'{filename_base}_abinit.npz',
                           nfft=nfft, nkpt=nkpt, ucvol=ucvol, rprim=rprim,
                           pcart=pcart, rho_moment=rho_moment)
        
        # Save metadata as text
        with open(f'{filename_base}_info.txt', 'w') as f:
            f.write(f"ABINIT DOPPLER File Analysis\n")
            f.write(f"============================\n")
            f.write(f"nfft = {nfft}\n")
            f.write(f"nkpt = {nkpt}\n")  
            f.write(f"ucvol = {ucvol:.6f}\n")
            f.write(f"rprim matrix:\n")
            for i in range(3):
                f.write(f"{rprim[i, 0]:12.6f}{rprim[i, 1]:12.6f}{rprim[i, 2]:12.6f}\n")
            f.write(f"\nArray shapes:\n")
            f.write(f"pcart: {pcart.shape}\n")
            f.write(f"rho_moment: {rho_moment.shape}\n")
            f.write(f"\nData ranges:\n")
            f.write(f"pcart: [{pcart.min():.3e}, {pcart.max():.3e}]\n")
            f.write(f"rho_moment: [{rho_moment.min():.3e}, {rho_moment.max():.3e}]\n")
        
        print(f"ABINIT data saved to {filename_base}_abinit.npz and {filename_base}_info.txt")
        
    except Exception as e:
        print(f"Warning: Could not save ABINIT data: {e}")


def merge_all_kpoints(grid_points_list, density_list):
    """
    Merge all k-points data into single arrays
    
    Parameters:
    -----------
    grid_points_list : list of ndarray
        List of grid points for each k-point
    density_list : list of ndarray
        List of density values for each k-point
        
    Returns:
    --------
    merged_grid_points : ndarray
        Combined grid points from all k-points
    merged_density : ndarray
        Combined density values from all k-points
    """
    
    if not grid_points_list:
        return np.array([]), np.array([])
    
    # Concatenate all grid points and densities
    merged_grid_points = np.vstack(grid_points_list)
    merged_density = np.concatenate(density_list)
    
    print(f"Merged data statistics:")
    print(f"  Total data points: {len(merged_density)}")
    print(f"  Grid points shape: {merged_grid_points.shape}")
    print(f"  Density range: [{merged_density.min():.3e}, {merged_density.max():.3e}]")
    
    return merged_grid_points, merged_density


def main():
    """Main function to process ABINIT DOPPLER data and compute APMD."""
    
    # Check command line arguments
    if len(sys.argv) < 2 or len(sys.argv) > 6:
        print("Usage: python doppler_abinit.py <doppler_file> [crystal_direction] [ecut] [dq] [merge_kpoints]")
        print("Example: python doppler_abinit.py Sio_DS1_DOPPLER")
        print("Example: python doppler_abinit.py Sio_DS1_DOPPLER 110")
        print("Example: python doppler_abinit.py Sio_DS1_DOPPLER 110 20.0 0.2")
        print("Example: python doppler_abinit.py Sio_DS1_DOPPLER 110 20.0 0.2 True")
        print("Arguments:")
        print("  doppler_file: ABINIT DOPPLER binary file")
        print("  crystal_direction: '100', '110', or '111' (default: '100')")
        print("  ecut: Energy cutoff in Hartree (default: 10.0)")
        print("  dq: Momentum step size (default: 0.1)")
        print("  merge_kpoints: True/False - merge all k-points before calculation (default: False)")
        print("Note: Energy cutoff (ecut) filters plane waves with E = |p|²/2 <= ecut")
        sys.exit(1)
    
    # Parse arguments
    doppler_file = sys.argv[1]
    
    # Set default values
    crystal_direction = "100"  # Default crystal direction
    ecut = 10.0                # Default energy cutoff
    dq = 0.1                   # Default momentum step
    merge_kpoints = True      # Default: combine all k-points before calculation
    
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
    
    if len(sys.argv) >= 6:
        merge_str = sys.argv[5].lower()
        if merge_str in ['true', '1', 'yes', 't']:
            merge_kpoints = True
        elif merge_str in ['false', '0', 'no', 'f']:
            merge_kpoints = False
        else:
            print("Error: merge_kpoints must be True/False")
            sys.exit(1)
    
    # Validate crystal direction
    valid_directions = ['100', '110', '111']
    if crystal_direction not in valid_directions:
        print(f"Error: crystal_direction must be one of {valid_directions}")
        sys.exit(1)
    
    # Check if doppler file exists
    if not os.path.exists(doppler_file):
        print(f"Error: DOPPLER file '{doppler_file}' not found")
        sys.exit(1)
    
    try:
        # Read ABINIT DOPPLER file
        print(f"Reading ABINIT DOPPLER file: {doppler_file}")
        nfft, nkpt, ucvol, rprim, pcart, rho_moment = read_doppler_file(doppler_file)
        
        # Print file information
        print_results(nfft, nkpt, ucvol, rprim)
        
        # Save ABINIT data
        base_name = os.path.splitext(doppler_file)[0]
        save_data(base_name, nfft, nkpt, ucvol, rprim, pcart, rho_moment)
        
        # Process data for APMD calculation (apply energy cutoff)
        grid_points_list, density_list = process_abinit_data(pcart, rho_moment, ucvol, ecut=ecut)
        
    except Exception as e:
        print(f"Error reading ABINIT file: {e}")
        sys.exit(1)
    
    # Compute APMD 1D
    try:
        print(f"\nComputing APMD 1D with parameters:")
        print(f"  Crystal direction: {crystal_direction}")
        print(f"  Energy cutoff (ecut): {ecut} Ha")
        print(f"  Momentum step (dq): {dq}")
        print(f"  Number of k-points: {len(grid_points_list)}")
        print(f"  Merge k-points: {merge_kpoints}")
        
        # Set up q-grid
        qmax = (jnp.sqrt(2.0 * ecut) + dq) // dq
        qz_full = jnp.arange(-qmax, qmax, 1) * dq
        
        if merge_kpoints:
            # Method 1: Merge all k-points data and compute APMD once
            print("\nUsing merged k-points calculation method...")
            
            # Merge all k-points data
            merged_grid_points, merged_density = merge_all_kpoints(grid_points_list, density_list)
            
            if len(merged_grid_points) == 0:
                print("Error: No valid data points found!")
                sys.exit(1)
            
            # Convert to JAX arrays
            grid_points_jax = jnp.array(merged_grid_points)
            density_jax = jnp.array(merged_density)
            
            # Compute tetrahedralization for merged data
            print("Computing Delaunay tetrahedralization for merged data...")
            tetrahedra, _ = delaunay_tetrahedralization(grid_points_jax)
            
            # Compute APMD for merged data
            print("Computing APMD for merged data...")
            apmd_1d_full = cal_apmd_1d(
                crystal_direction=crystal_direction,
                qz=qz_full,
                grid_points=grid_points_jax,
                density=density_jax,
                tetrahedra=tetrahedra
            )
            
            print(f"  Completed calculation with {len(merged_grid_points)} total data points")
            
        else:
            # Method 2: Calculate k-points separately and average (original method)
            print("\nUsing separate k-points calculation method...")
            
            # Initialize accumulator for APMD results
            apmd_1d_sum = jnp.zeros_like(qz_full)
            valid_kpts = 0
            
            # Loop over each k-point
            for ikpt, (grid_points_kpt, density_kpt) in enumerate(zip(grid_points_list, density_list)):
                if len(grid_points_kpt) == 0:
                    continue
                    
                # Convert to JAX arrays
                grid_points_jax = jnp.array(grid_points_kpt)
                density_jax = jnp.array(density_kpt)
                
                # Compute tetrahedralization for this k-point
                tetrahedra, _ = delaunay_tetrahedralization(grid_points_jax)
                
                # Compute APMD for this k-point
                apmd_1d_kpt = cal_apmd_1d(
                    crystal_direction=crystal_direction,
                    qz=qz_full,
                    grid_points=grid_points_jax,
                    density=density_jax,
                    tetrahedra=tetrahedra
                )
                
                # Add to accumulator
                apmd_1d_sum += apmd_1d_kpt
                valid_kpts += 1
                
                print(f"  Processed k-point {ikpt+1}/{len(grid_points_list)} ({len(grid_points_kpt)} points)")
            
            if valid_kpts == 0:
                print("Error: No valid k-points found!")
                sys.exit(1)
                
            # Average over k-points
            apmd_1d_full = apmd_1d_sum / valid_kpts
            print(f"  Averaged over {valid_kpts} valid k-points")
        
        # Symmetrize and take positive half
        n_points = len(qz_full)
        n_center = n_points // 2
        qz = qz_full[n_center:]  # positive half including zero
        
        # Average symmetric points for positive q values
        apmd_1d = jnp.zeros_like(qz)
        apmd_1d = apmd_1d.at[0].set(apmd_1d_full[n_center])  # q=0 point
        for i in range(1, len(qz)):
            # Average positive and negative q values
            apmd_1d = apmd_1d.at[i].set((apmd_1d_full[n_center + i] + apmd_1d_full[n_center - i]) / 2.0)
       
        # Save results to file
        method_suffix = '_merged' if merge_kpoints else '_averaged'
        output_filename = f'apmd_1d_{crystal_direction}{method_suffix}.txt'
        apmd_data = jnp.array([qz, apmd_1d]).T
        
        with open(output_filename, 'w') as apmd_file:
            np.savetxt(apmd_file, apmd_data, fmt='%.6f')
        
        print(f"\nSuccessfully generated: {output_filename}")
        print(f"APMD 1D Doppler spectrum computation completed!")
        
    except Exception as e:
        print(f"Error computing APMD: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()