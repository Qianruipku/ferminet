#!/usr/bin/env python3
"""Utility script for comparing NPZ, CSV, TXT, and NPY files."""

import argparse
import os
import numpy as np
from typing import Any, Dict, Union

# Try to import pandas for CSV comparison
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def compare_txt_files(txt1_path: str, txt2_path: str, 
                     atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    """Compare two txt files with first line as text header and rest as numerical data."""
    if not os.path.exists(txt1_path):
        print(f"  TXT file {txt1_path} not found")
        return False
    
    if not os.path.exists(txt2_path):
        print(f"  TXT file {txt2_path} not found")
        return False
    
    try:
        with open(txt1_path, 'r') as f1, open(txt2_path, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
    except Exception as e:
        print(f"  Error reading TXT files: {e}")
        return False
    
    # Check if files have data
    if len(lines1) == 0 or len(lines2) == 0:
        print(f"  Empty file(s): {len(lines1)} vs {len(lines2)} lines")
        return False
    
    # Compare first line (text header)
    header1 = lines1[0].strip()
    header2 = lines2[0].strip()
    if header1 != header2:
        print(f"  Header mismatch: '{header1}' vs '{header2}'")
        return False
    
    # Compare number of data lines
    if len(lines1) != len(lines2):
        print(f"  Line count mismatch: {len(lines1)} vs {len(lines2)}")
        return False
    
    # Compare numerical data (skip first line)
    all_match = True
    for i, (line1, line2) in enumerate(zip(lines1[1:], lines2[1:]), start=2):
        try:
            # Parse numerical data from each line
            data1 = np.array([float(x) for x in line1.strip().split()])
            data2 = np.array([float(x) for x in line2.strip().split()])
            
            if len(data1) != len(data2):
                print(f"  Line {i} column count mismatch: {len(data1)} vs {len(data2)}")
                all_match = False
                continue
            
            # Use both relative and absolute tolerance
            if not np.allclose(data1, data2, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(data1 - data2))
                print(f"  Line {i} data differs: max difference = {max_diff}")
                all_match = False
        except ValueError as e:
            print(f"  Line {i} parsing error: {e}")
            print(f"    Line 1: '{line1.strip()}'")
            print(f"    Line 2: '{line2.strip()}'")
            all_match = False
    
    return all_match


def compare_npy_files(npy1_path: str, npy2_path: str, 
                     atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    """Compare two .npy files."""
    if not os.path.exists(npy1_path):
        print(f"  NPY file {npy1_path} not found")
        return False
    
    if not os.path.exists(npy2_path):
        print(f"  NPY file {npy2_path} not found")
        return False
    
    try:
        arr1 = np.load(npy1_path, allow_pickle=True)
        arr2 = np.load(npy2_path, allow_pickle=True)
    except Exception as e:
        print(f"  Error reading NPY files: {e}")
        return False
    
    # Use the existing array comparison function
    return compare_arrays(arr1, arr2, atol, rtol, "npy_data")


def compare_csv_files(csv1_path: str, csv2_path: str, 
                     atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    """Compare two CSV files with numerical tolerance."""
    if not HAS_PANDAS:
        print("  Error: pandas is required for CSV comparison but not installed")
        return False
        
    if not os.path.exists(csv1_path):
        print(f"  CSV file {csv1_path} not found")
        return False
    
    if not os.path.exists(csv2_path):
        print(f"  CSV file {csv2_path} not found")
        return False
    
    try:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
    except Exception as e:
        print(f"  Error reading CSV files: {e}")
        return False
    
    # Compare shapes
    if df1.shape != df2.shape:
        print(f"  CSV shape mismatch: {df1.shape} vs {df2.shape}")
        return False
    
    # Compare column names
    if list(df1.columns) != list(df2.columns):
        print(f"  CSV column mismatch: {list(df1.columns)} vs {list(df2.columns)}")
        return False
    
    # Compare data
    all_match = True
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            # Numerical comparison with tolerance
            arr1 = np.asarray(df1[col].values)
            arr2 = np.asarray(df2[col].values)
            
            # Handle NaN values
            if np.isnan(arr1).any() or np.isnan(arr2).any():
                if not np.array_equal(np.isnan(arr1), np.isnan(arr2)):
                    print(f"  CSV column '{col}' NaN pattern differs")
                    all_match = False
                    continue
                # Compare non-NaN values
                mask = ~np.isnan(arr1)
                if mask.any():
                    if not np.allclose(arr1[mask], arr2[mask], rtol=rtol, atol=atol):
                        max_diff = np.max(np.abs(arr1[mask] - arr2[mask]))
                        print(f"  CSV column '{col}' differs: max difference = {max_diff}")
                        all_match = False
            else:
                if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
                    max_diff = np.max(np.abs(arr1 - arr2))
                    print(f"  CSV column '{col}' differs: max difference = {max_diff}")
                    all_match = False
        else:
            # Exact comparison for non-numerical columns
            if not df1[col].equals(df2[col]):
                print(f"  CSV column '{col}' differs (non-numerical)")
                all_match = False
    
    return all_match


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, 
                  atol: float = 1e-6, rtol: float = 1e-6, path: str = "") -> bool:
    """Compare two numpy arrays with tolerance."""
    if arr1.shape != arr2.shape:
        print(f"  Shape mismatch at {path}: {arr1.shape} vs {arr2.shape}")
        return False
    
    # Handle None values or other non-numeric data
    try:
        # Check if arrays contain None or other non-numeric values
        if arr1.dtype == object or arr2.dtype == object:
            # For object arrays, compare element by element
            flat1 = arr1.flatten()
            flat2 = arr2.flatten()
            for i, (v1, v2) in enumerate(zip(flat1, flat2)):
                if v1 is None and v2 is None:
                    continue
                elif v1 is None or v2 is None:
                    print(f"  None mismatch at {path}[{i}]: {v1} vs {v2}")
                    return False
                elif isinstance(v1, (int, float, complex)) and isinstance(v2, (int, float, complex)):
                    # For individual values, use absolute tolerance as a simple threshold
                    if abs(v1 - v2) > max(atol, rtol * max(abs(v1), abs(v2))):
                        print(f"  Value mismatch at {path}[{i}]: {v1} vs {v2}")
                        return False
                elif v1 != v2:
                    print(f"  Value mismatch at {path}[{i}]: {v1} vs {v2}")
                    return False
            return True
        else:
            # For numeric arrays, use allclose with both relative and absolute tolerance
            if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(arr1 - arr2))
                print(f"  Value mismatch at {path}: max difference = {max_diff}")
                return False
            return True
    except (TypeError, ValueError) as e:
        print(f"  Error comparing arrays at {path}: {e}")
        return False


def compare_nested_structure(obj1: Any, obj2: Any, 
                           atol: float = 1e-6, rtol: float = 1e-6, path: str = "") -> bool:
    """Recursively compare nested structures (dicts, lists, arrays)."""
    if type(obj1) != type(obj2):
        print(f"  Type mismatch at {path}: {type(obj1)} vs {type(obj2)}")
        return False
    
    if isinstance(obj1, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            print(f"  Key mismatch at {path}: {set(obj1.keys())} vs {set(obj2.keys())}")
            return False
        
        all_match = True
        for key in obj1.keys():
            new_path = f"{path}.{key}" if path else key
            if not compare_nested_structure(obj1[key], obj2[key], atol, rtol, new_path):
                all_match = False
        return all_match
    
    elif isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            print(f"  Length mismatch at {path}: {len(obj1)} vs {len(obj2)}")
            return False
        
        all_match = True
        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            if not compare_nested_structure(item1, item2, atol, rtol, new_path):
                all_match = False
        return all_match
    
    elif isinstance(obj1, np.ndarray):
        return compare_arrays(obj1, obj2, atol, rtol, path)
    
    elif isinstance(obj1, (int, float, complex)):
        # Use the same formula as allclose for individual values
        if abs(obj1 - obj2) > max(atol, rtol * max(abs(obj1), abs(obj2))):
            print(f"  Value mismatch at {path}: {obj1} vs {obj2}")
            return False
        return True
    
    else:
        # For other types, use equality
        # Handle JAX arrays and other array-like objects
        try:
            # Try to convert to numpy for comparison
            if hasattr(obj1, 'shape') and hasattr(obj2, 'shape'):
                # Treat as arrays
                return compare_arrays(np.asarray(obj1), np.asarray(obj2), atol, rtol, path)
            elif hasattr(obj1, '__dict__') and hasattr(obj2, '__dict__'):
                # Handle objects with attributes (like optimizer states)
                dict1 = obj1.__dict__ if hasattr(obj1, '__dict__') else {}
                dict2 = obj2.__dict__ if hasattr(obj2, '__dict__') else {}
                return compare_nested_structure(dict1, dict2, atol, rtol, path)
            else:
                # For scalar values, use safe comparison
                try:
                    # Try numpy comparison first for arrays
                    if hasattr(obj1, 'shape') or hasattr(obj2, 'shape'):
                        return compare_arrays(np.asarray(obj1), np.asarray(obj2), atol, rtol, path)
                    elif obj1 != obj2:
                        print(f"  Value mismatch at {path}: {obj1} vs {obj2}")
                        return False
                    return True
                except ValueError:
                    # If direct comparison fails, try converting to arrays
                    return compare_arrays(np.asarray(obj1), np.asarray(obj2), atol, rtol, path)
        except ValueError:
            # Handle JAX array comparison error
            if hasattr(obj1, 'shape') and hasattr(obj2, 'shape'):
                return compare_arrays(np.asarray(obj1), np.asarray(obj2), atol, rtol, path)
            elif hasattr(obj1, '__dict__') and hasattr(obj2, '__dict__'):
                # Handle objects with attributes (like optimizer states)
                dict1 = obj1.__dict__ if hasattr(obj1, '__dict__') else {}
                dict2 = obj2.__dict__ if hasattr(obj2, '__dict__') else {}
                return compare_nested_structure(dict1, dict2, atol, rtol, path)
            else:
                print(f"  Cannot compare objects at {path}: {type(obj1)} vs {type(obj2)}")
                return False


def load_npz_file(filepath: str) -> Dict[str, Any]:
    """Load an NPZ file and return its contents."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NPZ file {filepath} not found")
    
    return dict(np.load(filepath, allow_pickle=True))


def compare_npz(npz1_path: str, npz2_path: str, 
               atol: float = 1e-6, rtol: float = 1e-6, skip_keys: Union[list, None] = None) -> bool:
    """Compare two NPZ files."""
    if not os.path.exists(npz1_path):
        print(f"NPZ file {npz1_path} not found")
        return False
    if not os.path.exists(npz2_path):
        print(f"NPZ file {npz2_path} not found")
        return False
    
    if skip_keys is None:
        skip_keys = []
    
    print(f"Comparing NPZ files:")
    print(f"  File 1: {npz1_path}")
    print(f"  File 2: {npz2_path}")
    print(f"  Tolerance: atol={atol}, rtol={rtol}")
    if skip_keys:
        print(f"  Skipping keys: {skip_keys}")
    print()
    
    # Load NPZ files
    try:
        npz1 = load_npz_file(npz1_path)
        npz2 = load_npz_file(npz2_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    # Compare top-level keys
    keys1 = set(npz1.keys())
    keys2 = set(npz2.keys())
    
    if keys1 != keys2:
        print(f"Top-level keys differ:")
        print(f"  Only in file 1: {keys1 - keys2}")
        print(f"  Only in file 2: {keys2 - keys1}")
        return False
    
    print(f"Top-level keys: {sorted(keys1)}")
    print()
    
    # Compare each key
    all_match = True
    for key in sorted(keys1):
        if key in skip_keys:
            print(f"Skipping '{key}' (in skip list)")
            continue
            
        print(f"Comparing '{key}':")
        
        # Handle special cases for common NPZ keys
        if key == 'step':
            # For step, try to extract scalar value
            if hasattr(npz1[key], 'item') and npz1[key].size == 1:
                step1 = npz1[key].item()
            else:
                step1 = npz1[key]
                
            if hasattr(npz2[key], 'item') and npz2[key].size == 1:
                step2 = npz2[key].item()
            else:
                step2 = npz2[key]
                
            if step1 == step2:
                print(f"  ✓ Steps match: {step1}")
            else:
                print(f"  ✗ Steps differ: {step1} vs {step2}")
                all_match = False
        
        elif key in ['params', 'opt_state', 'data']:
            # Extract the actual data (often wrapped in numpy arrays)
            # Only use .item() for 0-dimensional arrays
            if hasattr(npz1[key], 'item') and npz1[key].ndim == 0:
                data1 = npz1[key].item()
            else:
                data1 = npz1[key]
                
            if hasattr(npz2[key], 'item') and npz2[key].ndim == 0:
                data2 = npz2[key].item()
            else:
                data2 = npz2[key]
            
            if compare_nested_structure(data1, data2, atol, rtol, key):
                print(f"  ✓ {key} matches")
            else:
                print(f"  ✗ {key} differs")
                if key == 'opt_state':
                    print(f"    Note: opt_state differences are often expected due to optimizer internals")
                all_match = False
        
        else:
            # Generic comparison
            if compare_nested_structure(npz1[key], npz2[key], atol, rtol, key):
                print(f"  ✓ {key} matches")
            else:
                print(f"  ✗ {key} differs")
                all_match = False
        
        print()
    
    return all_match


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare FermiNet NPZ, CSV, TXT, or NPY files",
        epilog="Numerical comparison uses numpy.allclose with formula: abs(a - b) <= (atol + rtol * abs(b))"
    )
    parser.add_argument("file1", help="Path to first file")
    parser.add_argument("file2", help="Path to second file")
    parser.add_argument("--tolerance", "-t", type=float, default=1e-6,
                       help="Tolerance for numerical comparisons (sets both atol and rtol, default: 1e-6)")
    parser.add_argument("--atol", type=float, default=None,
                       help="Absolute tolerance (default: same as --tolerance)")
    parser.add_argument("--rtol", type=float, default=None,
                       help="Relative tolerance (default: same as --tolerance)")
    parser.add_argument("--skip", "-s", nargs="+", default=[],
                       help="Keys to skip during comparison (for NPZ files only)")
    parser.add_argument("--type", choices=["auto", "npz", "csv", "txt", "npy"], default="auto",
                       help="File type to compare (auto-detect by default)")
    
    args = parser.parse_args()
    
    # Set atol and rtol if not explicitly provided
    atol = args.atol if args.atol is not None else args.tolerance
    rtol = args.rtol if args.rtol is not None else args.tolerance
    
    # Auto-detect file type
    file_type = args.type
    if file_type == "auto":
        if args.file1.endswith('.csv') and args.file2.endswith('.csv'):
            file_type = "csv"
        elif args.file1.endswith('.npz') and args.file2.endswith('.npz'):
            file_type = "npz"
        elif args.file1.endswith('.txt') and args.file2.endswith('.txt'):
            file_type = "txt"
        elif args.file1.endswith('.npy') and args.file2.endswith('.npy'):
            file_type = "npy"
        else:
            print("Error: Cannot auto-detect file type. Use --type to specify.")
            print("Supported extensions: .npz, .csv, .txt, .npy")
            exit(1)
    
    if file_type == "csv":
        print(f"Comparing CSV files:")
        print(f"  File 1: {args.file1}")
        print(f"  File 2: {args.file2}")
        print(f"  Tolerance: atol={atol}, rtol={rtol}")
        print()
        
        success = compare_csv_files(args.file1, args.file2, atol, rtol)
        
        if success:
            print("✓ CSV files match within tolerance!")
            exit(0)
        else:
            print("✗ CSV files differ!")
            exit(1)
    
    elif file_type == "txt":
        print(f"Comparing TXT files:")
        print(f"  File 1: {args.file1}")
        print(f"  File 2: {args.file2}")
        print(f"  Tolerance: atol={atol}, rtol={rtol}")
        print()
        
        success = compare_txt_files(args.file1, args.file2, atol, rtol)
        
        if success:
            print("✓ TXT files match within tolerance!")
            exit(0)
        else:
            print("✗ TXT files differ!")
            exit(1)
    
    elif file_type == "npy":
        print(f"Comparing NPY files:")
        print(f"  File 1: {args.file1}")
        print(f"  File 2: {args.file2}")
        print(f"  Tolerance: atol={atol}, rtol={rtol}")
        print()
        
        success = compare_npy_files(args.file1, args.file2, atol, rtol)
        
        if success:
            print("✓ NPY files match within tolerance!")
            exit(0)
        else:
            print("✗ NPY files differ!")
            exit(1)
    
    elif file_type == "npz":
        success = compare_npz(args.file1, args.file2, 
                             atol, rtol, args.skip)
        
        if success:
            print("✓ NPZ files match within tolerance!")
            exit(0)
        else:
            print("✗ NPZ files differ!")
            exit(1)


if __name__ == "__main__":
    main()
