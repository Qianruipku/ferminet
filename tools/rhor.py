#!/usr/bin/env python3
"""Radial distribution of electrons around a reference point or atom (rhor).

Usage examples:
    python rhor.py --dir ./positions --ref-pos 0.0 0.0 0.0 --rmax 5.0 --bins 100
    python rhor.py --dir ./positions --atoms-index-list 0 --rmax 6.0

Description:
    - The script loads all files matching `positions_*.npy` in the specified
        directory (sorted by filename).
    - Each positions file may be a multi-dimensional array (for example
        (nhosts, ndevices, batch, nelec, 3)) or a simple (N, 3) coordinate array.
        The script will reshape/collapse leading dimensions to produce arrays of
        shape (n_snapshots, n_particles, 3).
    - The reference can be provided with --ref-pos x y z or by specifying
        particle indices with --atoms-index-list (0-based indices). The script
        computes radial histograms of electron distances to the reference point(s).
    - Output is saved to an NPZ file (default: rhor.npz) containing
        bin_centers, counts, density_per_shell, dr, nbins, files_read, etc. A
        PNG plot can be saved with --save-plot (requires matplotlib).

Notes:
    - The script reports the average electron count per shell per reference
        event and divides by the shell volume to obtain a radial density n(r).
        To obtain a normalized pair distribution function g(r) you would need the
        system average electron density (this script does not perform that
        normalization since system volume or overall density may be unknown).
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Iterable, Tuple
import numpy as np
import jax.numpy as jnp
import math
from ferminet.utils.min_distance import min_image_distance_triclinic, Lattice


def parse_args():
    p = argparse.ArgumentParser(description='Compute radial electron distribution around a reference point or atom.')
    p.add_argument('--dir', '-d', default='.', help='Directory containing positions_*.npy files (default: current directory)')
    p.add_argument('--bins', type=int, default=200, help='Number of radial bins')
    p.add_argument('--rmax', type=float, default=10.0, help='Maximum radius to consider')
    p.add_argument('--ref-pos', '-rp', nargs=3, type=float, help='Reference point coordinates: x y z (overrides --atoms-file/--atom-index)')
    p.add_argument('--ref-index', '-ri', type=int, help='Index of atom in atoms-file to use as reference (0-based)')
    p.add_argument('--atoms-index-list', '-al', type=str, help='List of atom indices to calculate radial distribution around (comma-separated, 0-based)')
    return p.parse_args()

def parse_list_str(list_str: str) -> np.ndarray:
    """Parse a compact index string like "1, 2:3, 4:8:2, !1" and return
    a numpy integer array.

    Supported tokens:
      - "a"                -> include integer a
      - "start:stop"       -> include start, start+step, ..., stop (inclusive)
      - "start:stop:step"  -> same but with explicit step (step may be
                              negative for descending ranges)
      - "~n"               -> exclude integer n (multiple exclusions allowed)
    """
    if not list_str:
        return np.array([], dtype=int)

    include_vals = []
    exclude_vals = set()

    def parse_token(tok: str) -> Iterable[int]:
        tok = tok.strip()
        if not tok:
            return []
        # single integer
        if ':' not in tok:
            return [int(tok)]
        parts = tok.split(':')
        if len(parts) == 2:
            start = int(parts[0])
            stop = int(parts[1])
            step = 1 if stop >= start else -1
        elif len(parts) == 3:
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2])
            if step == 0:
                raise ValueError(f'step cannot be 0 in token "{tok}"')
        else:
            raise ValueError(f'Unrecognized range token: "{tok}"')
        # include stop (inclusive range)
        if step > 0:
            return list(range(start, stop + 1, step))
        else:
            return list(range(start, stop - 1, step))

    for raw in list_str.split(','):
        s = raw.strip()
        if not s:
            continue
        if s.startswith('~'):
            excl = s[1:].strip()
            if not excl:
                raise ValueError(f'Empty exclusion token in "{raw}"')
            exclude_vals.add(int(excl))
        else:
            vals = parse_token(s)
            include_vals.extend(vals)

    # Deduplicate and sort. If preserving original order is desired, use an
    # OrderedDict instead.
    result_set = set(include_vals)
    result_set.difference_update(exclude_vals)
    result = sorted(result_set)
    return np.array(result, dtype=int)

def reshape_positions(arr: np.ndarray, atom_list: np.ndarray, ref_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reshape a positions array into shape (n_snapshots, n_particles, 3).

    The input array may use a flattened last dimension (e.g. final dim = 3*n)
    or may already have a trailing dimension of 3. This function returns a
    JAX array of shape (n_snapshots, n_selected_particles, 3) according to the
    supplied atom_list and optional ref_idx.
    """
    shape = arr.shape
    assert shape[-1] % 3 == 0, f'Last dimension must be multiple of 3, got {shape[-1]}'
    nparticles = shape[-1] // 3
    arr = arr.reshape(-1, nparticles, 3)
    if atom_list is None:
        atom_list = np.arange(nparticles)
    
    ref_pos = None
    if ref_idx is not None:
        # remove ref_idx from atom_list
        ref_idx = (ref_idx if ref_idx >= 0 else nparticles + ref_idx)
        atom_list = atom_list[atom_list != ref_idx]
        ref_pos = arr[:, ref_idx, :].reshape(-1, 1, 3)  # shape (n_snapshots, 1, 3)
    result = arr[:, atom_list, :]

    return jnp.array(result), jnp.array(ref_pos) if ref_pos is not None else None


def main():
    args = parse_args()
    directory = args.dir
    if not os.path.isdir(directory):
        print(f'Error: directory {directory} does not exist', file=sys.stderr)
        sys.exit(2)

    # find files
    files = sorted([f for f in os.listdir(directory) if f.startswith('positions_') and f.endswith('.npy')])
    if not files:
        print(f'No positions_*.npy files found in {directory}', file=sys.stderr)
        sys.exit(1)

    bins = args.bins
    rmax = args.rmax
    edges = np.linspace(0.0, rmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    counts = np.zeros(bins, dtype=float)
    lattice_vec = jnp.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
    lat = Lattice(lattice_vec)

    files_read = []
    total_electrons_counted = 0

    if args.ref_pos is not None:
        ref_point = np.array(args.ref_pos, dtype=float).reshape(1, 3)
    else:
        ref_point = np.array([0.0, 0.0, 0.0]).reshape(1, 3)

    ref_idx = None
    if args.ref_index is not None:
        ref_idx = args.ref_index
    
    if args.atoms_index_list is not None:
        atom_list = parse_list_str(args.atoms_index_list)
    else:
        atom_list = None

    for fname in files:
        path = os.path.join(directory, fname)
        arr = np.load(path, allow_pickle=True)
        pos, ref_pos = reshape_positions(arr, atom_list, ref_idx)

    # compute distances
        if ref_pos is None:
            disp = pos - ref_point
        else:
            disp = pos - ref_pos
        disp = jnp.reshape(disp, (-1, 3))
        _, d = min_image_distance_triclinic(disp, lat, radius=1)
    # keep r <= rmax
        d = d[d <= rmax]
        hist, _ = jnp.histogram(d, bins=edges)
        counts += hist
        files_read.append(fname)
        total_electrons_counted += d.size
        print(f'Processed {fname}: counted {d.size} particles within rmax')

    n_snapshots = len(files_read) * pos.shape[0]
    if n_snapshots == 0:
        print('No valid snapshots processed', file=sys.stderr)
        sys.exit(1)



    # shell volumes for each radial bin
    shell_vol = 4.0 * math.pi * centers**2 * dr
    # density per shell: average counts per snapshot divided by shell volume
    density_per_shell = counts / float(n_snapshots) / shell_vol

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available; cannot save plot', file=sys.stderr)
        return
    plt.figure(figsize=(6,4))
    plt.plot(centers, density_per_shell, '-o')
    plt.xlabel('r')
    plt.ylabel('n(r) (per shell volume)')
    plt.title(f'radial density around {ref_point}')
    plt.grid(True)
    pngname =  'rhor.png'
    plt.savefig(pngname, dpi=200)
    print(f'Saved plot to {pngname}')


if __name__ == '__main__':
    main()
