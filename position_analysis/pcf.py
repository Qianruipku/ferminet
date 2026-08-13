"""Compute pair correlation function (PCF) for trajectories.

Example: all particles to last particle PCF.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from ferminet.utils.min_distance import Lattice, min_image_distance_triclinic

try:
  from position_analysis.read_positions import read_positions
except Exception:
  base = pathlib.Path(__file__).resolve().parent
  rp = base / 'read_positions.py'
  if rp.exists():
    spec = importlib.util.spec_from_file_location('position_analysis.read_positions', str(rp))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    read_positions = mod.read_positions
  else:
    from read_positions import read_positions


def _to_step_batch_view(positions: np.ndarray) -> np.ndarray:
  arr = np.asarray(positions)
  if arr.ndim == 5:
    nproc, nsteps, batch_local, nelec, dim = arr.shape
    return arr.transpose(1, 0, 2, 3, 4).reshape(nsteps, nproc * batch_local, nelec, dim)
  if arr.ndim == 4:
    return arr
  raise ValueError(f'Unsupported positions shape {arr.shape}')


def _select_view(view: np.ndarray, batch_indices: Optional[Sequence[int]], electrons: Optional[Sequence[int]]) -> np.ndarray:
  out = view
  if batch_indices is not None:
    out = out[:, batch_indices, :, :]
  if electrons is not None:
    out = out[:, :, electrons, :]
  return out


def compute_pcf(positions: np.ndarray,
                target_index: int,
                cell_matrix: np.ndarray,
                radius: int,
                nbins: int,
                rmax: float) -> tuple[np.ndarray, np.ndarray]:
  """Compute PCF (radial histogram) for distances between all electrons and target.

  positions: (nsteps, nbatch, nelec, dim)
  Returns (bin_centers, g_r) where g_r is averaged histogram normalized by number of samples.
  """
  nsteps, nbatch, nelec, dim = positions.shape
  # build pair vectors: for each step and batch, compute vectors from other electrons to target
  tgt = target_index if target_index >= 0 else nelec + target_index
  if tgt < 0 or tgt >= nelec:
    raise ValueError('target_index out of range')
  # shape (nsteps, nbatch, nelec-1, dim)
  others = [i for i in range(nelec) if i != tgt]
  vecs = positions[:, :, others, :] - positions[:, :, tgt:tgt+1, :]
  # reshape to (Nsamples, dim)
  vecs_flat = vecs.reshape(-1, dim)
  if vecs_flat.size == 0:
    return np.array([]), np.array([])
  lat = Lattice(jnp.asarray(cell_matrix))
  vecs_min, _ = min_image_distance_triclinic(jnp.asarray(vecs_flat), lat, radius=radius)
  vecs_min = np.asarray(vecs_min)
  dists = np.linalg.norm(vecs_min, axis=-1)
  # histogram
  bins = np.linspace(0.0, rmax, nbins + 1)
  counts, edges = np.histogram(dists, bins=bins)
  bin_centers = 0.5 * (edges[:-1] + edges[1:])
  # normalize by number of samples and shell volume
  Nsamples = vecs_flat.shape[0]
  shell_vol = 4.0 * np.pi * bin_centers**2 * (edges[1] - edges[0])
  # avoid division by zero for bin_centers[0]==0
  shell_vol[0] = 4.0/3.0 * np.pi * (edges[1]**3) if shell_vol.size > 0 else 1.0
  density = 1.0  # unknown absolute density; leave as counts per shell per sample
  g_r = counts / (Nsamples * shell_vol * density)
  return bin_centers, g_r


def plot_pcf(bin_centers: np.ndarray, g_r: np.ndarray, out_png: str) -> None:
  plt.figure(figsize=(6, 4))
  plt.plot(bin_centers, g_r, lw=2)
  plt.xlabel('r')
  plt.ylabel('g(r)')
  plt.title('Pair correlation function')
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  print(f'Saved PCF plot to {out_png}')


def analyze_pcf(path: str,
                out: Optional[str] = None,
                batch_indices: Optional[Sequence[int]] = None,
                electrons: Optional[Sequence[int]] = None,
                proc_index: Optional[int] = None,
                start: Optional[int] = None,
                end: Optional[int] = None,
                target_index: int = -1,
                nbins: int = 100,
                rmax: float = 5.0,
                lattice_constant: float = 1.0,
                cell_matrix: Optional[np.ndarray] = None) -> None:
  read_path = path
  if proc_index is not None and os.path.isdir(path):
    target = os.path.join(path, f'pos{int(proc_index)}_all.h5')
    if not os.path.isfile(target):
      raise FileNotFoundError(f'No file named {os.path.basename(target)!r} found in {path!r}')
    read_path = target
  raw = np.asarray(read_positions(read_path, start=start, end=end))
  view = _to_step_batch_view(raw)
  nsteps, nbatch, nelec, dim = view.shape
  print(f'Loaded positions shape(raw): {raw.shape}')
  print(f'Converted to (nsteps, batch, nelec, dim): {view.shape}')

  batch_sel = None
  if batch_indices is not None:
    batch_sel = np.asarray(batch_indices, dtype=np.int64)
  elec_sel = None
  if electrons is not None:
    elec_sel = np.asarray(electrons, dtype=np.int64)

  view = _select_view(view, batch_sel, elec_sel)
  print(f'Selected view shape (nsteps, selected_batch, selected_elec, dim): {view.shape}')

  if cell_matrix is None:
    # default to identity scaled by lattice_constant
    cell_matrix = np.eye(dim) * float(lattice_constant)
  cell_matrix = np.asarray(cell_matrix, dtype=np.float64)

  radius = 0
  bin_centers, g_r = compute_pcf(view, target_index, cell_matrix, radius, nbins, rmax)

  if out is None:
    out = os.path.join(path if os.path.isdir(path) else os.path.dirname(path), 'pcf.png')
  plot_pcf(bin_centers, g_r, out)

  npz_out = f'{os.path.splitext(out)[0]}.npz'
  np.savez_compressed(npz_out,
                      bin_centers=bin_centers,
                      g_r=g_r,
                      selected_batch_indices=(batch_sel if batch_sel is not None else np.arange(nbatch, dtype=np.int64)),
                      selected_electrons=(elec_sel if elec_sel is not None else np.arange(nelec, dtype=np.int64)),
                      nsteps=np.array([nsteps], dtype=np.int64),
                      nbatch=np.array([nbatch], dtype=np.int64),
                      nelec=np.array([nelec], dtype=np.int64))
  print(f'Saved PCF arrays to {npz_out}')


def main() -> None:
  p = argparse.ArgumentParser(description='Compute pair correlation function from saved trajectories')
  p.add_argument('path', type=str, help='HDF5 file path or folder containing pos*_all.h5')
  p.add_argument('--out', type=str, default=None, help='Output png file path')
  p.add_argument('--batch-indices', type=str, default=None,
                 help='Comma/range batch indices to select, e.g. 0,3,5-8')
  p.add_argument('--electrons', type=str, default=None,
                 help='Comma/range electron indices to select, e.g. 0,2,5-8')
  p.add_argument('--proc-index', type=int, default=None,
                 help='Only read one process file when path is a folder: pos{proc_index}_all.h5')
  p.add_argument('--start', type=int, default=None, help='Start step index (inclusive)')
  p.add_argument('--end', type=int, default=None, help='End step index (inclusive)')
  p.add_argument('--target', type=int, default=-1, help='Target electron index (default last)')
  p.add_argument('--nbins', type=int, default=100, help='Number of radial bins')
  p.add_argument('--rmax', type=float, default=5.0, help='Maximum radius for histogram')
  p.add_argument('--lattice-constant', type=float, default=1.0,
                 help='Lattice constant used when --cell-matrix is not provided.')
  p.add_argument('--cell-matrix', type=str, default=None,
                 help=('Full 3x3 cell matrix as 9 comma-separated floats '
                       '(row-major): a11,a12,a13,a21,a22,a23,a31,a32,a33.'))
  args = p.parse_args()

  def parse_indices(tok: Optional[str]) -> Optional[list[int]]:
    if tok is None:
      return None
    tmp = []
    for token in tok.split(','):
      token = token.strip()
      if not token:
        continue
      if '-' in token:
        a, b = token.split('-', 1)
        ia = int(a); ib = int(b)
        tmp.extend(range(ia, ib + 1))
      else:
        tmp.append(int(token))
    return sorted(set(tmp))

  batch_indices = parse_indices(args.batch_indices)
  electrons = parse_indices(args.electrons)

  cell_matrix = None
  if args.cell_matrix is not None:
    toks = [t.strip() for t in args.cell_matrix.split(',') if t.strip()]
    if len(toks) != 9:
      raise ValueError('--cell-matrix must contain 9 comma-separated floats')
    vals = np.asarray([float(t) for t in toks], dtype=np.float64)
    cell_matrix = vals.reshape(3, 3)

  analyze_pcf(path=args.path,
              out=args.out,
              batch_indices=batch_indices,
              electrons=electrons,
              proc_index=args.proc_index,
              start=args.start,
              end=args.end,
              target_index=args.target,
              nbins=args.nbins,
              rmax=args.rmax,
              lattice_constant=args.lattice_constant,
              cell_matrix=cell_matrix)


if __name__ == '__main__':
  main()
