"""Analyze per-batch trajectories and distance-to-mean distributions.

Input positions are expected to be shaped like
(1, nsteps, num_devices*batch_per_device, num_electrons, 3) for a single file,
or (num_processes, nsteps, local_batch, num_electrons, 3) for a folder.

For each global batch index, this script:
1) computes the mean position over all steps (per electron),
2) computes distances to the mean position at each step,
3) plots probability density distributions of those distances.

Example:
  python -m position_analysis.trajectory checkpoint_training \
  --bins 200 --batch-indices 0-15 --electrons 0-1 --out traj_dist.png
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
  """Convert input into shape (nsteps, global_batch, nelec, 3)."""
  arr = np.asarray(positions)

  if arr.ndim == 5:
    # (nproc, nsteps, batch_local, nelec, 3) -> (nsteps, nproc*batch_local, nelec, 3)
    nproc, nsteps, batch_local, nelec, dim = arr.shape
    if dim != 3:
      raise ValueError(f'Expected last dim 3, got shape {arr.shape}')
    arr = arr.transpose(1, 0, 2, 3, 4).reshape(nsteps, nproc * batch_local, nelec, 3)
    return arr

  if arr.ndim == 4:
    # Already (nsteps, batch, nelec, 3)
    if arr.shape[-1] != 3:
      raise ValueError(f'Expected last dim 3, got shape {arr.shape}')
    return arr

  raise ValueError(
      'Unsupported positions shape. Expected 4D or 5D array, got '
      f'{arr.shape} (ndim={arr.ndim}).')


def _default_cell_matrix(lattice_type: str, lattice_constant: float) -> np.ndarray:
  t = lattice_type.lower()
  a = float(lattice_constant)
  if t == 'sc':
    return np.diag([a, a, a]).astype(np.float64)
  if t == 'bcc':
    return ((a / 2.0) * np.array([[-1.0, 1.0, 1.0],
                                  [1.0, -1.0, 1.0],
                                  [1.0, 1.0, -1.0]], dtype=np.float64).T)
  if t == 'fcc':
    return ((a / 2.0) * np.array([[0.0, 1.0, 1.0],
                                  [1.0, 0.0, 1.0],
                                  [1.0, 1.0, 0.0]], dtype=np.float64).T)
  raise ValueError(f'Unsupported lattice type: {lattice_type}')


def _radius_for_lattice(lattice_type: str) -> int:
  t = lattice_type.lower()
  if t == 'sc':
    return 0
  if t in ('bcc', 'fcc'):
    return 1
  raise ValueError(f'Unsupported lattice type: {lattice_type}')


def _cumulative_displacement_with_min_distance(
    positions: np.ndarray,
    cell_matrix: np.ndarray,
    radius: int,
) -> np.ndarray:
  """Compute cumulative displacement using minimum-image step vectors."""
  if positions.ndim != 4 or positions.shape[-1] != 3:
    raise ValueError(f'Expected positions shape (nsteps, nbatch, nelec, 3), got {positions.shape}')

  lat = Lattice(jnp.asarray(cell_matrix))
  nsteps, nbatch, nelec, _ = positions.shape
  ntraj = nbatch * nelec

  if nsteps <= 1:
    return np.zeros((nsteps, ntraj, 3), dtype=positions.dtype)

  step_dr = positions[1:] - positions[:-1]  # (nsteps-1, nbatch, nelec, 3)
  step_dr_flat = step_dr.reshape(-1, 3)
  step_dr_min_flat, _ = min_image_distance_triclinic(
      jnp.asarray(step_dr_flat), lat, radius=radius)
  step_dr_min = np.asarray(step_dr_min_flat).reshape(nsteps - 1, ntraj, 3)

  cumulative = np.zeros((nsteps, ntraj, 3), dtype=positions.dtype)
  cumulative[1:] = np.cumsum(step_dr_min, axis=0)
  return cumulative


def _select_view(
    view: np.ndarray,
    batch_indices: Optional[np.ndarray],
    electrons: Optional[np.ndarray],
) -> np.ndarray:
  """Select batches/electrons from (nsteps, nbatch, nelec, 3) view."""
  out = view
  if batch_indices is not None:
    out = out[:, batch_indices, :, :]
  if electrons is not None:
    out = out[:, :, electrons, :]
  return out


def _fold_points_to_cell(points: np.ndarray, cell_matrix: np.ndarray) -> np.ndarray:
  """Fold Cartesian points into the primary cell defined by cell_matrix."""
  inv_cell = np.linalg.inv(cell_matrix)
  frac = points @ inv_cell.T
  frac_folded = np.mod(frac, 1.0)
  return frac_folded @ cell_matrix.T


def compute_average_distribution(
    cumulative_displacement: np.ndarray,
    bins: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Compute distribution relative to each trajectory's mean position.

  Args:
    cumulative_displacement: array with shape (nsteps, ntraj, 3).
    bins: histogram bins.

  Returns:
    distances: (nsteps, ntraj)
    edges: (bins+1,)
    pdf: (bins,)
  """
  if cumulative_displacement.ndim != 3 or cumulative_displacement.shape[-1] != 3:
    raise ValueError(
        f'Expected cumulative_displacement shape (nsteps, ntraj, 3), got {cumulative_displacement.shape}')

  centered = cumulative_displacement - cumulative_displacement.mean(axis=0, keepdims=True)
  distances = np.linalg.norm(centered, axis=-1)
  distances_flat = distances.reshape(-1)

  dmax = float(np.max(distances_flat)) if distances_flat.size else 1.0
  if dmax <= 0:
    dmax = 1.0

  edges = np.linspace(0.0, dmax, bins + 1)
  pdf, _ = np.histogram(distances_flat, bins=edges, density=True)
  return distances, edges, pdf


def plot_distribution_mean(
    edges: np.ndarray,
    pdf: np.ndarray,
    out_png: str,
) -> np.ndarray:
  """Plot single distance distribution curve."""
  centers = 0.5 * (edges[:-1] + edges[1:])

  plt.figure(figsize=(8, 5.5))
  plt.plot(centers, pdf, lw=2.0, label='distribution')

  plt.xlabel('Distance to mean position')
  plt.ylabel('Probability density')
  plt.title('Distance distribution around per-trajectory mean')
  plt.grid(alpha=0.25)
  plt.legend(fontsize=9)
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  print(f'Saved distribution plot to {out_png}')

  return centers


def analyze_trajectory(
    path: str,
    out: Optional[str] = None,
    bins: int = 200,
    batch_indices: Optional[Sequence[int]] = None,
    electrons: Optional[Sequence[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    lattice_type: str = 'sc',
    lattice_constant: float = 1.0,
    cell_matrix: Optional[np.ndarray] = None,
) -> None:
  raw = np.asarray(read_positions(path, start=start, end=end))
  view = _to_step_batch_view(raw)

  nsteps, nbatch, nelec, _ = view.shape
  print(f'Loaded positions shape(raw): {raw.shape}')
  print(f'Converted to (nsteps, batch, nelec, 3): {view.shape}')

  batch_sel = None
  if batch_indices is not None:
    batch_sel = np.asarray(batch_indices, dtype=np.int64)
    if batch_sel.size > 0 and (batch_sel.min() < 0 or batch_sel.max() >= nbatch):
      raise ValueError(f'batch_indices out of bounds. Valid range: [0, {nbatch - 1}]')
  elec_sel = None
  if electrons is not None:
    elec_sel = np.asarray(electrons, dtype=np.int64)
    if elec_sel.size > 0 and (elec_sel.min() < 0 or elec_sel.max() >= nelec):
      raise ValueError(f'electrons out of bounds. Valid range: [0, {nelec - 1}]')

  view = _select_view(view, batch_sel, elec_sel)
  print(f'Selected view shape (nsteps, selected_batch, selected_elec, 3): {view.shape}')

  if cell_matrix is None:
    cell_matrix = _default_cell_matrix(lattice_type, lattice_constant)
  cell_matrix = np.asarray(cell_matrix, dtype=np.float64)

  selected_count = view.shape[1] * view.shape[2]
  if selected_count < 10:
    init_pos = view[0].reshape(-1, 3)
    init_pos_folded = _fold_points_to_cell(init_pos, cell_matrix)
    print(f'Selected trajectory count is small ({selected_count} < 10).')
    print('Initial positions at step 0 folded into the primary cell ([selected_batch*selected_elec, 3]):')
    print(np.array2string(init_pos_folded, precision=6, suppress_small=True))

  radius = _radius_for_lattice(lattice_type)

  cumulative = _cumulative_displacement_with_min_distance(view, cell_matrix, radius)
  distances, edges, pdf = compute_average_distribution(cumulative, bins=bins)

  if out is None:
    out = os.path.join(path if os.path.isdir(path) else os.path.dirname(path),
                       'trajectory_distance_distribution.png')

  centers = plot_distribution_mean(edges=edges, pdf=pdf, out_png=out)

  npz_out = f'{os.path.splitext(out)[0]}.npz'
  np.savez_compressed(
      npz_out,
      cumulative_displacement=cumulative,
      distances=distances,
      hist_edges=edges,
      hist_centers=centers,
      hist_pdf=pdf,
      selected_batch_indices=(batch_sel if batch_sel is not None else np.arange(nbatch, dtype=np.int64)),
      selected_electrons=(elec_sel if elec_sel is not None else np.arange(nelec, dtype=np.int64)),
      nsteps=np.array([nsteps], dtype=np.int64),
      nbatch=np.array([nbatch], dtype=np.int64),
      nelec=np.array([nelec], dtype=np.int64),
  )
  print(f'Saved analysis arrays to {npz_out}')


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Analyze per-batch particle trajectories and distance-to-mean distributions.')
  parser.add_argument('path', type=str, help='HDF5 file path or folder containing pos*_all.h5')
  parser.add_argument('--out', type=str, default=None, help='Output png file path')
  parser.add_argument('--bins', type=int, default=200, help='Histogram bins')
  parser.add_argument('--batch-indices', type=str, default=None,
                      help='Comma/range batch indices to select, e.g. 0,3,5-8')
  parser.add_argument('--electrons', type=str, default=None,
                      help='Comma/range electron indices to select, e.g. 0,2,5-8')
  parser.add_argument('--start', type=int, default=None,
                      help='Start step index (inclusive)')
  parser.add_argument('--end', type=int, default=None,
                      help='End step index (inclusive)')
  parser.add_argument('--lattice-type', type=str, default='sc', choices=['sc', 'bcc', 'fcc'],
          help='Lattice type used to choose min-image search radius.')
  parser.add_argument('--lattice-constant', type=float, default=1.0,
          help='Lattice constant used when --cell-matrix is not provided.')
  parser.add_argument('--cell-matrix', type=str, default=None,
                      help=('Full 3x3 cell matrix as 9 comma-separated floats '
                            '(row-major): a11,a12,a13,a21,a22,a23,a31,a32,a33. '
          'If set, this matrix is used directly for min-image distances.'))

  args = parser.parse_args()

  # Parse optional explicit batch list after loading would also work, but parsing
  # here keeps CLI errors explicit.
  batch_indices = None
  if args.batch_indices is not None:
    # Temporarily allow parsing; bounds are validated after data shape is known.
    tmp = []
    for token in args.batch_indices.split(','):
      token = token.strip()
      if not token:
        continue
      if '-' in token:
        a, b = token.split('-', 1)
        ia = int(a)
        ib = int(b)
        if ib < ia:
          raise ValueError(f'Invalid range: {token}')
        tmp.extend(range(ia, ib + 1))
      else:
        tmp.append(int(token))
    batch_indices = sorted(set(tmp))

  electrons = None
  if args.electrons is not None:
    tmp = []
    for token in args.electrons.split(','):
      token = token.strip()
      if not token:
        continue
      if '-' in token:
        a, b = token.split('-', 1)
        ia = int(a)
        ib = int(b)
        if ib < ia:
          raise ValueError(f'Invalid range: {token}')
        tmp.extend(range(ia, ib + 1))
      else:
        tmp.append(int(token))
    electrons = sorted(set(tmp))

  cell_matrix = None
  if args.cell_matrix is not None:
    toks = [t.strip() for t in args.cell_matrix.split(',') if t.strip()]
    if len(toks) != 9:
      raise ValueError('--cell-matrix must contain 9 comma-separated floats')
    vals = np.asarray([float(t) for t in toks], dtype=np.float64)
    cell_matrix = vals.reshape(3, 3)

  analyze_trajectory(
      path=args.path,
      out=args.out,
      bins=args.bins,
      batch_indices=batch_indices,
      electrons=electrons,
      start=args.start,
      end=args.end,
      lattice_type=args.lattice_type,
      lattice_constant=args.lattice_constant,
      cell_matrix=cell_matrix,
  )


if __name__ == '__main__':
  main()
