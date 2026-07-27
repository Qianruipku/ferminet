"""Plot 3D scatter of a specific particle in a specific batch across saved steps.

Reads the same `pos*_all.h5` files as `position_analysis.read_positions` and
collects the coordinates of a single electron (particle index) from a single
batch index across all saved steps/files, then plots and saves a 3D scatter.

Example:
  python -m position_analysis.plot_scatter3d positions --particle 0 --batch 3
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import importlib.util
import time

try:
  from position_analysis.read_positions import read_positions, _find_pos_files
except Exception:
  base = pathlib.Path(__file__).resolve().parent
  rp = base / 'read_positions.py'
  if rp.exists():
    spec = importlib.util.spec_from_file_location('position_analysis.read_positions', str(rp))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    read_positions = mod.read_positions
    _find_pos_files = mod._find_pos_files
  else:
    from read_positions import read_positions, _find_pos_files


def _positions_to_step_batch(positions: np.ndarray) -> np.ndarray:
  """Convert saved positions into shape (nsteps, nbatch, nelec, 3)."""
  arr = np.asarray(positions)

  if arr.ndim >= 1 and arr.shape[-1] % 3 == 0 and arr.shape[-1] != 3:
    nelec = arr.shape[-1] // 3
    arr = arr.reshape(*arr.shape[:-1], nelec, 3)

  if arr.ndim < 4 or arr.shape[-1] != 3:
    raise ValueError('positions do not have a final coordinate dimension of size 3')

  if arr.ndim == 5:
    # (nproc, nsteps, batch_local, nelec, 3) -> (nsteps, nproc*batch_local, nelec, 3)
    nproc, nsteps, batch_local, nelec, _ = arr.shape
    return arr.transpose(1, 0, 2, 3, 4).reshape(nsteps, nproc * batch_local, nelec, 3)

  if arr.ndim == 4:
    # (nsteps, batch, nelec, 3)
    return arr

  # Fallback: collapse all leading dims except the last two into a sample axis.
  lead = arr.shape[:-2]
  samples = int(np.prod(lead))
  nelec = arr.shape[-2]
  return arr.reshape(samples, 1, nelec, 3)


def collect_particle_points(folder: str, particle: int, batch_idx: int,
                            start: Optional[int] = None, end: Optional[int] = None,
                            proc_index: Optional[int] = None) -> np.ndarray:
  """Collect (N,3) Cartesian points for `particle` in `batch_idx` across files."""
  files = _find_pos_files(folder)
  if not files:
    raise FileNotFoundError(f'No pos*_all.h5 files found in {folder!r}')
  if proc_index is not None:
    target_name = f'pos{int(proc_index)}_all.h5'
    files = [f for f in files if os.path.basename(f) == target_name]
    if not files:
      raise FileNotFoundError(
          f'No file named {target_name!r} found in {folder!r}')
  pts_list = []
  for f in files:
    positions = read_positions(f, start=start, end=end)
    steps = _positions_to_step_batch(positions)
    # steps: (nsteps, nbatch, nelec, 3)
    nsteps, nbatch, nelec, _ = steps.shape
    if batch_idx < 0 or batch_idx >= nbatch:
      raise ValueError(f'batch_idx {batch_idx} out of range for file {f} (nbatch={nbatch})')
    if particle < 0 or particle >= nelec:
      raise ValueError(f'particle index {particle} out of range for file {f} (nelec={nelec})')
    # select this particle across all steps for this batch
    part_pts = steps[:, batch_idx, particle, :]
    pts_list.append(np.asarray(part_pts))
  if not pts_list:
    return np.zeros((0, 3), dtype=float)
  return np.vstack(pts_list)


def _get_lattice_matrix(lattice_type: str, a: float) -> np.ndarray:
  t = lattice_type.lower()
  if t not in ('sc', 'bcc', 'fcc'):
    raise ValueError(f'Unsupported lattice type: {lattice_type}')
  return np.diag(np.array([a, a, a], dtype=float))


def _get_primitive_matrix(lattice_type: str, a: float) -> np.ndarray:
  t = lattice_type.lower()
  if t == 'sc':
    return np.diag(np.array([a, a, a], dtype=float))
  if t == 'bcc':
    return (a / 2.0) * np.array([[-1.0, 1.0, 1.0],
                                 [1.0, -1.0, 1.0],
                                 [1.0, 1.0, -1.0]], dtype=float).T
  if t == 'fcc':
    return (a / 2.0) * np.array([[0.0, 1.0, 1.0],
                                 [1.0, 0.0, 1.0],
                                 [1.0, 1.0, 0.0]], dtype=float).T
  raise ValueError(f'Unsupported lattice type: {lattice_type}')


def _map_points_into_cell(r: np.ndarray, lattice_type: str, lattice_constant: float,
                          tile: Optional[int] = None) -> np.ndarray:
  """Map Cartesian points `r` into the conventional cell, optionally tiling.

  Returns array of points inside the conventional cell (N,3).
  """
  if r.size == 0:
    return r.reshape(0, 3)
  L = _get_lattice_matrix(lattice_type, float(lattice_constant))
  if tile is None or tile <= 1:
    invL = np.linalg.inv(L)
    u = (invL @ r.T).T
    u = np.mod(u, 1.0)
    return (L @ u.T).T

  # tile > 1: generate candidate images and filter those inside conventional cell
  P = _get_primitive_matrix(lattice_type, float(lattice_constant))
  invP = np.linalg.inv(P)
  M = np.linalg.inv(L) @ P
  n = int(tile)
  start = -(n // 2)
  shifts = np.array([[i, j, k] for i in np.arange(start, start + n)
                     for j in np.arange(start, start + n)
                     for k in np.arange(start, start + n)], dtype=float)

  # fractional coords in primitive cell
  u_prim = (invP @ r.T).T
  u_prim = np.mod(u_prim, 1.0)
  tiled = u_prim[:, None, :] + shifts[None, :, :]
  flat = tiled.reshape(-1, 3)
  u_conv_flat = (M @ flat.T).T
  mask = np.all((u_conv_flat >= 0.0) & (u_conv_flat < 1.0), axis=1)
  valid = u_conv_flat[mask]
  if valid.shape[0] == 0:
    return np.zeros((0, 3), dtype=float)
  coords = (L @ valid.T).T
  return coords


def plot_3d_scatter(points: np.ndarray, out: Optional[str] = None,
                    max_points: Optional[int] = None, marker_size: float = 1.0) -> None:
  if points.size == 0:
    raise ValueError('No points to plot')
  N = points.shape[0]
  if max_points is not None and N > max_points:
    idx = np.linspace(0, N - 1, max_points).astype(int)
    samp = points[idx]
  else:
    samp = points

  fig = plt.figure(figsize=(6, 6))
  ax = fig.add_subplot(111, projection='3d')
  sc = ax.scatter(samp[:, 0], samp[:, 1], samp[:, 2], c=np.arange(samp.shape[0]),
                  cmap='viridis', s=marker_size)
  # Force a cubic plotting range so x/y/z share the same scale and box size.
  xyz_min = float(np.min(points))
  xyz_max = float(np.max(points))
  if xyz_max == xyz_min:
    delta = 1e-6
    xyz_min -= delta
    xyz_max += delta
  ax.set_xlim(xyz_min, xyz_max)
  ax.set_ylim(xyz_min, xyz_max)
  ax.set_zlim(xyz_min, xyz_max)
  if hasattr(ax, 'set_box_aspect'):
    ax.set_box_aspect((1.0, 1.0, 1.0))
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.colorbar(sc, label='step')
  if out is None:
    out = os.path.join(os.getcwd(), 'scatter3d.png')
  plt.savefig(out, dpi=200, bbox_inches='tight')
  np.savez_compressed(os.path.splitext(out)[0] + '.npz', points=points)
  print(f'Saved 3D scatter to {out}')



def main():
  p = argparse.ArgumentParser(description='Plot 3D scatter for a particle in a batch')
  p.add_argument('folder', type=str, help='folder containing pos*_all.h5')
  p.add_argument('--electrons', type=int, required=True, help='electron/particle index')
  p.add_argument('--batch-indices', type=int, required=True, help='batch index')
  p.add_argument('--proc-index', type=int, default=None,
                 help='only read one process file: pos{proc_index}_all.h5')
  p.add_argument('--start', type=int, default=None, help='start step (inclusive)')
  p.add_argument('--end', type=int, default=None, help='end step (inclusive)')
  p.add_argument('--out', type=str, default=None, help='output png path')
  p.add_argument('--max-points', type=int, default=200000, help='max points to plot (subsample)')
  p.add_argument('--lattice-type', type=str, default='sc', choices=['sc', 'bcc', 'fcc'],
                 help='lattice type for folding into primitive cell (sc, bcc, fcc)')
  p.add_argument('--lattice-constant', type=float, default=1.0,
                 help='lattice constant a (same units as positions)')
  p.add_argument('--tile', type=int, default=None,
                 help='supercell tiling factor per axis (default: 1 for sc, 3 for bcc/fcc)')
  args = p.parse_args()

  pts = collect_particle_points(
      args.folder,
      args.electrons,
      args.batch_indices,
      start=args.start,
      end=args.end,
      proc_index=args.proc_index)
  print(f'Collected {pts.shape[0]} points for particle {args.electrons} batch {args.batch_indices}')
  # map / tile points into cell
  if args.tile is None:
    # default tile selection to match density.py behaviour
    tile = 1 if args.lattice_type.lower() == 'sc' else 3
  else:
    tile = args.tile
  pts_cell = _map_points_into_cell(pts, args.lattice_type, args.lattice_constant, tile)
  print(f'{pts_cell.shape[0]} points after folding/tiling into cell')
  plot_3d_scatter(pts_cell, out=args.out, max_points=args.max_points)


if __name__ == '__main__':
  main()
