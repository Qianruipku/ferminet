"""Compute and plot a 2D density by compressing one axis of positions.

This module uses `read_positions(folder)` to load per-process saved
positions (all steps), compresses the requested axis (e.g. electron index) and creates a
2D histogram (density) of two coordinate axes (default: x and y).

Example:
  python -m position_analysis.density positions --proj 0,1 --bins 300
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import lax
import time
import importlib.util
import pathlib

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


# Note: positions are converted to (samples, Na, 3) by _positions_to_samples
# and then folded into the primitive cell. compute_hist2d operates on the
# flattened Cartesian points directly.


def compute_hist2d(points: np.ndarray, coord_axes: Sequence[int] = (0, 1),
                   bins: int = 200, range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                   normalize: bool = True):
  """Compute 2D histogram from `points`.

  `points` is expected shape (M, coord_dim) where coord_dim >= 2.
  Returns histogram (H, xedges, yedges). If `normalize` is True, H is
  normalized to sum to 1.
  """
  x_idx, y_idx = coord_axes
  xy = points[:, [x_idx, y_idx]]
  H, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=range)
  H = H.T
  if normalize:
    s = H.sum()
    if s > 0:
      H = H / float(s)
  return H, xedges, yedges


def plot_density(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray,
                 out: Optional[str] = None, cmap: str = 'viridis') -> None:
  extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
  plt.figure(figsize=(6, 5))
  plt.imshow(H, origin='lower', extent=extent, aspect='auto', cmap=cmap)
  plt.colorbar(label='counts')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.savefig(out, dpi=200, bbox_inches='tight')
  data_out = f'{os.path.splitext(out)[0]}.npz'
  np.savez_compressed(data_out, H=H, xedges=xedges, yedges=yedges)
  print(f'Saved density plot to {out}')
  print(f'Saved density data to {data_out}')


# Module-level lattice helpers (moved out of read_and_plot so tests can import)
def _get_lattice_matrix(lattice_type: str, a: float) -> jnp.ndarray:
  t = lattice_type.lower()
  if t not in ('sc', 'bcc', 'fcc'):
    raise ValueError(f'Unsupported lattice type: {lattice_type}')
  return jnp.diag(jnp.array([a, a, a], dtype=float))


def _get_primitive_matrix(lattice_type: str, a: float) -> jnp.ndarray:
  t = lattice_type.lower()
  if t == 'sc':
    return jnp.diag(jnp.array([a, a, a], dtype=float))
  if t == 'bcc':
    return (a / 2.0) * jnp.array([[-1.0, 1.0, 1.0],
                                 [1.0, -1.0, 1.0],
                                 [1.0, 1.0, -1.0]], dtype=float).T
  if t == 'fcc':
    return (a / 2.0) * jnp.array([[0.0, 1.0, 1.0],
                                 [1.0, 0.0, 1.0],
                                 [1.0, 1.0, 0.0]], dtype=float).T
  raise ValueError(f'Unsupported lattice type: {lattice_type}')

def read_and_plot(folder: str,
                  coord_axes: Sequence[int] = (0, 1), bins: int = 200,
                  out: Optional[str] = None,
                  electrons: Optional[Sequence[int]] = None,
                  lattice_type: str = 'sc', lattice_constant: float = 1.0,
                  tile: Optional[int] = None,
                  max_chunk: int = 200000) -> None:
  """Read positions and plot density for all processes.

  Args:
    This function always processes per-process files one-by-one and
    accumulates the histogram to avoid loading all positions at once.
  """
  if tile is None:
    if lattice_type.lower() == 'sc':
      tile = 1
    else:
      tile = 3

  def _positions_to_samples(positions: np.ndarray) -> np.ndarray:
    """Convert saved positions into shape (samples, Na, 3).

    Handles inputs where the last axis is either 3 (already (..., Na, 3))
    or flattened Na*3. Collapses all leading axes except the last two into
    a single samples axis.
    """
    arr = np.asarray(positions)
    # If last dim is flattened coords
    if arr.ndim >= 1 and arr.shape[-1] % 3 == 0 and arr.shape[-1] != 3:
      Na = arr.shape[-1] // 3
      arr = arr.reshape(*arr.shape[:-1], Na, 3)
    # Now ensure last dim is 3
    if arr.ndim < 2 or arr.shape[-1] != 3:
      raise ValueError('positions do not have a final coordinate dimension of size 3')
    # Collapse leading dims into samples; assume electron axis is second-last
    # after reshaping coords to Na x 3 when necessary.
    lead = arr.shape[:-2]
    samples = int(np.prod(lead))
    Na = arr.shape[-2]
    reshaped = arr.reshape(samples, Na, 3)
    return reshaped
  # lattice matrix used to fold positions into the primitive cell
  L = _get_lattice_matrix(lattice_type, float(lattice_constant))
  cell_sizes = np.array([L[0, 0], L[1, 1], L[2, 2]])

  assert tile and tile >= 1, f'Invalid tile value: {tile}'

  def _accumulate_hist_jax(r: np.ndarray, P: np.ndarray, L: np.ndarray, n: int,
                           coord_axes: Sequence[int], bins: int) -> np.ndarray:
    """Accumulate histogram using JAX, streaming over samples with lax.fori_loop.

    Returns numpy array H shape (bins,bins).
    """
    # Vectorized implementation: compute all shifts for all samples, filter
    # valid images, compute bin indices and use bincount to accumulate.
    start_time = time.perf_counter()
    rj = jnp.array(r)
    Pj = jnp.array(P)
    Lj = jnp.array(L)
    invP = jnp.linalg.inv(Pj)
    M = jnp.linalg.inv(Lj) @ Pj

    # prepare shifts
    start = -(n // 2)
    shifts_np = jnp.array([[i, j, k] for i in np.arange(start, start + n)
                           for j in np.arange(start, start + n)
                           for k in np.arange(start, start + n)], dtype=float)
    shifts = jnp.array(shifts_np)

    # fractional coords in primitive cell, folded into [0,1)
    u_prim = (invP @ rj.T).T
    u_prim = jnp.mod(u_prim, 1.0)

    # tiled candidate positions: (N, S, 3)
    tiled = u_prim[:, None, :] + shifts[None, :, :]
    # flatten (N*S, 3) and map to conventional fractional coords
    flat = tiled.reshape(-1, 3)
    u_conv_flat = (M @ flat.T).T
    # mask valid images in conventional cell
    mask = jnp.all((u_conv_flat >= 0.0) & (u_conv_flat < 1.0), axis=1)
    valid = u_conv_flat[mask]
    if valid.shape[0] == 0:
      elapsed = time.perf_counter() - start_time
      print(f'_accumulate_hist_jax: {elapsed:.3f}s N={r.shape[0]} shifts={shifts.shape[0]} valid=0')
      return np.zeros((bins, bins), dtype=int)

    # map to Cartesian coords and compute histogram indices
    coords = (Lj @ valid.T).T
    a_x = float(L[coord_axes[0], coord_axes[0]])
    a_y = float(L[coord_axes[1], coord_axes[1]])
    x = coords[:, coord_axes[0]]
    y = coords[:, coord_axes[1]]
    ix = jnp.floor((x / a_x) * bins).astype(jnp.int32)
    iy = jnp.floor((y / a_y) * bins).astype(jnp.int32)
    ix = jnp.clip(ix, 0, bins - 1)
    iy = jnp.clip(iy, 0, bins - 1)

    linear = ix * bins + iy
    counts = jnp.bincount(linear, length=bins * bins)
    H = counts.reshape((bins, bins)).astype(jnp.int32)
    elapsed = time.perf_counter() - start_time
    print(f'_accumulate_hist_jax: {elapsed:.3f}s N={r.shape[0]} shifts={shifts.shape[0]} valid={valid.shape[0]}')
    return np.array(H)

  def _accumulate_hist_jax_chunked(r: np.ndarray, P: np.ndarray, L: np.ndarray, n: int,
                                   coord_axes: Sequence[int], bins: int,
                                   max_chunk: int = 200000) -> np.ndarray:
    """Chunked wrapper around `_accumulate_hist_jax` to avoid memory spikes.

    Splits the first axis of `r` into pieces of at most `max_chunk` points
    and accumulates the histogram by calling `_accumulate_hist_jax` for each
    piece.
    """
    N = int(r.shape[0]) if hasattr(r, 'shape') and len(r.shape) >= 1 else 0
    if N == 0:
      return np.zeros((bins, bins), dtype=int)
    if N <= max_chunk:
      return _accumulate_hist_jax(r, P, L, n, coord_axes, bins)

    H_total = np.zeros((bins, bins), dtype=int)
    chunks = int(np.ceil(N / float(max_chunk)))
    for ci in range(chunks):
      s = ci * max_chunk
      e = min((ci + 1) * max_chunk, N)
      H_part = _accumulate_hist_jax(r[s:e], P, L, n, coord_axes, bins)
      H_total += H_part
      print(f'  chunk {ci+1}/{chunks} processed: points={e-s} cumulative_sum={int(H_total.sum())}')
    return H_total

  def _compute_pts_from_r(r: jnp.ndarray) -> np.ndarray:
    """Given Cartesian points r (N,3), return Cartesian points to histogram.

    This encapsulates the tile vs fold logic used in both single-file and
    per-file processing paths.
    """
    invL = jnp.linalg.inv(L)
    u = (invL @ r.T).T
    u = jnp.mod(u, 1.0)
    return np.array((L @ u.T).T)

  # Stream: get files via read_positions' helper and process one-by-one
  files = _find_pos_files(folder)
  if not files:
    raise FileNotFoundError(f'No pos*_all.h5 files found in {folder!r}')
  H = None
  xedges = yedges = None
  print(f'Found {len(files)} pos*_all.h5 files in {folder!r}')
  for idx, f in enumerate(files):
    print(f'Processing file {idx+1}/{len(files)}: {f}')
    positions = read_positions(f)
    print(f'  loaded positions shape: {positions.shape}')
    samples = _positions_to_samples(positions)
    print(f'  converted to samples shape: {samples.shape}')
    if electrons is not None:
      samples = samples[:, electrons, :]
    # prepare Cartesian points and compute points for histogram
    r = samples.reshape(-1, 3)
    if tile and tile > 1:
      P = _get_primitive_matrix(lattice_type, float(lattice_constant))
      H_i = _accumulate_hist_jax_chunked(r, P, L, int(tile), coord_axes, bins,
                                        max_chunk=max_chunk)
      
      xe = np.linspace(0.0, float(cell_sizes[coord_axes[0]]), bins + 1)
      ye = np.linspace(0.0, float(cell_sizes[coord_axes[1]]), bins + 1)
    else:
      pts = _compute_pts_from_r(r)
      cell_range = ((0.0, float(cell_sizes[coord_axes[0]])),
                    (0.0, float(cell_sizes[coord_axes[1]])))
      H_i, xe, ye = compute_hist2d(pts, coord_axes=coord_axes, bins=bins, range=cell_range, normalize=False)
    
    print(f'  histogram computed (tile) for file {idx+1}: sum={int(H_i.sum())}')
    if H is None:
      H = H_i
      xedges = xe
      yedges = ye
    else:
      H += H_i

  s = H.sum()
  if s > 0:
    dx = float(xedges[1] - xedges[0])
    dy = float(yedges[1] - yedges[0])
    area = dx * dy
    H = H / (float(s) * area)

  if out is None:
    out = os.path.join(folder, 'density_all_steps.png')
  plot_density(H, xedges, yedges, out=out)


def main():
  p = argparse.ArgumentParser(description='Plot 2D density for saved steps')
  p.add_argument('folder', type=str, help='folder containing pos*_all.h5')
  # positions are assumed to be reshaped to (-1, Na, 3)
  p.add_argument('--proj', type=str, default='0,1',
                 help='projection coordinate indices (comma separated), e.g. "0,1" or "1,2"')
  p.add_argument('--bins', type=int, default=200, help='histogram bins')
  # `separate` option removed; streaming per-process processing is always used.
  p.add_argument('--electrons', type=str, default=None,
                 help='comma-separated electron indices or ranges (e.g. 0,2,5-7)')
  p.add_argument('--out', type=str, default=None, help='output png path')
  p.add_argument('--lattice-type', type=str, default='sc', choices=['sc', 'bcc', 'fcc'],
                 help='lattice type for folding into primitive cell (sc, bcc, fcc)')
  p.add_argument('--lattice-constant', type=float, default=1.0,
                 help='lattice constant a (same units as positions)')
  p.add_argument('--tile', type=int, default=None,
                 help='supercell tiling factor per axis (default: 1 for sc, 3 for bcc/fcc)')
  p.add_argument('--max-chunk', type=int, default=200000,
                 help='maximum number of points to process per chunk to avoid OOM')
  args = p.parse_args()
  def _parse_electrons(s: Optional[str]):
    if s is None:
      return None
    parts = []
    for token in s.split(','):
      token = token.strip()
      if not token:
        continue
      if '-' in token:
        a, b = token.split('-', 1)
        parts.extend(list(range(int(a), int(b) + 1)))
      else:
        parts.append(int(token))
    return parts

  electrons = _parse_electrons(args.electrons)
  def _parse_proj(s: str):
    toks = [t.strip() for t in s.split(',') if t.strip()]
    if len(toks) != 2:
      raise ValueError('proj must be two comma-separated indices')
    return (int(toks[0]), int(toks[1]))
  coord_axes = _parse_proj(args.proj)
  read_and_plot(args.folder,
                coord_axes=coord_axes, bins=args.bins, out=args.out,
                electrons=electrons,
                lattice_type=args.lattice_type, lattice_constant=args.lattice_constant,
                tile=args.tile, max_chunk=args.max_chunk)


if __name__ == '__main__':
  main()
