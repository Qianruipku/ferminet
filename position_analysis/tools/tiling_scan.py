"""Scan tiling expansion factors for FCC and BCC lattices.

Usage:
  python -m position_analysis.tools.tiling_scan --max-tile 6 --samples 8192 --seed 0

Prints counts and ratios relative to tile=1 and shows theoretical determinant ratio.
"""
from __future__ import annotations

import argparse
import sys
import pathlib
import numpy as np
import jax
import jax.numpy as jnp

# Ensure repository root is on sys.path so `position_analysis` imports work
_root = pathlib.Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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

def _tile_via_fractional_clip(r: jnp.ndarray, P: jnp.ndarray, L: jnp.ndarray, n: int, lattice_type: str = 'sc') -> np.ndarray:
  """Tile using fractional coordinates (primitive -> conventional) and clip to conventional [0,1).

  Steps:
    - compute u_prim = invP @ r (fractional coords in primitive cell)
    - fold u_prim into [0,1)
    - generate integer shifts s in {0..n-1}^3 and form u_prim + s
    - map to conventional fractional coords u_conv = invL @ (P @ u_prim_shift) = (invL@P) @ u_prim_shift
    - keep those with 0<=u_conv<1 in all axes
    - return Cartesian coords r_out = L @ u_conv
  """
  if n <= 1:
    invP = jnp.linalg.inv(P)
    invL = jnp.linalg.inv(L)
    u_prim = (invP @ r.T).T
    u_prim = jnp.mod(u_prim, 1.0)
    M = invL @ P
    u_conv = (M @ u_prim.T).T
    mask = (u_conv[:, 0] >= 0.0) & (u_conv[:, 0] < 1.0) & \
           (u_conv[:, 1] >= 0.0) & (u_conv[:, 1] < 1.0) & \
           (u_conv[:, 2] >= 0.0) & (u_conv[:, 2] < 1.0)
    u_sel = u_conv[mask]
    return np.array((L @ u_sel.T).T)

  # general n>1
  invP = jnp.linalg.inv(P)
  invL = jnp.linalg.inv(L)
  u_prim = (invP @ r.T).T
  u_prim = jnp.mod(u_prim, 1.0)
  # always use centered shifts
  start = -(n // 2)
  idx_np = np.arange(start, start + n)

  candidate_shifts = np.array([[i, j, k] for i in idx_np for j in idx_np for k in idx_np], dtype=float)
  # use full candidate shifts (centered or 0..n-1) and let per-point filtering
  # decide which shifted images fall into the conventional cell
  shifts = jnp.array(candidate_shifts)
  tiled = (u_prim[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
  M = invL @ P
  u_conv = (M @ tiled.T).T
  mask = (u_conv[:, 0] >= 0.0) & (u_conv[:, 0] < 1.0) & \
         (u_conv[:, 1] >= 0.0) & (u_conv[:, 1] < 1.0) & \
         (u_conv[:, 2] >= 0.0) & (u_conv[:, 2] < 1.0)
  u_sel = u_conv[mask]
  return np.array((L @ u_sel.T).T)

def scan(lattice: str, samples: int, max_tile: int, seed: int):
    key = jax.random.PRNGKey(seed)
    u_prim = jax.random.uniform(key, shape=(samples, 3), minval=0.0, maxval=1.0)
    P = _get_primitive_matrix(lattice, 1.0)
    L = _get_lattice_matrix(lattice, 1.0)
    # Cartesian
    r = (P @ u_prim.T).T

    print(f"\nLattice: {lattice}")
    detP = float(np.linalg.det(np.asarray(P)))
    detL = float(np.linalg.det(np.asarray(L)))
    expected = detL / detP
    print(f"Theoretical det ratio (det(L)/det(P)) = {expected:.6f}")
    n0 = samples
    print(f"Samples: {n0}")
    print("tile  count    ratio")
    for n in range(1, max_tile + 1):
        pts = _tile_via_fractional_clip(r, P, L, n, lattice_type=lattice)
        cnt = int(np.asarray(pts).shape[0])
        ratio = cnt / n0
        print(f"{n:4d}  {cnt:8d}  {ratio:10.6f}")
        


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--max-tile', type=int, default=6)
    p.add_argument('--samples', type=int, default=8192)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    scan('sc', args.samples, args.max_tile, args.seed)
    scan('bcc', args.samples, args.max_tile, args.seed)
    scan('fcc', args.samples, args.max_tile, args.seed)
