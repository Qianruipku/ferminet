#!/usr/bin/env python3
"""Utility: read positions saved by the training checkpoint code.

This module provides a single function `read_positions(step, folder)` that
returns a stacked numpy array of per-process positions for the requested
step. It intentionally does not include a CLI entry; use
`position_analysis.density` to run analyses/plots that call this function.
"""
from __future__ import annotations

import glob
import os
import re
from typing import List, Optional

import h5py
import numpy as np
import jax.numpy as jnp

__all__ = ["read_positions"]


def _find_pos_files(folder: str) -> List[str]:
  pattern = os.path.join(folder, 'pos*_all.h5')
  files = glob.glob(pattern)
  return files


def _proc_key(fname: str) -> int:
  b = os.path.basename(fname)
  m = re.match(r'pos(\d+)_all\.h5$', b)
  if m:
    return int(m.group(1))
  return 0


def read_positions(path: str, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
  """Return stacked positions for steps from an HDF5 file or folder.

  By default (when `start` and `end` are None) this function reads all
  available per-step datasets. If `path` is a single file the return
  shape is `(1, num_steps, ...)`. If `path` is a directory of
  per-process files the return shape is `(num_processes, num_steps, ...)`.
  All files in a directory are required to have the same set of
  `step_######` datasets.

  If `start` and/or `end` are provided they are interpreted as step numbers
  (the integer encoded in dataset names like `step_000123`) when `step_*`
  datasets exist; datasets are filtered to those with `start <= step <= end`.
  For legacy `positions` datasets (axis 0 = step index) `start`/`end` are
  treated as array indices and used to slice that axis (`arr[start:end+1]`).

  Args:
    path: path to a single HDF5 file or a directory containing per-process
      HDF5 files.

  Returns:
    ndarray with shape `(num_processes, num_steps, ...)` containing the
    positions stacked by process and step.
  """

  if os.path.isfile(path):
    fname = path
    with h5py.File(fname, 'r') as hf:
      # Expect new-style `positions` dataset with axis 0 == step.
      if 'positions' not in hf:
        raise RuntimeError(f'No positions dataset in {fname}')
      d = hf['positions']
      if d.ndim < 1:
        raise RuntimeError(f'positions dataset in {fname} has unexpected shape')
      arr_full = d[()]
      if start is None and end is None:
        arr = arr_full
      else:
        s = start or 0
        e = end if end is not None else (arr_full.shape[0] - 1)
        if s < 0 or e < s or e >= arr_full.shape[0]:
          raise RuntimeError(f'positions slice {s}..{e} out of range for {fname}')
        arr = arr_full[s:e+1]
    # add a process axis for single-file input
    return jnp.expand_dims(jnp.asarray(np.asarray(arr)), axis=0)

  # Treat path as folder
  files = _find_pos_files(path)
  if not files:
    raise FileNotFoundError(f'No pos*_all.h5 files found in {path!r}')

  files = sorted(files, key=_proc_key)
  parts = []
  # Read `positions` from each per-process file; treat start/end as indices
  for f in files:
    with h5py.File(f, 'r') as hf:
      if 'positions' not in hf:
        raise RuntimeError(f'No positions dataset in {f}')
      d = hf['positions']
      if d.ndim < 1:
        raise RuntimeError(f'positions dataset in {f} has unexpected shape')
      arr_full = d[()]
      if start is None and end is None:
        parts.append(arr_full)
      else:
        s = start or 0
        e = end if end is not None else (arr_full.shape[0] - 1)
        if s < 0 or e < s or e >= arr_full.shape[0]:
          raise RuntimeError(f'positions slice {s}..{e} out of range for {f}')
        parts.append(arr_full[s:e+1])
  # parts: list of arrays with shape (num_steps, ...), stack into (num_processes, num_steps, ...)
  return jnp.stack([jnp.asarray(np.asarray(p)) for p in parts], axis=0)

