"""Simplified radial two-body density (rho(r)) and condensate fraction.

This module provides a compact `RadialTwoBodyDensity` class that computes the
rotationally- and translationally-averaged radial two-body density `rho(r)` for
a chosen pair of spin species, and estimates a condensate fraction from the
large-r plateau. It intentionally keeps a small API and does not use
inheritance.
"""

from dataclasses import dataclass, field
from typing import Sequence, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import chex
import os

from ferminet import network_blocks
from ferminet.utils.min_distance import min_image_distance_triclinic, Lattice
from ferminet import constants


@dataclass
class RadialTwoBodyDensity:
  """Compute radial two-body density rho(r) and condensate fraction.

  Attributes:
    lattice: 3x3 lattice matrix (for PBC). If None, PBC are not applied.
    nspins: sequence with number of particles per spin/species, e.g. (N_up, N_down)
    pair_type: which species pair to consider, (ia, ib) as indices into nspins
    nbins: number of radial bins
    rmax: maximum radius to consider
    ndim: spatial dimensionality (only 3 supported)
    key: PRNG key (unused but kept for API compatibility)
    compute_condensate_fraction: if True, return an estimate of condensate
      fraction as the mean of rho(r) over the largest 10% of r-bins scaled
      by Volume^2 / min(nspins) (matches previous convention).
  """
  lattice: jnp.ndarray
  nspins: Sequence[int]
  pair_type: Tuple[int, int] = (0, 1)
  nbins: int = 50
  rmax: float = 10.0
  ndim: int = 3
  key: chex.PRNGKey = field(default_factory=lambda: jax.random.PRNGKey(0))
  compute_condensate_fraction: bool = True
  signed_network: Optional[object] = None  # networks.FermiNetLike, set at runtime
  fraction_pairs: float = 0.05
  n_dirs: int = 1  # directions per radial bin (kept small for efficiency)
  save_freq: int = 1000  # save every N steps (only host 0)

  def __post_init__(self):
    if self.ndim != 3:
      raise ValueError('Only 3D supported for radial two-body density')
    if self.lattice is None:
      self.volume = 1.0
      self._lat = None
    else:
      self._lat = self.lattice
      self.volume = float(self._lat.volume)

    # radial grid
    self.bin_edges = jnp.linspace(0.0, float(self.rmax), int(self.nbins) + 1)
    self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
    # host-side running accumulators to avoid unbounded memory growth.
    # We maintain sums and counts; the average is written on `save()`.
    # pre-allocate running sum to avoid per-step None checks
    self._rho_sum = np.zeros(int(self.nbins), dtype=float)
    self._rho_count = 0
    self._condensate_sum = 0.0
    self._condensate_count = 0
    # Decide whether to precompute all pairs (full sampling) or sample each
    # call (partial sampling). If `fraction_pairs >= 1.0`, treat as full
    # sampling and precompute all ordered pairs (i!=j for same species).
    ia, ib = self.pair_type
    partitions = network_blocks.array_partitions(self.nspins)
    total_particles = int(sum(self.nspins))
    partitions_ext = [0] + [int(x) for x in partitions] + [total_particles]
    # validate pair_type indices against number of species
    nspecies = len(self.nspins)
    if not (0 <= ia < nspecies) or not (0 <= ib < nspecies):
      raise ValueError(
          f'Invalid pair_type {self.pair_type} for nspins {self.nspins}. '
          f'Indices must be in [0, {nspecies-1}].')
    self._precomputed_pairs = False
    if float(self.fraction_pairs) >= 1.0:
      start_a, end_a = partitions_ext[ia], partitions_ext[ia+1]
      start_b, end_b = partitions_ext[ib], partitions_ext[ib+1]
      if ia != ib:
        a_range = np.arange(start_a, end_a, dtype=np.int32)
        b_range = np.arange(start_b, end_b, dtype=np.int32)
        ia_idx = np.repeat(a_range, b_range.size)
        ib_idx = np.tile(b_range, a_range.size)
      else:
        ia_list = []
        ib_list = []
        for i in range(start_a, end_a):
          for j in range(start_a, end_a):
            if i != j:
              ia_list.append(i)
              ib_list.append(j)
        ia_idx = np.array(ia_list, dtype=np.int32)
        ib_idx = np.array(ib_list, dtype=np.int32)
      # store as jnp arrays for use in lax loops
      self.ia_idx = jnp.asarray(ia_idx)
      self.ib_idx = jnp.asarray(ib_idx)
      self.npairs = int(self.ia_idx.shape[0])
      self._precomputed_pairs = True

  def __call__(self, params, data) -> None:
    """Compute rho(r) and optional condensate fraction.

    Args:
      params: network parameters (unused here, kept for uniform observable API)
      data: `networks.FermiNetData` with `positions` shaped (..., 3N)
    Side effects:
      Accumulates the computed `rho_r` and optional `condensate_fraction` into
      the instance running sums (`_rho_sum`, `_condensate_sum`). Use `save()`
      to persist averaged results to disk. The function does not return a
      value.
    """
    # Detect device axis (common pattern: (n_devices, nwalkers_per_device, ...))
    positions = data.positions

    # Expect device axis to be present: per-process shape should begin with
    # (nlocal_devices, nbatch_local, ...). Enforce local-device-first layout.
    if not (positions.ndim >= 3 and positions.shape[0] == jax.local_device_count()):
      raise ValueError(
          'Expected `data.positions` to have local device axis first with '
          'size jax.local_device_count()')

    # prepare pair indices: either use precomputed all-pairs or sample
    if self._precomputed_pairs:
      ia_idx = self.ia_idx
      ib_idx = self.ib_idx
      npairs = self.npairs
    else:
      ia, ib = self.pair_type
      partitions = network_blocks.array_partitions(self.nspins)
      total_particles = int(sum(self.nspins))
      parts_ext = [0] + [int(x) for x in partitions] + [total_particles]
      Na = parts_ext[ia+1] - parts_ext[ia]
      Nb = parts_ext[ib+1] - parts_ext[ib]
      if ia != ib:
        total_pairs = Na * Nb
      else:
        total_pairs = Na * (Na - 1)
      npairs = max(1, int(self.fraction_pairs * total_pairs))
      key, sk1 = jax.random.split(self.key)
      if ia != ib:
        ia_idx = jax.random.randint(sk1, (npairs,), parts_ext[ia], parts_ext[ia+1])
        key, sk2 = jax.random.split(key)
        ib_idx = jax.random.randint(sk2, (npairs,), parts_ext[ib], parts_ext[ib+1])
      else:
        key, sk1 = jax.random.split(key)
        ia_idx = jax.random.randint(sk1, (npairs,), parts_ext[ia], parts_ext[ia+1])
        key, sk2 = jax.random.split(key)
        j_rel = jax.random.randint(sk2, (npairs,), 0, Na - 1)
        i_rel = ia_idx - partitions[ia]
        j_rel = j_rel + (j_rel >= i_rel).astype(jnp.int32)
        ib_idx = (j_rel + parts_ext[ia])
      ia_idx = jnp.asarray(ia_idx)
      ib_idx = jnp.asarray(ib_idx)
      self.key = key

    # sample random directions per bin for rotational averaging. Create
    # shift_vec of shape (nbins, n_sample, 3) = unit_dirs * bin_center
    n_sample = max(1, int(self.n_dirs))
    key, subkey_dirs = jax.random.split(self.key)

    def _base_unit_points(n):
      # explicit small symmetric sets
      if n == 1:
        return jnp.array([[0.0, 0.0, 1.0]])
      if n == 2:
        return jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
      if n == 6:
        # Octahedron vertices
        return jnp.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ])
      if n == 12:
        spherical_points = [[0, 0], [np.pi, 0]]
        spherical_points += [[np.arctan(2), 2 * np.pi * i / 5] for i in range(1, 6)]
        spherical_points += [
            [np.pi - np.arctan(2), np.pi / 5 * (2 * i - 11)] for i in range(6, 11)
        ]
        theta, phi = zip(*spherical_points)
        pts = jnp.stack([
            jnp.asarray(np.cos(phi) * np.sin(theta)),
            jnp.asarray(np.sin(phi) * np.sin(theta)),
            jnp.asarray(np.cos(theta)),
        ], axis=1)
        return pts
      else:
        raise ValueError(
          f'Unsupported `n_dirs`={n}. Supported explicit values: 1,2,6,12.')

    base_points = _base_unit_points(n_sample)  # (n_sample, 3)

    # rotate base symmetric points to a random axis per-bin
    keys_bins = jax.random.split(subkey_dirs, self.nbins)

    def _rotate_to_random_axis(key):
      z = jax.random.normal(key, (3,))
      ep3 = z / (jnp.linalg.norm(z) + 1e-12)
      # choose a stable 'up' vector
      up = jnp.array([0.0, 0.0, 1.0])
      up = jnp.where(jnp.abs(ep3[2]) < 0.9, up, jnp.array([0.0, 1.0, 0.0]))
      ep1 = jnp.cross(up, ep3)
      ep1 = ep1 / (jnp.linalg.norm(ep1) + 1e-12)
      ep2 = jnp.cross(ep3, ep1)
      R = jnp.stack([ep1, ep2, ep3], axis=1)  # columns are basis vectors
      aligned = (R @ base_points.T).T
      return aligned  # (n_sample, 3)

    dirs = jax.vmap(_rotate_to_random_axis)(keys_bins)  # (nbins, n_sample, 3)
    shift_vec = dirs * self.bin_centers[:, None, None]
    # update stored key
    self.key = key

    # batch network for evaluating sign and log for many positions
    if self.signed_network is None:
      raise ValueError('signed_network must be provided for two-body estimator')
    batch_network = jax.vmap(self.signed_network, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0))

    def per_device_hist(pos_device, spins_device, atoms_device, charges_device, params):
      # pos_expand: (nwalkers_per_device, nparticles, 3)
      nwalker_per_device = pos_device.shape[0]
      pos_expand = jnp.reshape(pos_device, (nwalker_per_device, -1, 3))
      pos_flat = jnp.reshape(pos_device, (nwalker_per_device, -1))
      denom_signs, denom_logs = batch_network(params, pos_flat, spins_device, atoms_device, charges_device)

      # accumulator for bins
      acc = jnp.zeros((self.nbins,))

      # Vectorized pair loop using `lax.scan` + batch `vmap` over shifts.
      # For each pair, build all (nbins * n_sample) shifts, compute the
      # modified positions in one batch, call the network once per-shift
      # (and internally batched over walkers), then aggregate back to bins.
      def pair_step(acc_in, p_idx):
        i_idx = ia_idx[p_idx]
        j_idx = ib_idx[p_idx]

        # mask over particle index where the shift should be added
        nparticles = pos_expand.shape[1]
        mask = jnp.zeros((nparticles, 3))
        mask = mask.at[i_idx].add(1.0)
        mask = mask.at[j_idx].add(1.0)

        nshifts = self.nbins * n_sample
        shifts = jnp.reshape(shift_vec, (nshifts, 3))

        # base positions: (1, nwalker, nparticles, 3)
        base = pos_expand[None, ...]
        # pos_mod: (nshifts, nwalker, nparticles, 3)
        pos_mod = base + shifts[:, None, None, :] * mask[None, None, :, :]
        pos_mod_flat = jnp.reshape(pos_mod, (nshifts, nwalker_per_device, -1))

        # compute per-shift contributions using a fori_loop to avoid allocating
        # large intermediate arrays when vmap would OOM. This is memory-friendly
        # and preferred for small-ish `n_dirs`.
        def shift_body(s, acc_s):
          pos_s = pos_mod_flat[s]  # (nwalker, nparticles*3)
          numer_s, logs_s = batch_network(params, pos_s, spins_device, atoms_device, charges_device)
          contrib_s = numer_s * denom_signs * jnp.exp(logs_s - denom_logs)
          val = jnp.sum(contrib_s)
          return acc_s.at[s].set(val)

        s_per_shift = jax.lax.fori_loop(0, nshifts, shift_body, jnp.zeros((nshifts,), dtype=denom_logs.dtype))
        s_per_bin = jnp.reshape(s_per_shift, (self.nbins, n_sample))
        s_bin_sum = jnp.sum(s_per_bin, axis=1)  # (nbins,)

        acc_out = acc_in + s_bin_sum
        return acc_out, None

      acc, _ = jax.lax.scan(pair_step, acc, jnp.arange(npairs))

      # normalize accumulator: divide by (nwalker_per_device * npairs * shell_vol)
      shell_vol = 4.0 * jnp.pi / 3.0 * (self.bin_edges[1:]**3 - self.bin_edges[:-1]**3)
      rho_r_device = acc / (nwalker_per_device * npairs) / shell_vol
      # convert to density units consistent with previous convention
      return constants.pmean(rho_r_device)

    hist_all = constants.pmap(per_device_hist)(positions, data.spins, data.atoms, data.charges, params)
    rho_r = hist_all[0]

    out = {'r': self.bin_centers, 'rho_r': rho_r}

    if self.compute_condensate_fraction:
      # estimate condensate fraction from large-r plateau: mean over last 10% bins
      n_tail = max(1, int(0.1 * self.nbins))
      tail_mean = jnp.mean(rho_r[-n_tail:])
      cf = float(tail_mean * (self.volume**2) / float(min(self.nspins)))
      out['condensate_fraction'] = cf

    rho_np = np.asarray(out['rho_r'])
    self._rho_sum += rho_np
    self._rho_count += 1
    if 'condensate_fraction' in out:
      self._condensate_sum += float(out['condensate_fraction'])
      self._condensate_count += 1

    # do not return anything; results are stored in the instance history
    return None

  def save(self, ckpt_save_path: str,
           step: int,
           end_step: int):
    """Save accumulated history arrays to disk (only on host 0).

    Args:
      ckpt_save_path: directory path where to save the arrays.
    """
    if not ((step+1) % self.save_freq == 0 or (step+1) == end_step):
      return  # only save at specified frequency or at the end of training
    if jax.process_index() == 0:
      if self._rho_count > 0:
        if step+1 < end_step:
          id = (step + 1) // self.save_freq
          filename = os.path.join(ckpt_save_path, f'two_body_rho_r_{id}.txt')
        else:
          filename = os.path.join(ckpt_save_path, f'two_body_rho_r_final.txt')
        avg = np.asarray(self._rho_sum) / float(self._rho_count)
        # Save two-column text: r  rho_r, convenient for plotting.
        out = np.vstack([np.asarray(self.bin_centers), avg]).T
        np.savetxt(filename, out, fmt='%.6e', header='r rho_r')
      if self._condensate_count > 0:
        avgc = float(self._condensate_sum) / float(self._condensate_count)
        # Print condensate fraction to stdout instead of saving to file.
        print(f'Condensate fraction (avg over steps): {avgc:.6e}')
