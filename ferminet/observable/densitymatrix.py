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
    raw_dirs = jax.random.normal(subkey_dirs, (self.nbins, n_sample, 3))
    norms = jnp.linalg.norm(raw_dirs, axis=2, keepdims=True)
    dirs = raw_dirs / (norms + 1e-12)
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

      # iterate over random pairs using lax.fori_loop instead of Python loops
      def pair_body(p, acc_in):
        i_idx = ia_idx[p]
        j_idx = ib_idx[p]

        def body_bin(b, acc_bin):
          def body_dir(k, acc_dir):
            shift = shift_vec[b, k]
            # build shifted positions for all walkers
            shift_b = jnp.broadcast_to(shift, (nwalker_per_device, 3))
            pos_mod = pos_expand.at[:, i_idx, :].add(shift_b)
            pos_mod = pos_mod.at[:, j_idx, :].add(shift_b)
            pos_mod_flat = jnp.reshape(pos_mod, (nwalker_per_device, -1))
            numer_signs, numer_logs = batch_network(params, pos_mod_flat, spins_device, atoms_device, charges_device)
            contrib = numer_signs * denom_signs * jnp.exp(numer_logs - denom_logs)
            s = jnp.sum(contrib)
            return acc_dir.at[b].add(s)
          acc_bin = jax.lax.fori_loop(0, n_sample, body_dir, acc_bin)
          return acc_bin

        acc_out = jax.lax.fori_loop(0, self.nbins, body_bin, acc_in)
        return acc_out

      acc = jax.lax.fori_loop(0, npairs, pair_body, acc)

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
