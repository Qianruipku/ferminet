# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

import chex
from ferminet import constants
from ferminet import networks
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
from typing import Tuple
from ferminet.utils.min_distance import min_image_distance_triclinic, Lattice


def _harmonic_mean(x, atoms):
  """Calculates the harmonic mean of each electron distance to the nuclei.

  Args:
    x: electron positions. Shape (batch, nelectrons, 1, ndim). Note the third
      dimension is already expanded, which allows for avoiding additional
      reshapes in the MH algorithm.
    atoms: atom positions. Shape (natoms, ndim)

  Returns:
    Array of shape (batch, nelectrons, 1, 1), where the (i, j, 0, 0) element is
    the harmonic mean of the distance of the j-th electron of the i-th MCMC
    configuration to all atoms.
  """
  ae = x - atoms[None, ...]
  r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
  return 1.0 / jnp.mean(1.0 / r_ae, axis=-2, keepdims=True)


def _log_prob_gaussian(x, mu, sigma):
  """Calculates the log probability of Gaussian with diagonal covariance.

  Args:
    x: Positions. Shape (batch, nelectron, 1, ndim) - as used in mh_update.
    mu: means of Gaussian distribution. Same shape as or broadcastable to x.
    sigma: standard deviation of the distribution. Same shape as or
      broadcastable to x.

  Returns:
    Log probability of Gaussian distribution with shape as required for
    mh_update - (batch, nelectron, 1, 1).
  """
  numer = jnp.sum(-0.5 * ((x - mu)**2) / (sigma**2), axis=[1, 2, 3])
  denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
  return numer - denom


def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts, species_idx):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  if jnp.ndim(num_accepts) == 0:
    num_accepts += jnp.sum(cond)
  else:
    num_accepts = num_accepts.at[species_idx].add(jnp.sum(cond))
  return x_new, key, lp_new, num_accepts


def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev: jnp.ndarray,
    nspins: Tuple[int, ...],
    ndim: int,
    atoms=None,
    blocks=1,
    i=0,
    mix_width: float = 1.0,
    mix_prob: float = 0.0,
):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: If not None, atom positions. Shape (natoms, 3). If present, then the
      Metropolis-Hastings move proposals are drawn from a Gaussian distribution,
      N(0, (h_i stddev)^2), where h_i is the harmonic mean of distances between
      the i-th electron and the atoms, otherwise the move proposal drawn from
      N(0, stddev^2).
    ndim: dimensionality of system.
    blocks: Ignored.
    i: Ignored.

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks  # electron index ignored for all-electron moves
  if atoms is not None:
    raise NotImplementedError("Asymmetric moves not implemented")

  key, subkey = jax.random.split(key)
  start_idx = 0
  x1 = data.positions
  std_large = jnp.asarray(mix_width)
  p_large = mix_prob
  if jnp.ndim(stddev) == 0:
    # Avoid unnecessary uniform draw when mix is disabled (p_large == 0).
    if p_large == 0:
      key_noise = subkey
      sigma = stddev
    else:
      key_choice, key_noise = jax.random.split(subkey)
      use_large = jax.random.uniform(key_choice) < p_large
      sigma = jnp.where(use_large, std_large, stddev)
    x2 = x1 + sigma * jax.random.normal(key_noise, shape=x1.shape)  # proposal
    lp_2 = 2.0 * f(
        params, x2, data.spins, data.atoms, data.charges
    )  # log prob of proposal
    ratio = lp_2 - lp_1
    x_new, key, lp_new, num_accepts = mh_accept(
        x1, x2, lp_1, lp_2, ratio, key, num_accepts, 0)
  else:
    for species_idx, nspecies in enumerate(nspins):
      species_width = stddev[species_idx]
      species_shape = (x1.shape[0], nspecies * ndim)
      # Avoid unnecessary uniform draw when mix is disabled (p_large == 0).
      if p_large == 0:
        key_noise = subkey
        sigma = species_width
      else:
        key_choice, key_noise = jax.random.split(subkey)
        use_large = jax.random.uniform(key_choice) < p_large
        sigma = jnp.where(use_large, std_large, species_width)
      x2 = x1.at[:, start_idx * ndim:(start_idx + nspecies) * ndim].add(
          sigma * jax.random.normal(key_noise, shape=species_shape))  # proposal
      lp_2 = 2. * f(params, x2, data.spins, data.atoms, data.charges)  # log prob of proposal
      ratio = lp_2 - lp_1

      start_idx += nspecies
      key, subkey = jax.random.split(key)
      x1, key, lp_1, num_accepts = mh_accept(
          x1, x2, lp_1, lp_2, ratio, key, num_accepts, species_idx)
    x_new = x1
    lp_new = lp_1
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts


def mh_update_contact(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev: jnp.ndarray,
    nspins: Tuple[int, ...],
    ndim: int,
    lat: Lattice | None = None,
    alpha: float = 0.0,
):
  """Contact-biased MH: per-species updates, move nearest electron toward positron.

  Behaviour:
  - For each species (like `mh_update`), freeze other species and propose
    Gaussian moves for this species only.
  - Additionally, for each walker we find the electron nearest the positron
    (positron assumed last particle) and add an `alpha*(rp-ri)` shift to that
    electron in the species where it belongs.
  - The proposal is asymmetric; we compute the log q(R'|R)-log q(R|R') term
    for the moved block and include it in the MH ratio.
  """
  # Prepare shapes
  key_loop = key
  x1 = data.positions  # (batch, n_particles*ndim)
  batch_size = x1.shape[0]
  n_particles = x1.shape[1] // ndim

  batch_idx = jnp.arange(batch_size)

  # mh_update_contact requires lattice-aware minimum-image distances.
  if lat is None:
    raise NotImplementedError("mh_update_contact requires `lat` (periodic boundaries)")

  # We'll iterate species (small number) and perform vectorized batch moves.
  start_idx = 0
  new_pos = x1
  for species_idx, nspecies in enumerate(nspins):
    key_loop, key_noise = jax.random.split(key_loop)
    species_width = stddev if jnp.ndim(stddev) == 0 else stddev[species_idx]

    # slice for this species in flattened coords
    slice_start = start_idx * ndim
    slice_end = (start_idx + nspecies) * ndim
    x1_species = new_pos[:, slice_start:slice_end]
    # reshape according to current positions (so selection uses up-to-date state)
    x1_species_reshaped = jnp.reshape(x1_species, (batch_size, nspecies, ndim))

    # Get positron position (last particle) without reshaping the whole array.
    pos_positron = jnp.reshape(new_pos[:, -ndim:], (batch_size, ndim))
    pos_positron_expanded = pos_positron[:, None, :]
    # If this species block contains the positron, perform a plain Gaussian
    # proposal (no contact shift) for that block and continue.
    if start_idx <= (n_particles - 1) < start_idx + nspecies:
      noise = jax.random.normal(key_noise, shape=(batch_size, nspecies, ndim))
      x2_species_reshaped = x1_species_reshaped + species_width * noise
      x2 = new_pos.at[:, slice_start:slice_end].set(
        jnp.reshape(x2_species_reshaped, (batch_size, nspecies * ndim)))
      lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
      ratio = lp_2 - lp_1
      new_pos, key_loop, lp_1, num_accepts = mh_accept(
        new_pos, x2, lp_1, lp_2, ratio, key_loop, num_accepts, species_idx)
      start_idx += nspecies
      continue

    diff = pos_positron_expanded - x1_species_reshaped
    diff_flat = jnp.reshape(diff, (-1, ndim))
    dr, dr_norm = min_image_distance_triclinic(diff_flat, lat)
    dr_norm = jnp.reshape(dr_norm, (batch_size, nspecies))
    chosen_local = jnp.argmin(dr_norm, axis=1)

    # gaussian noise proposal for all particles of this species
    noise = jax.random.normal(key_noise, shape=(batch_size, nspecies, ndim))
    x2_species_reshaped = x1_species_reshaped + species_width * noise

    # use min-image vector dr (rp - ri) for the shift
    dr_reshaped = jnp.reshape(dr, (batch_size, nspecies, ndim))
    chosen_dr = dr_reshaped[batch_idx, chosen_local, :]
    shift = alpha * chosen_dr

    # apply alpha shift to chosen local index
    x2_species_reshaped = x2_species_reshaped.at[batch_idx, chosen_local, :].add(shift)

    # construct full proposed positions by replacing this species block
    x2 = new_pos.at[:, slice_start:slice_end].set(
        jnp.reshape(x2_species_reshaped, (batch_size, nspecies * ndim)))

    # compute asymmetric proposal log-density difference for this species block
    # forward delta and forward residuals (uses chosen_local computed above)
    delta_fwd = jnp.zeros_like(x1_species_reshaped)
    delta_fwd = delta_fwd.at[batch_idx, chosen_local, :].set(shift)
    eps_fwd = x2_species_reshaped - x1_species_reshaped - delta_fwd

    # Recompute chosen index on the proposed configuration (reverse selector).
    # This handles the small-probability case where the argmin index changes.
    diff_rev = pos_positron_expanded - x2_species_reshaped
    diff_rev_flat = jnp.reshape(diff_rev, (-1, ndim))
    dr_rev, dr_rev_norm = min_image_distance_triclinic(diff_rev_flat, lat)
    dr_rev_norm = jnp.reshape(dr_rev_norm, (batch_size, nspecies))
    chosen_local_rev = jnp.argmin(dr_rev_norm, axis=1)

    # reverse delta (what reverse proposal would have added) and reverse residuals
    delta_rev = jnp.zeros_like(x1_species_reshaped)
    dr_rev_reshaped = jnp.reshape(dr_rev, (batch_size, nspecies, ndim))
    chosen_dr_rev = dr_rev_reshaped[batch_idx, chosen_local_rev, :]
    # dr_rev equals rp - ri', so reverse shift = alpha * dr_rev
    shift_rev = alpha * chosen_dr_rev
    delta_rev = delta_rev.at[batch_idx, chosen_local_rev, :].set(shift_rev)
    eps_rev = x1_species_reshaped - x2_species_reshaped - delta_rev

    sigma2 = (species_width ** 2)
    norm_fwd = jnp.sum((eps_fwd ** 2), axis=(1, 2)) / sigma2
    norm_rev = jnp.sum((eps_rev ** 2), axis=(1, 2)) / sigma2
    log_q_diff = 0.5 * (norm_rev - norm_fwd)

    lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
    ratio = lp_2 - lp_1 - log_q_diff

    # accept/reject for this species block (species_idx used for per-species counting)
    new_pos, key_loop, lp_1, num_accepts = mh_accept(
        new_pos, x2, lp_1, lp_2, ratio, key_loop, num_accepts, species_idx)

    start_idx += nspecies

  new_data = networks.FermiNetData(**(dict(data) | {'positions': new_pos}))
  return new_data, key_loop, lp_1, num_accepts


def mh_block_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev: jnp.ndarray,
    nspins: Tuple[int, ...],
    ndim: int,
    atoms=None,
    blocks=1,
    i=0,
    mix_width: float = 1.0,
    mix_prob: float = 0.0,
):
  """Performs one Metropolis-Hastings step for a block of electrons.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with LogFermiNetLike signature which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configuration (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: Not implemented. Raises an error if not None.
    ndim: dimensionality of system.
    blocks: number of blocks to split electron updates into.
    i: index of block of electrons to move.

  Returns:
    (x, key, lp, num_accepts), where:
      x: MCMC configurations with updated positions.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.

  Raises:
    NotImplementedError: if atoms is supplied.
  """
  if atoms is not None:
    raise NotImplementedError("Asymmetric moves not implemented")
  p_large = mix_prob
  std_large = jnp.asarray(mix_width)
  key, subkey = jax.random.split(key)
  batch_size = data.positions.shape[0]
  start_idx = 0
  new_pos = data.positions
  for species_idx, nspecies in enumerate(nspins):
    species_width = stddev[species_idx]

    x1 = new_pos[:, start_idx * ndim:(start_idx + nspecies) * ndim]
    pad = (blocks - nspecies % blocks) % blocks
    # reshape into blocks
    x1 = jnp.reshape(
        jnp.pad(x1, ((0, 0), (0, pad * ndim))),
        [batch_size, blocks, -1, ndim],
    )
    ii = i % blocks
    # update block ii
    # Avoid unnecessary uniform draw when mix is disabled (p_large == 0).
    if p_large == 0:
      key_noise = subkey
      sigma = species_width
    else:
      key_choice, key_noise = jax.random.split(subkey)
      use_large = jax.random.uniform(key_choice) < p_large
      sigma = jnp.where(use_large, std_large, species_width)
    x2 = x1.at[:, ii].add(
        sigma * jax.random.normal(key_noise, shape=x1[:, ii].shape))
    x2 = jnp.reshape(x2, [batch_size, -1])
    # re-implant block into original array
    if pad > 0:
      x2 = x2[..., :-pad*ndim]
    x2 = new_pos.at[:, start_idx * ndim:(start_idx + nspecies) * ndim].set(x2)
    x1 = new_pos
    # log prob of proposal
    lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
    ratio = lp_2 - lp_1

    x1, key, lp_1, num_accepts = mh_accept(
        x1, x2, lp_1, lp_2, ratio, key, num_accepts, species_idx)
    new_pos = x1

    start_idx += nspecies
  new_data = networks.FermiNetData(**(dict(data) | {'positions': new_pos}))
  return new_data, key, lp_1, num_accepts


def make_mcmc_step(batch_network,
                   batch_per_device,
                   nspins,
                   ndim,
                   steps=10,
                   atoms=None,
                   lat: Lattice = None,
                   sample_all=True,
                   blocks=1,
                   mix_width: float = 1.0,
                   mix_prob: float = 0.0,
                   enhance_alpha: float = 0.0):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    atoms: atom positions. If given, an asymmetric move proposal is used based
      on the harmonic mean of electron-atom distances for each electron.
      Otherwise the (conventional) normal distribution is used.
    ndim: Dimensionality of the system (usually 3).
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  enhance_contact = enhance_alpha > 0.0
  inner_fun = mh_block_update if blocks > 1 else mh_update

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the move proposal.

    Returns:
      (data, pmove), where data is the updated MCMC configurations, key the
      updated RNG state and pmove the average probability a move was accepted.
    """
    pos = data.positions

    def step_fn(i, x):
      if enhance_contact and blocks == 1:
        return mh_update_contact(
            params,
            batch_network,
            *x,
            stddev=width,
            nspins=nspins,
            ndim=ndim,
            lat=lat,
            alpha=enhance_alpha)
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev=width,
          mix_width=mix_width,
          mix_prob=mix_prob,
          nspins=nspins,
          ndim=ndim,
          atoms=atoms,
          blocks=blocks,
          i=i)

    nsteps = steps * blocks
    nspecies = len(nspins)
    logprob = 2.0 * batch_network(
        params, pos, data.spins, data.atoms, data.charges
    )
    if sample_all:
      new_data, key, _, num_accepts = lax.fori_loop(
          0, nsteps, step_fn, (data, key, logprob, 0.0))
      assert jnp.ndim(num_accepts) == 0
      pmove = num_accepts / (nsteps * batch_per_device)
    else:
      new_data, key, _, num_accepts = lax.fori_loop(
          0, nsteps, step_fn, (data, key, logprob, 
          jnp.zeros(nspecies))
      )
      assert jnp.ndim(num_accepts) == 1
      pmove = num_accepts / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)
    return new_data, pmove

  return mcmc_step


def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    min_width: float = 1e-3,
    max_width: float = 20.0,
    apply_pbc: bool = False,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
  """Updates the width in MCMC steps.

  Args:
    t: Current step.
    width: Current MCMC width.
    adapt_frequency: The number of iterations after which the update is applied.
    pmove: Acceptance ratio in the last step.
    pmoves: Acceptance ratio over the last N steps, where N is the number of
      steps between MCMC width updates.
    max_width: Maximum allowed MCMC step width.
    apply_pbc: Whether periodic boundary conditions are applied. If True,
      the width will be constrained by max_width.
    pmove_max: The upper threshold for the range of allowed pmove values
    pmove_min: The lower threshold for the range of allowed pmove values

  Returns:
    width: Updated MCMC width.
    pmoves: Updated `pmoves`.
  """

  t_since_mcmc_update = t % adapt_frequency
  if np.ndim(pmoves) == 1: # sample_all is True
    pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
    if t > 0 and t_since_mcmc_update == 0:
      if np.mean(pmoves) > pmove_max:
        width *= 1.1
      elif np.mean(pmoves) < pmove_min:
        width /= 1.1
  else:
      if t > 0 and t_since_mcmc_update == 0:
        mean_pmoves = jnp.mean(pmoves, axis=1)
        width = width.at[:, jnp.where(mean_pmoves > pmove_max)].multiply(1.1)
        width = width.at[:, jnp.where(mean_pmoves < pmove_min)].divide(1.1)
        pmoves[:,:] = 0
      pmoves[:,t%adapt_frequency] = pmove
  if apply_pbc:
    width = jnp.minimum(width, max_width)
    width = jnp.maximum(width, min_width)
  return width, pmoves
