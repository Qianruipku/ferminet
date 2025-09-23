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

"""Initialization functions for FermiNet training."""

from typing import Tuple, Optional, Mapping, Any, Sequence
import jax
import jax.numpy as jnp
import kfac_jax
from absl import logging
import ml_collections
from ferminet import checkpoint
from ferminet import networks
from ferminet.utils import system


def _assign_spin_configuration(
    particles: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
  spin_values = [1. if i % 2 == 0 else -1. for i in range(len(particles))]
  spins = jnp.concatenate([jnp.full(count, value) for count, value in zip(particles, spin_values)])
  return jnp.tile(spins[None], reps=(batch_size, 1))


def init_electrons(  # pylint: disable=dangerous-default-value
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    ndim: int,
    batch_size: int,
    init_width: float,
    core_electrons: Mapping[str, int] = {},
    max_iter: int = 10_000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    ndim: number of dimensions
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.
    core_electrons: mapping of element symbol to number of core electrons
      included in the pseudopotential.
    max_iter: maximum number of iterations to try to find a valid initial
        electron configuration for each atom. If reached, all electrons are
        initialised from a Gaussian distribution centred on the origin.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
    of spin configurations, where 1 and -1 indicate alpha and beta electrons
    respectively.
  """

  atomic_charges = jnp.array([atom.charge for atom in molecule])
  if len(atomic_charges) == 1:
    electron_positions = jnp.tile(jnp.asarray(molecule[0].coords), sum(electrons))
    electron_positions = jnp.tile(electron_positions, (batch_size, 1))
  else:
    if sum(atomic_charges) == 0:
      raise ValueError("If there are no charged atoms, please add only one neutral atom")
    
    atomic_positions = jnp.array([atom.coords for atom in molecule])

    key, subkey = jax.random.split(key)
    inds = jax.random.choice(subkey, len(molecule), shape=(batch_size, sum(electrons)), p=atomic_charges)

    electron_positions = atomic_positions[inds].reshape(batch_size, sum(electrons) * ndim)

  key, subkey = jax.random.split(key)
  electron_positions += (
      jax.random.normal(subkey, shape=electron_positions.shape)
      * init_width
  )

  electron_spins = _assign_spin_configuration(
      electrons, batch_size
  )

  return electron_positions, electron_spins


def init_mcmc_data(
    key: jax.random.PRNGKey,
    cfg: ml_collections.ConfigDict,
    data_shape: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    batch_charges: jnp.ndarray,
    total_host_batch_size: int,
    core_electrons: Mapping[str, int],
) -> Tuple[jax.random.PRNGKey, networks.FermiNetData]:
  """Initialize MCMC walker data from scratch.
  
  Args:
    key: Random key for initialization.
    cfg: Configuration object.
    data_shape: Shape of data arrays (devices, batch_size_per_device).
    batch_atoms: Atomic positions for each batch element.
    batch_charges: Atomic charges for each batch element.
    total_host_batch_size: Total batch size per host.
    core_electrons: Core electrons per atom.
    
  Returns:
    Tuple of (updated_key, FermiNetData object with initialized walker positions and spins).
  """
  key, subkey = jax.random.split(key)
  # make sure data on each host is initialized differently
  subkey = jax.random.fold_in(subkey, jax.process_index())
  # create electron state (position and spin)
  pos, spins = init_electrons(
      subkey,
      cfg.system.molecule,
      cfg.system.particles,
      cfg.system.ndim,
      batch_size=total_host_batch_size,
      init_width=cfg.mcmc.init_width,
      core_electrons=core_electrons,
  )
  # For excited states, each device has a batch of walkers, where each walker
  # is nstates * nelectrons. The vmap over nstates is handled in the function
  # created in make_total_ansatz
  pos = jnp.reshape(pos, data_shape + (-1,))
  pos = kfac_jax.utils.broadcast_all_local_devices(pos)
  spins = jnp.reshape(spins, data_shape + (-1,))
  spins = kfac_jax.utils.broadcast_all_local_devices(spins)
  data = networks.FermiNetData(
      positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
  )
  return key, data


def initialize_training_data_and_checkpoints(
    cfg: ml_collections.ConfigDict,
    key: jax.random.PRNGKey,
    data_shape: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    batch_charges: jnp.ndarray,
    total_host_batch_size: int,
    host_batch_size: int,
    core_electrons: Mapping[str, int],
) -> Tuple[
    int,  # t_init
    networks.FermiNetData,  # data
    jax.random.PRNGKey,  # key (updated)
    Any,  # params
    Optional[Any],  # opt_state_ckpt
    Optional[jnp.ndarray],  # mcmc_width_ckpt
    Optional[Any],  # density_state_ckpt
    Optional[jax.random.PRNGKey],  # sharded_key_ckpt
    Optional[Any],  # weighted_stats_ckpt
]:
  """Initialize training data and handle checkpoint restoration.
  
  Args:
    cfg: Configuration object.
    key: Random key for initialization.
    data_shape: Shape of data arrays (devices, batch_size_per_device).
    batch_atoms: Atomic positions for each batch element.
    batch_charges: Atomic charges for each batch element.
    total_host_batch_size: Total batch size per host.
    host_batch_size: Batch size per host.
    core_electrons: Core electrons per atom.
    
  Returns:
    Tuple containing initialization time, data, updated key, params, and checkpoint states.
  """
  # Set up checkpointing paths
  ckpt_save_path = checkpoint.get_restore_path(cfg.log.save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  ckpt_restore_filename = (
      checkpoint.find_last_checkpoint(ckpt_save_path) or
      checkpoint.find_last_checkpoint(ckpt_restore_path))

  if ckpt_restore_filename:
    (t_init,
     data,
     params,
     opt_state_ckpt,
     mcmc_width_ckpt,
     density_state_ckpt,
     sharded_key_ckpt,
     weighted_stats_ckpt) = checkpoint.restore(
         ckpt_restore_filename, host_batch_size, cfg.restart.load_data)
    
    # If we didn't load MCMC data from checkpoint, initialize it fresh
    if not cfg.restart.load_data:
      logging.info('Do not use MCMC data in checkpoint. Initializing new MCMC data.')
      key, data = init_mcmc_data(
          key,
          cfg, 
          data_shape,
          batch_atoms,
          batch_charges,
          total_host_batch_size,
          core_electrons,
      )
  else:
    logging.info('No checkpoint found. Training new model.')
    key, data = init_mcmc_data(
        key,
        cfg, 
        data_shape,
        batch_atoms,
        batch_charges,
        total_host_batch_size,
        core_electrons,
    )

    t_init = 0
    params = None  # This will need to be set by the caller after network initialization
    opt_state_ckpt = None
    mcmc_width_ckpt = None
    density_state_ckpt = None
    sharded_key_ckpt = None
    weighted_stats_ckpt = None

  return (
      t_init,
      data,
      key,
      params,
      opt_state_ckpt,
      mcmc_width_ckpt,
      density_state_ckpt,
      sharded_key_ckpt,
      weighted_stats_ckpt
  )