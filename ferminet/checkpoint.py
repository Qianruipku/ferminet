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

"""Super simple checkpoints using numpy."""

import dataclasses
import datetime
import os
from typing import Optional
import zipfile

from absl import logging
from ferminet import networks
from ferminet import observables
from ferminet.utils import statistics
from ferminet.utils import state_consistency as stat_cons
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
import kfac_jax


def find_last_checkpoint(ckpt_path: Optional[str] = None) -> Optional[str]:
  """Finds most recent valid checkpoint in a directory.

  Args:
    ckpt_path: Directory containing checkpoints.

  Returns:
    Last QMC checkpoint (ordered by sorting all checkpoints by name in reverse)
    or None if no valid checkpoint is found or ckpt_path is not given or doesn't
    exist. A checkpoint is regarded as not valid if it cannot be read
    successfully using np.load.
  """
  if ckpt_path and os.path.exists(ckpt_path):
    files = [f for f in os.listdir(ckpt_path) if 'qmcjax_ckpt_' in f]
    # Handle case where last checkpoint is corrupt/empty.
    for file in sorted(files, reverse=True):
      fname = os.path.join(ckpt_path, file)
      with open(fname, 'rb') as f:
        try:
          np.load(f, allow_pickle=True)
          return fname
        except (OSError, EOFError, zipfile.BadZipFile):
          logging.info('Error loading checkpoint %s. Trying next checkpoint...',
                       fname)
  return None


def create_save_path(save_path: Optional[str]) -> str:
  """Creates the directory for saving checkpoints, if it doesn't exist.

  Args:
    save_path: directory to use. If false, create a directory in the working
      directory based upon the current time.

  Returns:
    Path to save checkpoints to.
  """
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  default_save_path = os.path.join(os.getcwd(), f'ferminet_{timestamp}')
  ckpt_save_path = save_path or default_save_path
  if ckpt_save_path and not os.path.isdir(ckpt_save_path):
    os.makedirs(ckpt_save_path)
  return ckpt_save_path


def get_restore_path(restore_path: Optional[str] = None) -> Optional[str]:
  """Gets the path containing checkpoints from a previous calculation.

  Args:
    restore_path: path to checkpoints.

  Returns:
    The path or None if restore_path is falsy.
  """
  if restore_path:
    ckpt_restore_path = restore_path
  else:
    ckpt_restore_path = None
  return ckpt_restore_path


def save_positions(positions: jnp.ndarray, step: int, save_path: str):
  """Saves out the electron positions in unwrapped coordinates.

  Args:
    positions: electron positions to save, shape (num_devices, batch_per_device, num_electrons, 3)
    step: current training step, used to name the file
    save_path: path to directory to save positions to. The position file is
      save_path/positions_$step.npy, where $step is the current training step.
  """
  process = jax.process_index()
  all_positions = multihost_utils.process_allgather(positions) # shape (num_hosts, num_local_devices, batch_per_device, num_electrons*3)
  if process == 0:
    pos_filename = os.path.join(save_path, f'positions_{step:06d}.npy')
    np.save(pos_filename, all_positions)

def save(save_path: str,
         t: int,
         data: networks.FermiNetData,
         params,
         opt_state,
         mcmc_width,
         pmoves,
         density_state: Optional[observables.DensityState] = None,
         sharded_key: Optional[jax.random.PRNGKey] = None,
         weighted_stats: Optional[statistics.WeightedStats] = None,
         sync_states: bool = False,
         check_consistency: bool = False,
         consistency_rtol: float = 1e-5,
         consistency_atol: float = 1e-8) -> str:
  """Saves checkpoint information to a npz file.

  Args:
    save_path: path to directory to save checkpoint to. The checkpoint file is
      save_path/qmcjax_ckpt_$t.npz, where $t is the number of completed
      iterations.
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.
    density_state: optional state of the density matrix calculation
    sharded_key: optional sharded random key state
    weighted_stats: optional exponentially weighted statistics
    sync_states: whether to synchronize state (params, opt_state, mcmc_width) from device 0 to all other devices/hosts
    check_consistency: whether to check state consistency (params, opt_state, mcmc_width) across devices/hosts
    consistency_rtol: relative tolerance for consistency check
    consistency_atol: absolute tolerance for consistency check

  Returns:
    path to checkpoint file.
  """
  process = jax.process_index()
  
  # Check state consistency (params, opt_state, mcmc_width)
  if check_consistency:
    state_dict = {
        'params': params,
        'opt_state': opt_state,
        'mcmc_width': mcmc_width
    }
    consistency_results = stat_cons.check_state_consistency(
        state_dict, rtol=consistency_rtol, atol=consistency_atol)
    
    if not consistency_results['overall']:
      inconsistent_states = [name for name, consistent in consistency_results.items() 
                           if name != 'overall' and not consistent]
      logging.warning(f'Detected inconsistency in states: {inconsistent_states}')

  if sync_states:
    synced_states = stat_cons.synchronize_state_from_device0(state_dict)
    params = synced_states['params']
    opt_state = synced_states['opt_state'] 
    mcmc_width = synced_states['mcmc_width']
  
  data = multihost_utils.process_allgather(data) # shape (num_hosts, num_local_devices, batch_per_device, ...)
  sharded_key = multihost_utils.process_allgather(sharded_key) # shape (num_hosts, num_local_devices, 2)
  ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}.npz')
  if process == 0:
    logging.info('Saving checkpoint %s', ckpt_filename)
    single_device_params = jax.tree.map(lambda x: x[0], params)
    single_device_mcmc_width = jax.tree.map(lambda x: x[0], mcmc_width)
    def use_0_if_needed(x):
      if hasattr(x, 'shape') and len(x.shape) > 0:
        return x[0]
      return x
    if isinstance(opt_state, tuple): # lamb, adam, ...
      single_device_opt_state = tuple(jax.tree_util.tree_map(use_0_if_needed, s) for s in opt_state)
      single_device_opt_state = np.asarray(single_device_opt_state, dtype=object)
    else: # kfac, ...
      single_device_opt_state = jax.tree_util.tree_map(use_0_if_needed, opt_state)


    with open(ckpt_filename, 'wb') as f:
      np.savez(
          f,
          t=t,
          data=dataclasses.asdict(data),
          params=single_device_params,
          opt_state=single_device_opt_state,
          mcmc_width=single_device_mcmc_width,
          pmoves=pmoves,
          density_state=(dataclasses.asdict(density_state)
                         if density_state else None),
          sharded_key=sharded_key,
          weighted_stats=weighted_stats)
  return ckpt_filename


def restore(restore_filename: str,
            load_opt_state: bool = True,
            load_data: bool = True,
            host_batch_size: int = None):
  """Restores data saved in a checkpoint.

  Args:
    restore_filename: filename containing checkpoint.
    host_batch_size: batch size per host to be used.

  Returns:
    (t, data, params, opt_state, mcmc_width, density_state, sharded_key, weighted_stats) tuple, where
    t: number of completed iterations.
    data: MCMC walker configurations, or None if load_data is False.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.
    density_state: optional state of the density matrix calculation
    sharded_key: optional sharded random key state
    weighted_stats: optional exponentially weighted statistics

  Raises:
    ValueError: if the leading dimension of data does not match the number of
    devices (i.e. the number of devices being parallelised over has changed) or
    if the total batch size is not equal to the number of MCMC configurations in
    data.
  """
  process = jax.process_index()
  current_num_processes = jax.process_count()
  current_local_devices = jax.local_device_count()
  current_total_devices = current_local_devices * current_num_processes
  default_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  logging.info('Loading checkpoint %s on process %i', restore_filename, process)
  with open(restore_filename, 'rb') as f:
    ckpt_data = np.load(f, allow_pickle=True)
    # Retrieve data from npz file. Non-array variables need to be converted back
    # to natives types using .tolist().

    # Number of completed iterations
    t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.

    # Load sharded_key and get current process and local devices info
    sharded_key = ckpt_data['sharded_key']
    previous_num_processes = sharded_key.shape[0]
    previous_local_devices = sharded_key.shape[1]
    previous_total_devices = previous_num_processes * previous_local_devices

    if previous_total_devices != current_total_devices:
      logging.warning(
          f'Number of devices has changed from {previous_total_devices} to {current_total_devices} since checkpoint was saved. '
          'Recreating sharded_key.')
      sharded_key = None
    else:
      if previous_local_devices != current_local_devices:
        sharded_key = jax.tree_util.tree_map(lambda x: x.reshape((current_num_processes, current_local_devices, 2)), sharded_key)
      sharded_key = jax.tree_util.tree_map(lambda x: x[process], sharded_key)

    # Load mcmc data
    if load_data:
      data = networks.FermiNetData(**ckpt_data['data'].item())
      previous_batch_per_device = data.positions.shape[2]
      previous_total_batch = previous_total_devices * previous_batch_per_device
      needed_total_batch = host_batch_size * current_num_processes
      if previous_total_batch < needed_total_batch:
        logging.warning(
            f'Batch size has increased from {previous_total_batch} to {needed_total_batch} since checkpoint was saved.')
        data = None
      else:
        if previous_num_processes != current_num_processes or \
           previous_local_devices != current_local_devices:
          data = jax.tree_util.tree_map(lambda x:
                                        x.reshape((current_num_processes, current_local_devices, -1) + x.shape[3:]),
                                        data)
        data = jax.tree_util.tree_map(lambda x: x[process], data)
        now_batch_per_device = previous_total_batch // current_total_devices
        needed_batch_per_device = host_batch_size // current_local_devices
        if (needed_batch_per_device < now_batch_per_device):
          logging.warning(
              f'Batch size per device (={now_batch_per_device}) in checkpoint is larger than requested batch size (={needed_batch_per_device}). '
              'Truncating data to match requested batch size.')
          data = jax.tree_util.tree_map(lambda x: x[:,:needed_batch_per_device], data)
    else:
      data = None
    if data is not None:
      need_convert = data.positions.dtype != default_dtype
      if need_convert:
        data.positions = data.positions.astype(default_dtype)
    else:
      need_convert = True
        
    # Load params
    single_device_params = ckpt_data['params'].item()
    params = kfac_jax.utils.replicate_all_local_devices(single_device_params)
    if need_convert and params is not None:
      params = jax.tree_util.tree_map(lambda x: jax.lax.convert_element_type(x, default_dtype), params)
    
    # Load opt_state
    if load_opt_state:
      previous_opt_state = ckpt_data['opt_state'].tolist()

      def adapt_opt_state(opt_state, current_local_devices):
        def copy_first_dim(arr):
          if hasattr(arr, 'shape') and len(arr.shape) >= 0:
              if need_convert:
                arr = jax.lax.convert_element_type(arr, default_dtype)
              if current_local_devices == 1:
                return jnp.expand_dims(arr, axis=0)  # Shape (1, ...)
              else:
                return jnp.array([arr] * current_local_devices)  # Shape (current_local_devices, ...)
          return arr
        if isinstance(opt_state, list):  # lamb, adam, ...
          adapted_state = tuple(jax.tree_util.tree_map(copy_first_dim, s) for s in opt_state)
        else: # kfac, ...
          adapted_state = jax.tree_util.tree_map(copy_first_dim, opt_state)
        return adapted_state

      opt_state = adapt_opt_state(previous_opt_state, current_local_devices)
    else:
      opt_state = None

    # Load MCMC width and pmoves and weighted_stats
    single_device_mcmc_width = ckpt_data['mcmc_width']
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(single_device_mcmc_width)
    pmoves = ckpt_data['pmoves']
    weighted_stats_data = ckpt_data.get('weighted_stats', None)
    if weighted_stats_data is not None:
      weighted_stats = weighted_stats_data.item()
    else:
      weighted_stats = None
    
    if ckpt_data['density_state']:
      density_state = observables.DensityState(
          **ckpt_data['density_state'].item())
    else:
      density_state = None
  return t, data, params, opt_state, mcmc_width, pmoves, density_state, sharded_key, weighted_stats
