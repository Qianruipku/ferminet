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
  
  if (jax.device_count() // jax.local_device_count()) > 1:
    data = multihost_utils.process_allgather(data)
    sharded_key = multihost_utils.process_allgather(sharded_key)
  ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}.npz')
  if process == 0:
    logging.info('Saving checkpoint %s', ckpt_filename)
    single_device_params = jax.tree.map(lambda x: x[0], params)
    single_device_mcmc_width = jax.tree.map(lambda x: x[0], mcmc_width)

    with open(ckpt_filename, 'wb') as f:
      np.savez(
          f,
          t=t,
          data=dataclasses.asdict(data),
          params=single_device_params,
          opt_state=np.asarray(opt_state, dtype=object),
          mcmc_width=single_device_mcmc_width,
          pmoves=pmoves,
          density_state=(dataclasses.asdict(density_state)
                         if density_state else None),
          sharded_key=sharded_key,
          weighted_stats=weighted_stats)
  return ckpt_filename


def restore(restore_filename: str, batch_size: Optional[int] = None):
  """Restores data saved in a checkpoint.

  Args:
    restore_filename: filename containing checkpoint.
    batch_size: total batch size to be used. If present, check the data saved in
      the checkpoint is consistent with the batch size requested for the
      calculation.

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
  logging.info('Loading checkpoint %s on process %i', restore_filename, process)
  with open(restore_filename, 'rb') as f:
    ckpt_data = np.load(f, allow_pickle=True)
    # Retrieve data from npz file. Non-array variables need to be converted back
    # to natives types using .tolist().
    t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.
    
    # Load MCMC data conditionally
    data = networks.FermiNetData(**ckpt_data['data'].item())
    previous_devices = data.positions.shape[0]
    current_devices = jax.local_device_count()

    single_device_params = ckpt_data['params'].item()
    params = kfac_jax.utils.replicate_all_local_devices(single_device_params)
    
    # Handle optimizer state adaptation for different device counts
    previous_opt_state = ckpt_data['opt_state'].tolist()
    
    def adapt_kfac_opt_state(opt_state, current_devices):
      """Adapt KFAC optimizer state to target number of devices."""
      if previous_devices == current_devices:
        return opt_state
      
      def adapt_array_first_dim(arr):
        """Adapt array's first dimension to match current_devices."""
        if hasattr(arr, 'shape') and len(arr.shape) > 0:
          previous_devices_in_arr = arr.shape[0]
          if previous_devices_in_arr == current_devices:
            # Already correct size
            return arr
          else:
            # All other cases: take first device data and adapt to current_devices
            data = arr[0]  # Shape (...,) - extract data from first device
            if current_devices == 1:
              return jnp.expand_dims(data, axis=0)  # Shape (1, ...)
            else:
              return jnp.array([data] * current_devices)  # Shape (current_devices, ...)
        return arr
      
      # Use JAX tree operations to recursively apply the transformation
      adapted_state = jax.tree_util.tree_map(adapt_array_first_dim, opt_state)
      return adapted_state
    
    opt_state = adapt_kfac_opt_state(previous_opt_state, current_devices)
    
    single_device_mcmc_width = ckpt_data['mcmc_width']
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(single_device_mcmc_width)
    pmoves = ckpt_data.get('pmoves', None)
    if ckpt_data['density_state']:
      density_state = observables.DensityState(
          **ckpt_data['density_state'].item())
    else:
      density_state = None

    sharded_key = ckpt_data.get('sharded_key', None)
    weighted_stats_data = ckpt_data.get('weighted_stats', None)
    if weighted_stats_data is not None:
      weighted_stats = weighted_stats_data.item()
    else:
      weighted_stats = None

    if len(data.positions.shape) > 3: # More than one host for data
      if data is not None:
        data = jax.tree_util.tree_map(lambda x: x[process], data)
      sharded_key = jax.tree_util.tree_map(lambda x: x[process], sharded_key)

    if (data.positions.shape[0] * data.positions.shape[1] > batch_size):
      logging.warning(
          f'Batch size (={data.positions.shape[0] * data.positions.shape[1]}) in checkpoint does not match requested batch size (={batch_size}). '
          'Truncating data to match requested batch size.')
      batch_per_device = batch_size // data.positions.shape[0]
      data.spins = data.spins[:,:batch_per_device,:]
      data.atoms = data.atoms[:,:batch_per_device,:, :]
      data.charges = data.charges[:,:batch_per_device,:]
      data.positions = data.positions[:,:batch_per_device,:]
    elif(data.positions.shape[0] * data.positions.shape[1] == batch_size):
      # Redistribute data across devices if device count changed
      if previous_devices != current_devices:
        total_batch = data.positions.shape[0] * data.positions.shape[1]
        batch_per_device = total_batch // current_devices

        def redistribute_array(arr):
          # Flatten the first two dimensions, then reshape to target device count
          original_shape = arr.shape
          flattened = arr.reshape((total_batch,) + original_shape[2:])
          new_shape = (current_devices, batch_per_device) + original_shape[2:]
          return flattened.reshape(new_shape)

        data = networks.FermiNetData(
            positions=redistribute_array(data.positions),
            spins=redistribute_array(data.spins),
            atoms=redistribute_array(data.atoms),
            charges=redistribute_array(data.charges)
        )
    
    # When restarting with float32, we need to convert to float64 
    default_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    if(data.positions.dtype != default_dtype):
      data.positions = data.positions.astype(default_dtype)
      params = jax.tree_util.tree_map(lambda x: jax.lax.convert_element_type(x, default_dtype), params)
      opt_state = jax.tree_util.tree_map(lambda x: jax.lax.convert_element_type(x, default_dtype), opt_state)
  return t, data, params, opt_state, mcmc_width, pmoves, density_state, sharded_key, weighted_stats
