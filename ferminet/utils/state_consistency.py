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

"""State consistency checking and synchronization utilities for multi-device/multi-host training."""

from typing import Optional
from absl import logging
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import kfac_jax


def check_state_consistency(state_dict, rtol: float = 1e-5, atol: float = 1e-8, 
                          state_names: Optional[list] = None) -> dict:
  """Check consistency of multiple state variables across multiple hosts and devices.
  
  Args:
    state_dict: Dictionary of state pytrees (e.g., {'params': params, 'opt_state': opt_state})
                Each state's first dimension should correspond to different devices
    rtol: Relative tolerance
    atol: Absolute tolerance
    state_names: Optional list of state names to check. If None, check all states in state_dict
    
  Returns:
    Dictionary with consistency results for each state, e.g., 
    {'params': True, 'opt_state': False, 'overall': False}
  """
  process = jax.process_index()
  num_processes = jax.process_count()
  local_device_count = jax.local_device_count()
  
  # Determine which states to check
  if state_names is None:
    state_names = list(state_dict.keys())
  
  results = {}
  overall_consistent = True
  
  # Check each state
  for state_name in state_names:
    if state_name not in state_dict:
      logging.warning(f'State {state_name} not found in state_dict, skipping...')
      results[state_name] = True  # Skip missing states
      continue
    
    state = state_dict[state_name]
    
    # Get global reference: device 0 from process 0
    single_device_state = jax.tree_util.tree_map(lambda x: x[0], state)
    if num_processes > 1:
      # Broadcast process 0's device 0 state to all processes as reference
      global_reference = multihost_utils.broadcast_one_to_all(
          single_device_state, is_source=process == 0)
    else:
      global_reference = single_device_state
    
    def check_array_consistency(arr, ref_arr):
      """Check consistency of array against global reference"""
      if not hasattr(arr, 'shape') or len(arr.shape) == 0:
        return True
      
      # Check all local devices against global reference
      for i in range(local_device_count):
        if not jnp.allclose(ref_arr, arr[i], rtol=rtol, atol=atol):
          max_diff = jnp.max(jnp.abs(ref_arr - arr[i]))
          logging.warning(
              f'Process {process} detected {state_name} inconsistency between '
              f'global reference (process 0, device 0) and local device {i}, '
              f'max difference: {max_diff:.2e}')
          return False
      return True
    
    # Check consistency against global reference
    state_consistent = jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda x, ref: check_array_consistency(x, ref), 
            state, global_reference))
    
    results[state_name] = state_consistent
    if not state_consistent:
      overall_consistent = False
  
  results['overall'] = overall_consistent
  return results

def synchronize_state_from_device0(state_dict, state_names: Optional[list] = None):
  """Synchronize state variables from device 0 to all other devices and hosts.
  
  This function ensures all devices and hosts have exactly the same state as device 0
  on process 0. Useful for correcting small numerical drift that can accumulate
  during long training runs.
  
  Args:
    state_dict: Dictionary of state pytrees (e.g., {'params': params, 'opt_state': opt_state})
    state_names: Optional list of state names to synchronize. If None, sync all states
    
  Returns:
    The modified state_dict with synchronized states (in-place modification).
  """
  process = jax.process_index()
  num_processes = jax.process_count()
  local_device_count = jax.local_device_count()
  
  # Determine which states to synchronize
  if state_names is None:
    state_names = list(state_dict.keys())
  
  for state_name in state_names:
    if state_name not in state_dict:
      logging.warning(f'State {state_name} not found in state_dict, skipping synchronization...')
      continue
    
    state = state_dict[state_name]
    
    if num_processes > 1:
      # Multi-host case: first get device 0 state from process 0
      single_device_state = jax.tree_util.tree_map(lambda x: x[0], state)
      
      # Broadcast device 0 state from process 0 to all processes
      reference_state = multihost_utils.broadcast_one_to_all(
          single_device_state, is_source=process == 0)
      
      # Replicate to all local devices
      synchronized_state = kfac_jax.utils.replicate_all_local_devices(reference_state)
    else:
      # Single-host case: replicate device 0 state to all local devices
      reference_state = jax.tree_util.tree_map(lambda x: x[0], state)
      synchronized_state = kfac_jax.utils.replicate_all_local_devices(reference_state)
    
    # Modify state_dict in place
    state_dict[state_name] = synchronized_state
    
    logging.info(f'Synchronized {state_name} from device 0 (process 0) to all devices/hosts')
  
  return state_dict