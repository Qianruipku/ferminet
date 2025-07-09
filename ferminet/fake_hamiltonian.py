# Copyright 2022 DeepMind Technologies Limited.
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
# limitations under the License

"""Fake Hamiltonian for testing purposes.
This Hamiltonian is used to test the training and inference code without
requiring a real Hamiltonian.
"""

import chex
from ferminet import networks
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Sequence, Tuple
from ferminet import hamiltonian

def local_energy() -> hamiltonian.LocalEnergy:
  """Creates the local energy function in periodic boundary conditions.

  Args:
    None

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    
    return 0.0, None

  return _e_l