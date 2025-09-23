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

"""Multiplicative Jastrow factors."""

import enum
from typing import Any, Callable, Iterable, Mapping, Union

import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
    masses: jnp.ndarray,
    charges: jnp.ndarray,
    ndim: int,
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps."""

  masses = jnp.asarray(masses)
  charges = jnp.asarray(charges)
  
  red_masses = (masses.reshape(1, -1) * masses.reshape(-1, 1)) / (masses.reshape(1, -1) + masses.reshape(-1, 1))
  charge_prods = charges.reshape(1, -1) * charges.reshape(-1, 1)

  diag_cusp = jnp.eye(len(nspins)) * red_masses * charge_prods / 2
  offdiag_cusp = (1 - jnp.eye(len(nspins))) * red_masses * charge_prods
  neg_cusp = diag_cusp + offdiag_cusp

  splits = [sum(nspins[:i+1]) for i in range(len(nspins) - 1)]

  r_ees = [
      jnp.split(r, splits, axis = 1)
      for r in jnp.split(r_ee, splits, axis = 0)
  ]

  # This part needs to be refactored; it is a double for loop,
  # but only on the number of species which is always a small number
  jastrow_value = jnp.asarray(0.0)
  for i in range(len(nspins)):
    for j in range(i, len(nspins)):
      if i == j:
        pos = r_ees[i][j][jnp.triu_indices(nspins[i], k=1)].flatten()
      else:
        pos = r_ees[i][j].flatten()
            
      jastrow_value += jnp.sum(jastrow_fun(pos, neg_cusp[i, j], params['alpha'][i, j]))

  return jastrow_value


def make_simple_ee_jastrow(
      nspins: jnp.ndarray,
      masses: jnp.ndarray,
      charges: jnp.ndarray,
      ndim: int,
  ) -> ...:
  """Creates a Jastrow factor for electron-electron cusps."""
  if ndim != 3:
    raise NotImplementedError("Jastrow only implemented for ndim = 3")

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    params['alpha'] = jnp.ones(
        shape=(len(nspins), len(nspins)),
    )
    return params

  def apply(
      r_ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    return _jastrow_ee(r_ee, params, nspins, 
        jastrow_fun=simple_ee_cusp_fun, masses = masses, charges = charges, ndim = ndim)

  return init, apply


def get_jastrow(
      jastrow: JastrowType,
      nspins: jnp.ndarray,
      masses: jnp.ndarray,
      charges: jnp.ndarray,
      ndim: int,
  ) -> ...:
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    jastrow_init, jastrow_apply = make_simple_ee_jastrow(
      nspins, masses, charges, ndim)
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

  return jastrow_init, jastrow_apply
