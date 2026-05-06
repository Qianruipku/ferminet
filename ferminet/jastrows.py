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
  CUT_EE = enum.auto()
  MIXED_EE = enum.auto()


def _strict_upper_triangle_mask(size: int, dtype: jnp.dtype) -> jnp.ndarray:
  """Returns a strict upper-triangular mask with the requested dtype."""
  return jnp.triu(jnp.ones((size, size), dtype=dtype), k=1)


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
  cusp_matrix = diag_cusp + offdiag_cusp

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
        pos = r_ees[i][j]
        pair_mask = _strict_upper_triangle_mask(nspins[i], pos.dtype)
      else:
        pos = r_ees[i][j]
        pair_mask = jnp.ones_like(pos)

      term = jastrow_fun(pos, cusp_matrix[i, j], params['alpha'][i, j])
      jastrow_value += jnp.sum(term * pair_mask)

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


def _jastrow_ee_cut(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    masses: jnp.ndarray,
    charges: jnp.ndarray,
    cutoff_length: float,
    poly_order: int,
    C: int,
) -> jnp.ndarray:
  """Cutoff Jastrow factor with envelope exponent C.

  Implements the form

    x = r / L
    u(r) = L*(1 - x)^C Theta(1 - x) * [ alpha0
              + (cusp + alpha0*C) * x
              + sum_{l=2}^{N_u} alpha_l x^l ]

  Args:
    cutoff_length: cutoff L.
    poly_order: N_u, highest polynomial power in the bracket (>=1).
    C: envelope exponent.
  """
  masses = jnp.asarray(masses)
  charges = jnp.asarray(charges)

  red_masses = (masses.reshape(1, -1) * masses.reshape(-1, 1)) / (
      masses.reshape(1, -1) + masses.reshape(-1, 1))
  charge_prods = charges.reshape(1, -1) * charges.reshape(-1, 1)

  diag_cusp = jnp.eye(len(nspins)) * red_masses * charge_prods / 2
  offdiag_cusp = (1 - jnp.eye(len(nspins))) * red_masses * charge_prods
  cusp_matrix = diag_cusp + offdiag_cusp

  splits = [sum(nspins[:i+1]) for i in range(len(nspins) - 1)]
  r_ees = [
      jnp.split(r, splits, axis=1)
      for r in jnp.split(r_ee, splits, axis=0)
  ]

  jastrow_value = jnp.asarray(0.0)
  for i in range(len(nspins)):
    for j in range(i, len(nspins)):
      if i == j:
        pos = r_ees[i][j]
        pair_mask = _strict_upper_triangle_mask(nspins[i], pos.dtype)
      else:
        pos = r_ees[i][j]
        pair_mask = jnp.ones_like(pos)

      cusp = cusp_matrix[i, j]
      # params['alpha'] layout (length poly_order):
      #  index 0 -> alpha0
      #  index 1..(poly_order-1) -> alpha_l for l=2..N_u (stored at index l-1)
      alpha_arr = params['alpha'][i, j]
      alpha0 = alpha_arr[0]
      x = pos / cutoff_length

      # linear coefficient in the dimensionless polynomial P(x).
      linear_coeff = cusp + alpha0 * C

      # Build coefficient list from highest degree down to constant for Horner:
      coeffs = []
      # degrees l = N_u .. 2 -> stored at alpha_arr[l-1]
      for l in range(poly_order, 1, -1):
        coeffs.append(alpha_arr[l - 1])
      coeffs.append(linear_coeff)
      coeffs.append(alpha0)

      poly = jnp.zeros_like(x)
      for coeff in coeffs:
        poly = coeff + x * poly
      # envelope L * (1 - x)^C, active only when x < 1
      envelope = (1.0 - x) ** C
      cutoff_mask = (x < 1.0).astype(pos.dtype)
      u = cutoff_length * envelope * poly * cutoff_mask * pair_mask
      jastrow_value += jnp.sum(u)

  return jastrow_value


def make_cut_ee_jastrow(
    nspins: jnp.ndarray,
    masses: jnp.ndarray,
    charges: jnp.ndarray,
    ndim: int,
    cut_length: float = 1.0,
    poly_order: int = 3,
    C: int = 3,
) -> ...:
  """Cutoff Jastrow for PBC using x=r/L.

  u(r)=L*(1-x)^C * [alpha0 + linear*x + sum_{l>=2} alpha_l x^l], x=r/L.

  Args:
    nspins: number of particles in each spin/species channel.
    masses: particle masses, shape (n_species,).
    charges: particle charges, shape (n_species,).
    ndim: spatial dimension (must be 3).
    cut_length: fixed cutoff length L_u, default 1.0.
    poly_order: N_u, highest polynomial order in the inner bracket (>=1).
    C: envelope exponent (integer), passed through to the evaluator.
  """
  if ndim != 3:
    raise NotImplementedError('CUT_EE Jastrow only implemented for ndim = 3')
  if poly_order < 1:
    raise ValueError(f'poly_order must be >= 1, got {poly_order}')
  if cut_length <= 0:
    raise ValueError(f'cut_length must be > 0, got {cut_length}')

  n_species = len(nspins)
  cutoff_length = float(cut_length)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    # params['alpha'] stores the polynomial coefficients used inside the
    # bracket: index 0 -> alpha0; indices 1..(poly_order-1) -> alpha_l for
    # l=2..N_u stored at index l-1. Length = poly_order.
    params['alpha'] = jnp.zeros((n_species, n_species, poly_order))
    return params

  def apply(
      r_ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Evaluate cutoff Jastrow factor."""
    return _jastrow_ee_cut(
        r_ee, params, nspins,
      masses=masses,
      charges=charges,
      cutoff_length=cutoff_length,
      poly_order=poly_order,
      C=C,
    )

  return init, apply


def make_mixed_ee_jastrow(
    nspins: jnp.ndarray,
    masses: jnp.ndarray,
    charges: jnp.ndarray,
    ndim: int,
    pair_kind,
    cut_length: float = 1.0,
    poly_order: int = 3,
    C: int = 3,
) -> ...:
  """Per-species-pair Jastrow: each (i,j) pair independently uses SIMPLE_EE, CUT_EE, or NONE.

  Args:
    pair_kind: n_species x n_species indexable of JastrowType (or strings
      convertible via JastrowType[s.upper()]). Must be symmetric. Only the
      upper triangle (j >= i) is read; the lower triangle is ignored.
    cut_length: cutoff L for any CUT_EE pairs.
    poly_order: polynomial order N_u for any CUT_EE pairs.
    C: envelope exponent for any CUT_EE pairs.
  """
  if ndim != 3:
    raise NotImplementedError('MIXED_EE Jastrow only implemented for ndim = 3')
  if poly_order < 1:
    raise ValueError(f'poly_order must be >= 1, got {poly_order}')

  n_species = len(nspins)
  cutoff_length = float(cut_length)

  # Normalise pair_kind to JastrowType.
  def _to_jt(v):
    if isinstance(v, str):
      return JastrowType[v.upper()]
    return v

  norm_kind = [[_to_jt(pair_kind[i][j]) for j in range(n_species)]
               for i in range(n_species)]

  def init() -> Mapping[str, jnp.ndarray]:
    return {
        'simple_alpha': jnp.ones((n_species, n_species)),
        'cut_alpha': jnp.zeros((n_species, n_species, poly_order)),
    }

  def apply(
      r_ee,
      params: ParamTree,
      nspins: tuple,
  ) -> jnp.ndarray:
    """Evaluate mixed Jastrow factor.

    Args:
      r_ee: either a single array (non-PBC, used for all pairs) or a tuple
        (r_ee_cut, r_ee_simple) where r_ee_cut is used for CUT_EE pairs and
        r_ee_simple is used for SIMPLE_EE pairs (PBC case).
    """
    if isinstance(r_ee, tuple):
      r_ee_cut, r_ee_simple = r_ee
    else:
      r_ee_cut = r_ee_simple = r_ee

    masses_arr = jnp.asarray(masses)
    charges_arr = jnp.asarray(charges)
    red_masses = (masses_arr.reshape(1, -1) * masses_arr.reshape(-1, 1)) / (
        masses_arr.reshape(1, -1) + masses_arr.reshape(-1, 1))
    charge_prods = charges_arr.reshape(1, -1) * charges_arr.reshape(-1, 1)
    diag_cusp = jnp.eye(n_species) * red_masses * charge_prods / 2
    offdiag_cusp = (1 - jnp.eye(n_species)) * red_masses * charge_prods
    cusp_matrix = diag_cusp + offdiag_cusp

    splits = [sum(nspins[:k+1]) for k in range(len(nspins) - 1)]
    r_ees_cut = [
        jnp.split(r, splits, axis=1)
        for r in jnp.split(r_ee_cut, splits, axis=0)
    ]
    r_ees_simple = [
        jnp.split(r, splits, axis=1)
        for r in jnp.split(r_ee_simple, splits, axis=0)
    ]

    jastrow_value = jnp.asarray(0.0)
    for i in range(n_species):
      for j in range(i, n_species):
        kind = norm_kind[i][j]
        if kind == JastrowType.NONE:
          continue
        cusp = cusp_matrix[i, j]

        if kind == JastrowType.SIMPLE_EE:
          pos = r_ees_simple[i][j]
          if i == j:
            pair_mask = _strict_upper_triangle_mask(nspins[i], pos.dtype)
          else:
            pair_mask = jnp.ones_like(pos)
          alpha = params['simple_alpha'][i, j]
          term = -(cusp * alpha**2) / (alpha + pos)
          jastrow_value += jnp.sum(term * pair_mask)
        elif kind == JastrowType.CUT_EE:
          pos = r_ees_cut[i][j]
          if i == j:
            pair_mask = _strict_upper_triangle_mask(nspins[i], pos.dtype)
          else:
            pair_mask = jnp.ones_like(pos)
          alpha_arr = params['cut_alpha'][i, j]
          alpha0 = alpha_arr[0]
          x = pos / cutoff_length
          linear_coeff = cusp + alpha0 * C
          coeffs = []
          for l in range(poly_order, 1, -1):
            coeffs.append(alpha_arr[l - 1])
          coeffs.append(linear_coeff)
          coeffs.append(alpha0)
          poly = jnp.zeros_like(x)
          for coeff in coeffs:
            poly = coeff + x * poly
          envelope = (1.0 - x) ** C
          cutoff_mask = (x < 1.0).astype(pos.dtype)
          u = cutoff_length * envelope * poly * cutoff_mask * pair_mask
          jastrow_value += jnp.sum(u)

    return jastrow_value

  return init, apply


def get_jastrow(
      jastrow: JastrowType,
      nspins: jnp.ndarray,
      masses: jnp.ndarray,
      charges: jnp.ndarray,
      ndim: int,
      cut_length: float = 1.0,
      poly_order: int = 3,
      C: int = 3,
      pair_kind=None,
  ) -> ...:
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    jastrow_init, jastrow_apply = make_simple_ee_jastrow(
      nspins, masses, charges, ndim)
  elif jastrow == JastrowType.CUT_EE:
    jastrow_init, jastrow_apply = make_cut_ee_jastrow(
        nspins, masses, charges, ndim, cut_length=cut_length, poly_order=poly_order, C=C)
  elif jastrow == JastrowType.MIXED_EE:
    if pair_kind is None:
      raise ValueError('pair_kind must be provided for MIXED_EE Jastrow')
    jastrow_init, jastrow_apply = make_mixed_ee_jastrow(
        nspins, masses, charges, ndim, pair_kind=pair_kind,
        cut_length=cut_length, poly_order=poly_order, C=C)
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

  return jastrow_init, jastrow_apply
