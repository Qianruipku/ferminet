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

"""Feature layer for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

from typing import Optional, Tuple, Sequence

import chex
from ferminet import networks
import jax.numpy as jnp
from ferminet.utils import Lattice


def periodic_norm(metric: jnp.ndarray, scaled_r: jnp.ndarray) -> jnp.ndarray:
  """Returns the periodic norm of a set of vectors.

  Args:
    metric: metric tensor in fractional coordinate system, A.T A, where A is the
      lattice vectors.
    scaled_r: vectors in fractional coordinates of the lattice cell, with
      trailing dimension ndim, to compute the periodic norm of.
  """
  chex.assert_rank(metric, expected_ranks=2)
  a = (1 - jnp.cos(2 * jnp.pi * scaled_r))
  b = jnp.sin(2 * jnp.pi * scaled_r)
  cos_term = jnp.einsum('...m,mn,...n->...', a, metric, a)
  sin_term = jnp.einsum('...m,mn,...n->...', b, metric, b)
  return (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)

def put_in_box(r: jnp.ndarray, lat: Lattice) -> jnp.ndarray:
  """Maps a set of vectors into the periodic box defined by the lattice.
  Args:
    r: vectors in Cartesian coordinates, with trailing dimension ndim, to map
      into the periodic box.
    lattice: Matrix whose columns are the primitive lattice vectors of the
      system, shape (ndim, ndim).
  """
  rshape = r.shape
  r = r.reshape(rshape[:-1] + (-1, 3))
  scaled_r = r @ lat.lattice_inv_T
  r_pbc = (scaled_r % 1) @ lat.lattice_vector_T
  r_pbc = r_pbc.reshape(rshape)
  return r_pbc


def make_pbc_feature_layer(
  natoms: Optional[int] = None,
  nspins: Optional[Tuple[int, ...]] = None,
  ndim: int = 3,
  rescale_inputs: bool = False,
  lattice: jnp.ndarray = jnp.eye(3),
  translation_symm: bool = False,
  primitive_vectors: Optional[jnp.ndarray] = None,
  primitive_atoms_id: Optional[Sequence[int]] = None,
  include_r_ae: bool = True,
  feature_order1: int = 1,
  feature_order2: int = 1,
) -> networks.FeatureLayer:
  """Returns the init and apply functions for periodic features.

  Args:
      natoms: number of atoms.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      rescale_inputs: If true, rescales r_ae for stability. Note that unlike in
        the OBC case, we do not rescale r_ee as well.
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim).
      primitive_vectors: primitive lattice vectors
      include_r_ae: Flag to enable electron-atom distance features. Set to False
        to avoid cusps with ghost atoms in, e.g., homogeneous electron gas.
  """

  del nspins

  # Calculate reciprocal vectors, factor 2pi omitted
  reciprocal_vecs = jnp.linalg.inv(lattice)
  if primitive_vectors is not None:
    primitive_reciprocal = jnp.linalg.inv(primitive_vectors)
  else:
    primitive_reciprocal = reciprocal_vecs

  lattice_metric = lattice.T @ lattice
  lattice_metric_primitive = primitive_vectors.T @ primitive_vectors
  natom_primitive = len(primitive_atoms_id) if primitive_atoms_id is not None else natoms

  def init() -> Tuple[Tuple[int, int], networks.Param]:
    ee_feat_dim = feature_order2 * 2 * ndim + 1
    if not translation_symm:
      ae_feat_dim = feature_order1 * 2 * ndim  * natoms
      if include_r_ae:
        ae_feat_dim += feature_order1 * natoms
    else:
      ae_feat_dim = feature_order1 * 2 * ndim  * natoms + natom_primitive * ndim
      if include_r_ae:
        ae_feat_dim += feature_order1 * natoms + natom_primitive
        
    return (ae_feat_dim, ee_feat_dim), {}

  def apply(ae, r_ae, ee, r_ee, **params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # One e features in phase coordinates, (s_ae)_i = k_i . ae
    s_ae = jnp.einsum('il,jkl->jki', reciprocal_vecs, ae)
    # If primitive reciprocal vectors provided, compute s_ae_primitive
    ae_feats_symm = None
    r_ae_feats_symm = None
    if translation_symm:
      if primitive_atoms_id is not None:
        ae_primitive = ae[:, primitive_atoms_id, :]
      else:
        ae_primitive = ae
      s_ae_primitive = jnp.einsum('il,jkl->jki', primitive_reciprocal, ae_primitive) # (ne, nprim, 3)
      s = jnp.cos(2 * jnp.pi * s_ae_primitive)
      ae_feats_symm = jnp.reshape(s, [jnp.shape(s)[0], -1])  # (ne, nprim*3)
      r_ae_feats_symm = periodic_norm(lattice_metric_primitive, s_ae_primitive)  # (ne, nprim)

    # Two e features in phase coordinates
    s_ee = jnp.einsum('il,jkl->jki', reciprocal_vecs, ee)
    # Periodized features
    ae_feats = []
    for order in range(1, feature_order1 + 1):
      ae_feats.append(jnp.sin(order * 2 * jnp.pi * s_ae))
      ae_feats.append(jnp.cos(order * 2 * jnp.pi * s_ae))
    if ae_feats:
      ae = jnp.concatenate(ae_feats, axis=-1)
    else:
      ae = jnp.zeros(s_ae.shape[:-1] + (0,), dtype=s_ae.dtype)
    ee_feats = []
    for order in range(1, feature_order2 + 1):
      ee_feats.append(jnp.sin(order * 2 * jnp.pi * s_ee))
      ee_feats.append(jnp.cos(order * 2 * jnp.pi * s_ee))
    ee = jnp.concatenate(ee_feats, axis=-1)
    # Distance features defined on orthonormal projections
    r_ae_list = []
    for order in range(1, feature_order1 + 1):
      r_ae_list.append(periodic_norm(lattice_metric, order * s_ae))
    if r_ae_list:
      r_ae = jnp.stack(r_ae_list, axis=-1)
    else:
      r_ae = jnp.zeros(s_ae.shape[:-1] + (0,), dtype=s_ae.dtype)
    if rescale_inputs:
      r_ae = jnp.log(1 + r_ae)
    # Don't take gradients through |0|
    n = ee.shape[0]
    s_ee += jnp.eye(n)[..., None]
    r_ee = periodic_norm(lattice_metric, s_ee) * (1.0 - jnp.eye(n))

    if include_r_ae:
      ae_features = jnp.concatenate((r_ae, ae), axis=2)
    else:
      ae_features = ae
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    if r_ae_feats_symm is not None:
      ae_features = jnp.concatenate((ae_features, r_ae_feats_symm), axis=-1)
    # concatenate symmetry embedding if computed
    if ae_feats_symm is not None:
      ae_features = jnp.concatenate((ae_features, ae_feats_symm), axis=-1)
    ee_features = jnp.concatenate((r_ee[..., None], ee), axis=2)
    return ae_features, ee_features

  return networks.FeatureLayer(init=init, apply=apply)
