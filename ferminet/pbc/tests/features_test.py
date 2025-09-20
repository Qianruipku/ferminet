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

"""Tests for ferminet.pbc.feature_layer."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import networks
from ferminet.pbc import feature_layer as pbc_feature_layer
import jax
import jax.numpy as jnp
import numpy as np


class FeatureLayerTest(parameterized.TestCase):

  @parameterized.parameters([True, False])
  def test_shape(self, heg):
    """Asserts that output shape of apply matches what is expected by init."""
    nspins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    natom = atoms.shape[0]
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    xs = jax.random.uniform(subkey, shape=(sum(nspins), 3))

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natom, nspins, 3, lattice=jnp.eye(3), include_r_ae=heg
    )

    dims, params = feature_layer.init()
    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features, ee_features = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    assert dims[0] == ae_features.shape[-1]
    assert dims[1] == ee_features.shape[-1]

  def test_periodicity(self):
    nspins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    natom = atoms.shape[0]
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    xs = jax.random.uniform(subkey, shape=(sum(nspins), 3))

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natom, nspins, 3, lattice=jnp.eye(3), include_r_ae=False
    )

    _, params = feature_layer.init()
    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features_1, ee_features_1 = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    # Select random electron coordinate to displace by a random lattice vector
    key, subkey = jax.random.split(key)
    e_idx = jax.random.randint(subkey, (1,), 0, xs.shape[0])
    key, subkey = jax.random.split(key)
    randvec = jax.random.randint(subkey, (3,), 0, 100).astype(jnp.float32)
    xs = xs.at[e_idx].add(randvec)

    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features_2, ee_features_2 = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    atol, rtol = 4.e-3, 4.e-3
    np.testing.assert_allclose(
        ae_features_1, ae_features_2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        ee_features_1, ee_features_2, atol=atol, rtol=rtol)

  def test_put_in_box_identity_lattice(self):
    """Test put_in_box with identity lattice."""
    # Test with identity lattice (unit cube)
    lattice = jnp.eye(3)
    
    # Test positions inside the box [0,1)^3 - should remain unchanged
    r_inside = jnp.array([[0.2, 0.3, 0.4],
                          [0.0, 0.5, 0.9]])
    r_result = pbc_feature_layer.put_in_box(r_inside, lattice)
    np.testing.assert_allclose(r_result, r_inside, atol=1e-6)
    
    # Test positions outside the box - should be wrapped back
    r_outside = jnp.array([[1.2, 0.3, -0.4],   # x > 1, z < 0
                           [0.0, 2.5, 1.9]])    # y > 1, z > 1
    r_expected = jnp.array([[0.2, 0.3, 0.6],   # wrapped back
                            [0.0, 0.5, 0.9]])
    r_result = pbc_feature_layer.put_in_box(r_outside, lattice)
    np.testing.assert_allclose(r_result, r_expected, atol=1e-6)

  def test_put_in_box_flat_input(self):
    """Test put_in_box with flattened input format (like FermiNet positions)."""
    # Test with identity lattice
    lattice = jnp.eye(3)
    
    # Test with flat array representing 2 particles in 3D: [x1,y1,z1,x2,y2,z2]
    positions_flat = jnp.array([1.2, 0.3, -0.4, 0.0, 2.5, 1.9])
    expected_flat = jnp.array([0.2, 0.3, 0.6, 0.0, 0.5, 0.9])
    
    result_flat = pbc_feature_layer.put_in_box(positions_flat.reshape(-1, 3), lattice)
    np.testing.assert_allclose(result_flat.flatten(), expected_flat, atol=1e-6)

  def test_put_in_box_non_cubic_lattice(self):
    """Test put_in_box with non-cubic lattice."""
    # Test with a simple rectangular lattice
    lattice = jnp.array([[2.0, 0.0, 0.0],
                         [0.0, 3.0, 0.0],
                         [0.0, 0.0, 1.0]])
    
    # Position outside the lattice cell
    r = jnp.array([[2.5, 4.0, -0.5]])  # Outside in all dimensions
    r_result = pbc_feature_layer.put_in_box(r, lattice)
    
    # Expected: wrapped back into [0,2) x [0,3) x [0,1)
    r_expected = jnp.array([[0.5, 1.0, 0.5]])
    np.testing.assert_allclose(r_result, r_expected, atol=1e-6)

  def test_put_in_box_periodicity(self):
    """Test that put_in_box is periodic - shifting by lattice vectors should give same result."""
    lattice = jnp.array([[1.5, 0.2, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.1, 0.0, 1.2]])
    
    # Original position
    r_orig = jnp.array([[0.3, 0.7, 0.4]])
    
    # Add lattice vectors (should wrap to same position)
    r_shifted = r_orig + lattice[:, 0] + 2 * lattice[:, 1] - lattice[:, 2]
    r_shifted = r_shifted.reshape(1, -1)
    
    result_orig = pbc_feature_layer.put_in_box(r_orig, lattice)
    result_shifted = pbc_feature_layer.put_in_box(r_shifted, lattice)
    
    np.testing.assert_allclose(result_orig, result_shifted, atol=1e-6)

if __name__ == '__main__':
  absltest.main()
