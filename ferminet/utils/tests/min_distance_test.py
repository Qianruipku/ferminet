"""Tests for min_distance module."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from ferminet.utils import min_distance
from ferminet.utils.min_distance import Lattice


class MinDistanceTest(parameterized.TestCase):

    def test_cubic_lattice_simple(self):
        """Test with a simple cubic lattice."""
        # Unit cubic lattice
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Test a simple case: vector with distance 0.6
        # Minimum distance should be 0.4 (through periodic boundary condition)
        r_ij = jnp.array([[0.6, 0.0, 0.0]])
        
        dr_min, dr_norm = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector))
        
        # Expected result: should be -0.4 (since 0.6 - 1.0 = -0.4 is shorter)
        expected_dr = jnp.array([[-0.4, 0.0, 0.0]])
        expected_norm = jnp.array([0.4])
        
        np.testing.assert_allclose(dr_min, expected_dr, rtol=1e-6)
        np.testing.assert_allclose(dr_norm, expected_norm, rtol=1e-6)

    def test_cubic_lattice_multiple_vectors(self):
        """Test with multiple vectors in cubic lattice."""
        lattice_vector = jnp.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        # Test multiple vectors
        r_ij = jnp.array([
            [1.2, 0.0, 0.0],  # should become -0.8
            [0.3, 1.7, 0.0],  # should become [0.3, -0.3, 0.0]
            [0.0, 0.0, 1.9],  # should become [0.0, 0.0, -0.1]
        ])
        
        dr_min, dr_norm = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector))
        
        expected_dr = jnp.array([
            [-0.8, 0.0, 0.0],
            [0.3, -0.3, 0.0],
            [0.0, 0.0, -0.1]
        ])
        expected_norm = jnp.linalg.norm(expected_dr, axis=1)
        
        np.testing.assert_allclose(dr_min, expected_dr, rtol=1e-6)
        np.testing.assert_allclose(dr_norm, expected_norm, rtol=1e-6)

    def test_triclinic_lattice(self):
        """Test with a triclinic lattice."""
        # Simple triclinic lattice
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Test a vector
        r_ij = jnp.array([[0.8, 0.0, 0.0]])
        
        dr_min, dr_norm = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector))
        
        # Verify that the result distance is shorter than the original vector
        original_distance = jnp.linalg.norm(r_ij[0])
        
        self.assertLessEqual(dr_norm[0], original_distance)
        # Verify that dr_norm matches the actual norm of dr_min
        np.testing.assert_allclose(dr_norm, jnp.linalg.norm(dr_min, axis=1), rtol=1e-6)

    def test_radius_parameter(self):
        """Test that radius parameter affects the search."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        r_ij = jnp.array([[0.6, 0.0, 0.0]])
        
        # Test with different radius values
        dr_min_1, dr_norm_1 = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector), radius=1)
        dr_min_2, dr_norm_2 = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector), radius=2)
        
        # Results should be the same (since the closest image is within radius=1)
        np.testing.assert_allclose(dr_min_1, dr_min_2, rtol=1e-6)
        np.testing.assert_allclose(dr_norm_1, dr_norm_2, rtol=1e-6)

    def test_zero_vector(self):
        """Test with zero vector."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        r_ij = jnp.array([[0.0, 0.0, 0.0]])
        
        dr_min, dr_norm = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector))
        
        # Zero vector should remain zero
        expected_dr = jnp.array([[0.0, 0.0, 0.0]])
        expected_norm = jnp.array([0.0])
        
        np.testing.assert_allclose(dr_min, expected_dr, rtol=1e-6)
        np.testing.assert_allclose(dr_norm, expected_norm, rtol=1e-6)

    def test_symmetry_property(self):
        """Test that f(-r) = -f(r) for displacement vectors."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        r_ij = jnp.array([[0.3, 0.4, 0.2]])
        r_ij_neg = -r_ij
        
        dr_min_pos, dr_norm_pos = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector))
        dr_min_neg, dr_norm_neg = min_distance.min_image_distance_triclinic(r_ij_neg, Lattice(lattice_vector))
        
        # Should satisfy f(-r) = -f(r) for displacement vectors
        np.testing.assert_allclose(dr_min_neg, -dr_min_pos, rtol=1e-6)
        # Norms should be equal
        np.testing.assert_allclose(dr_norm_neg, dr_norm_pos, rtol=1e-6)

    def test_large_vectors(self):
        """Test with vectors larger than the lattice."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Vectors larger than lattice size
        r_ij = jnp.array([
            [2.3, 0.0, 0.0],  # should become 0.3
            [0.0, 3.7, 0.0],  # should become 0.0, -0.3, 0.0
            [0.0, 0.0, -1.2], # should become 0.0, 0.0, -0.2
        ])
        
        dr_min, dr_norm = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector))
        
        expected_dr = jnp.array([
            [0.3, 0.0, 0.0],
            [0.0, -0.3, 0.0],
            [0.0, 0.0, -0.2]
        ])
        expected_norm = jnp.linalg.norm(expected_dr, axis=1)
        
        np.testing.assert_allclose(dr_min, expected_dr, rtol=1e-6)
        np.testing.assert_allclose(dr_norm, expected_norm, rtol=1e-6)

    def test_batch_consistency(self):
        """Test that batch processing gives same results as individual processing."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Batch processing
        r_ij_batch = jnp.array([
            [0.3, 0.4, 0.2],
            [0.7, 0.1, 0.9],
            [0.2, 0.8, 0.5]
        ])
        
        dr_min_batch, dr_norm_batch = min_distance.min_image_distance_triclinic(r_ij_batch, Lattice(lattice_vector))
        
        # Individual processing
        dr_min_individual = []
        dr_norm_individual = []
        for i in range(r_ij_batch.shape[0]):
            r_single = r_ij_batch[i:i+1]
            dr_single, norm_single = min_distance.min_image_distance_triclinic(r_single, Lattice(lattice_vector))
            dr_min_individual.append(dr_single[0])
            dr_norm_individual.append(norm_single[0])
        
        dr_min_individual = jnp.array(dr_min_individual)
        dr_norm_individual = jnp.array(dr_norm_individual)
        
        # Results should be the same
        np.testing.assert_allclose(dr_min_batch, dr_min_individual, rtol=1e-6)
        np.testing.assert_allclose(dr_norm_batch, dr_norm_individual, rtol=1e-6)

    @parameterized.parameters(
        (1,), (2,), (3,)
    )
    def test_different_radius_values(self, radius):
        """Test with different radius values."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        r_ij = jnp.array([[0.4, 0.3, 0.2]])
        
        dr_min, dr_norm = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector), radius=radius)
        
        # Check that result is reasonable (distance should be <= 0.5 * sqrt(3) for unit cubic lattice)
        self.assertLessEqual(dr_norm[0], 0.5 * jnp.sqrt(3.0) + 1e-6)
        # Verify that dr_norm matches the actual norm of dr_min
        np.testing.assert_allclose(dr_norm, jnp.linalg.norm(dr_min, axis=1), rtol=1e-6)

    def test_norm_consistency(self):
        """Test that returned norms are consistent with displacement vectors."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        r_ij = jnp.array([
            [0.3, 0.4, 0.2],
            [0.7, 0.1, 0.9],
            [1.2, 1.8, 0.5]
        ])
        
        dr_min, dr_norm = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector))
        
        # Computed norms should match actual norms of displacement vectors
        computed_norm = jnp.linalg.norm(dr_min, axis=1)
        np.testing.assert_allclose(dr_norm, computed_norm, rtol=1e-6)

    def test_cubic_vs_triclinic_consistency(self):
        """Test that cubic and triclinic methods give same results for radius=0."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        r_ij = jnp.array([
            [0.3, 0.4, 0.2],
            [0.7, 0.1, 0.9],
        ])
        
        dr_min_cubic, dr_norm_cubic = min_distance.min_image_distance_cubic(r_ij, Lattice(lattice_vector))
        dr_min_triclinic, dr_norm_triclinic = min_distance.min_image_distance_triclinic(r_ij, Lattice(lattice_vector), radius=0)
        
        # Results should be identical
        np.testing.assert_allclose(dr_min_cubic, dr_min_triclinic, rtol=1e-6)
        np.testing.assert_allclose(dr_norm_cubic, dr_norm_triclinic, rtol=1e-6)

    def test_find_neighbors_within_cutoff(self):
        """Test the neighbor finding function."""
        lattice_vector = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        r_ij = jnp.array([
            [0.3, 0.0, 0.0],
            [0.0, 0.4, 0.0],
        ])
        
        rcut = 0.5
        neighbors, distances = min_distance.find_neighbors_within_cutoff(r_ij, Lattice(lattice_vector), rcut)
        
        # All returned distances should be less than rcut
        self.assertTrue(jnp.all(distances < rcut))
        
        # Verify that neighbors and distances have the same length
        self.assertEqual(len(neighbors), len(distances))
        
        # Check that computed distances match the norms of neighbors
        computed_distances = jnp.linalg.norm(neighbors, axis=1)
        np.testing.assert_allclose(distances, computed_distances, rtol=1e-6)


if __name__ == '__main__':
    absltest.main()
