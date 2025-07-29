# Copyright 2023 DeepMind Technologies Limited.
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

"""Tests for planewave utilities."""

import unittest
import jax.numpy as jnp
import numpy as np
from ferminet.utils import planewave


class PlanewaveTest(unittest.TestCase):
    """Test planewave utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple cubic lattice for testing
        self.cubic_lattice = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # FCC lattice for more complex testing
        a = 1.0
        self.fcc_lattice = jnp.array([
            [0.0, a/2, a/2],
            [a/2, 0.0, a/2],
            [a/2, a/2, 0.0]
        ])

    def test_initgrids_basic_functionality(self):
        """Test basic functionality of initgrids."""
        ecut = 1.0
        lat0 = 1.0
        
        grid_points, g_magnitudes = planewave.initgrids(self.cubic_lattice, ecut, lat0)
        
        # Check that we get 2D arrays with proper shapes
        self.assertEqual(len(grid_points.shape), 2)
        self.assertEqual(grid_points.shape[1], 3)
        self.assertEqual(len(g_magnitudes.shape), 1)
        self.assertEqual(grid_points.shape[0], g_magnitudes.shape[0])
        
        # Check that we get at least the origin point G=[0,0,0]
        self.assertGreaterEqual(grid_points.shape[0], 1)
        
        # Check that origin point is included
        origin_found = jnp.any(jnp.allclose(grid_points, jnp.array([0.0, 0.0, 0.0]), atol=1e-10))
        self.assertTrue(origin_found)
        
        # Check that g_magnitudes are non-negative
        self.assertTrue(jnp.all(g_magnitudes >= 0))
        
        # Check that g_magnitudes match the computed magnitudes
        computed_magnitudes = jnp.sqrt(jnp.sum(grid_points**2, axis=1))
        self.assertTrue(jnp.allclose(g_magnitudes, computed_magnitudes, atol=1e-10))

    def test_energy_cutoff_enforcement(self):
        """Test that all returned G vectors satisfy the energy cutoff."""
        ecut = 2.0  # Hartree
        lat0 = 1.0
        
        grid_points, g_magnitudes = planewave.initgrids(self.cubic_lattice, ecut, lat0)
        
        # Calculate |G|^2/2 for all points
        g_squared_half = 0.5 * jnp.sum(grid_points**2, axis=1)
        
        # All points should satisfy |G|^2/2 <= ecut
        self.assertTrue(jnp.all(g_squared_half <= ecut + 1e-10))  # Small tolerance for numerical errors
        
        # Check that g_magnitudes are consistent
        computed_magnitudes = jnp.sqrt(jnp.sum(grid_points**2, axis=1))
        self.assertTrue(jnp.allclose(g_magnitudes, computed_magnitudes, atol=1e-10))

    def test_symmetry_properties(self):
        """Test symmetry properties for cubic lattice."""
        ecut = 1.5
        lat0 = 1.0
        
        grid_points, g_magnitudes = planewave.initgrids(self.cubic_lattice, ecut, lat0)
        
        # For cubic lattice, if G is included, then -G should also be included
        # (except for the origin which is its own negative)
        for i, g_vec in enumerate(grid_points):
            if jnp.allclose(g_vec, jnp.zeros(3), atol=1e-10):
                continue  # Skip origin
            
            # Check if -g_vec exists in the grid
            neg_g = -g_vec
            found_negative = False
            for j, other_g in enumerate(grid_points):
                if jnp.allclose(other_g, neg_g, atol=1e-10):
                    found_negative = True
                    break
            
            self.assertTrue(found_negative, 
                          f"G vector {g_vec} found but -G vector {neg_g} not found")

    def test_different_lattice_constants(self):
        """Test with different lattice constants."""
        ecut = 50.0  # Hartree - use larger cutoff to see differences
        
        # Test with different lat0 values
        lat0_values = [0.5, 1.0, 2.0]
        point_counts = []
        
        for lat0 in lat0_values:
            grid_points, g_magnitudes = planewave.initgrids(self.cubic_lattice, ecut, lat0)
            point_counts.append(grid_points.shape[0])
            # Verify magnitude consistency
            computed_magnitudes = jnp.sqrt(jnp.sum(grid_points**2, axis=1))
            self.assertTrue(jnp.allclose(g_magnitudes, computed_magnitudes, atol=1e-10))
        
        # With smaller lat0, the reciprocal lattice vectors become larger,
        # so we should get fewer G vectors for the same energy cutoff
        self.assertGreaterEqual(point_counts[0], 1)  # Should get at least some points
        self.assertGreaterEqual(point_counts[1], 1) 
        self.assertGreaterEqual(point_counts[2], 1)
        
        # lat0=2.0 should give more points than lat0=0.5
        self.assertGreater(point_counts[2], point_counts[0])

    def test_fcc_lattice(self):
        """Test with FCC lattice structure."""
        ecut = 2.0
        lat0 = 1.0
        
        grid_points, g_magnitudes = planewave.initgrids(self.fcc_lattice, ecut, lat0)
        
        # Basic sanity checks
        self.assertEqual(len(grid_points.shape), 2)
        self.assertEqual(grid_points.shape[1], 3)
        self.assertEqual(len(g_magnitudes.shape), 1)
        self.assertEqual(grid_points.shape[0], g_magnitudes.shape[0])
        self.assertGreater(grid_points.shape[0], 0)
        
        # Check energy cutoff
        g_squared_half = 0.5 * jnp.sum(grid_points**2, axis=1)
        self.assertTrue(jnp.all(g_squared_half <= ecut + 1e-10))
        
        # Check magnitude consistency
        computed_magnitudes = jnp.sqrt(jnp.sum(grid_points**2, axis=1))
        self.assertTrue(jnp.allclose(g_magnitudes, computed_magnitudes, atol=1e-10))

    def test_zero_cutoff(self):
        """Test with very small energy cutoff."""
        ecut = 1e-10
        lat0 = 1.0
        
        grid_points, g_magnitudes = planewave.initgrids(self.cubic_lattice, ecut, lat0)
        
        # With very small cutoff, should only get the origin
        self.assertEqual(grid_points.shape[0], 1)
        self.assertEqual(g_magnitudes.shape[0], 1)
        self.assertTrue(jnp.allclose(grid_points[0], jnp.zeros(3), atol=1e-10))
        self.assertTrue(jnp.allclose(g_magnitudes[0], 0.0, atol=1e-10))

    def test_large_cutoff(self):
        """Test with larger energy cutoff."""
        ecut = 50.0  # Hartree - Much larger cutoff to include more G vectors
        lat0 = 1.0
        
        grid_points, g_magnitudes = planewave.initgrids(self.cubic_lattice, ecut, lat0)
        
        # Should get more points than just the origin
        self.assertGreater(grid_points.shape[0], 1)  # At least more than just origin
        self.assertEqual(grid_points.shape[0], g_magnitudes.shape[0])
        
        # All should still satisfy cutoff
        g_squared_half = 0.5 * jnp.sum(grid_points**2, axis=1)
        self.assertTrue(jnp.all(g_squared_half <= ecut + 1e-10))
        
        # Check magnitude consistency
        computed_magnitudes = jnp.sqrt(jnp.sum(grid_points**2, axis=1))
        self.assertTrue(jnp.allclose(g_magnitudes, computed_magnitudes, atol=1e-10))

    def test_reciprocal_lattice_relationship(self):
        """Test the relationship between direct and reciprocal lattice."""
        ecut = 1.0
        lat0 = 1.0
        
        # For simple cubic lattice, reciprocal lattice vectors should be 2π times identity
        expected_reciprocal = 2 * jnp.pi * jnp.eye(3) / lat0
        
        # Calculate reciprocal lattice vectors manually
        GT = jnp.linalg.inv(self.cubic_lattice)
        G = GT.T
        actual_reciprocal = G * (2.0 * jnp.pi / lat0)
        
        self.assertTrue(jnp.allclose(actual_reciprocal, expected_reciprocal, atol=1e-10))

    def test_empty_result_handling(self):
        """Test handling when no valid G vectors exist."""
        # This test is tricky because even with very small cutoff we get origin
        # But we can test the code path by checking the empty array creation
        ecut = 0.0  # Exactly zero
        lat0 = 1.0
        
        grid_points, g_magnitudes = planewave.initgrids(self.cubic_lattice, ecut, lat0)
        
        # Should handle this gracefully and return at least origin
        # (since |G=0|^2/2 = 0 <= 0)
        self.assertGreaterEqual(grid_points.shape[0], 1)
        self.assertEqual(grid_points.shape[0], g_magnitudes.shape[0])
        
        # Check magnitude consistency
        computed_magnitudes = jnp.sqrt(jnp.sum(grid_points**2, axis=1))
        self.assertTrue(jnp.allclose(g_magnitudes, computed_magnitudes, atol=1e-10))


if __name__ == '__main__':
    unittest.main()
