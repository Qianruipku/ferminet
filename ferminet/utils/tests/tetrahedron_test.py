"""
Unit tests for delaunay_tetrahedralization module.

This module contains tests for the tetrahedralization integration algorithm
using analytical functions with known integration results.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from ferminet.utils.tetrahedron import tetra_integration


class TestTetrahedralizationIntegration:
    """Test cases for tetrahedralization integration using analytical functions."""
    
    def test_linear_function_integration(self):
        """
        Test integration with a simple linear function f(x,y,z) = x.
        
        For a linear function f(x,y,z) = x, the integral over a plane at x=x0
        should be the area of the intersection multiplied by x0.
        """
        # Create a simple cubic grid [0,1]^3
        n_points = 8
        x = jnp.array([0, 1, 0, 1, 0, 1, 0, 1])
        y = jnp.array([0, 0, 1, 1, 0, 0, 1, 1])
        z = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
        grid_points = jnp.stack([x, y, z], axis=1)
        
        # Linear function f(x,y,z) = x
        values = x
        
        # Integration direction (x-direction)
        direction = jnp.array([1.0, 0.0, 0.0])
        
        # Plane positions
        qz = jnp.array([0, 0.2, 0.5, 0.8, 1.0])  # Planes at x = 0, 0.2, 0.5, 0.8, 1.0
        
        # Compute integral
        result = tetra_integration(
            grid_points, values, direction, qz
        )

        expected = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])

        # Allow some numerical tolerance (increased for realistic expectations)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert jnp.abs(res - exp) < 1e-4, f"Plane {i}: Expected {exp}, got {res}"
    
    def test_quadratic_function_integration(self):
        """
        Test integration with a quadratic function f(x,y,z) = x^2.
        """
        # Create a finer grid for better accuracy
        n = 20
        x_coords = jnp.linspace(0, 1, n)
        y_coords = jnp.linspace(0, 1, n)
        z_coords = jnp.linspace(0, 1, n)
        
        # Create 3D grid
        X, Y, Z = jnp.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid_points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Quadratic function f(x,y,z) = x^2
        values = grid_points[:, 0]**2
        
        # Integration direction (x-direction)
        direction = jnp.array([1.0, 0.0, 0.0])
        
        # Multiple plane positions
        qz = jnp.linspace(0, 1, 8)
        
        # Compute integrals
        results = tetra_integration(
            grid_points, values, direction, qz
        )
        
        # For f(x,y,z) = x^2 and plane at x = x0:
        # The intersection is a square [0,1] x [0,1] with area = 1
        # The function value at x = x0 is x0^2
        # Expected integral ≈ 1.0 * x0^2 = x0^2
        expected = qz**2
        
        # Check each plane (relaxed tolerance for coarse grid)
        for i, (result, exp) in enumerate(zip(results, expected)):
            assert jnp.abs(result - exp) < 1e-2, f"Plane {i}: Expected {exp}, got {result}"

    def test_constant_function_integration(self):
        """
        Test integration with a constant function f(x,y,z) = c.
        """
        # Create a regular grid
        n = 4
        coords = jnp.linspace(0, 2, n)
        X, Y, Z = jnp.meshgrid(coords, coords, coords, indexing='ij')
        grid_points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Constant function
        constant_value = 3.14
        values = jnp.full(grid_points.shape[0], constant_value)
        
        # Integration direction (x-direction)
        direction = jnp.array([1.0, 0.0, 0.0])
        
        # Plane positions within the domain
        qz = jnp.array([0.5, 1.0, 1.5])
        
        # Compute integrals
        results = tetra_integration(
            grid_points, values, direction, qz
        )
        
        # For a constant function, the integral should be constant_value * intersection_area
        # The intersection area should be approximately (2-0)^2 = 4
        expected_area = 4.0
        expected_integral = constant_value * expected_area
        
        for i, result in enumerate(results):
            assert jnp.abs(result - expected_integral) < 1e-4, f"Plane {i}: Expected ~{expected_integral}, got {result}"

    def test_polynomial_function_integration(self):
        """
        Test integration with a polynomial function f(x,y,z) = x + y + z.
        """
        # Create grid points
        n = 11
        coords = jnp.linspace(0, 1, n)
        X, Y, Z = jnp.meshgrid(coords, coords, coords, indexing='ij')
        grid_points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Polynomial function f(x,y,z) = x + y + z
        values = grid_points[:, 0] + grid_points[:, 1] + grid_points[:, 2]
        
        # Integration direction (x-direction)
        direction = jnp.array([1.0, 0.0, 0.0])
        
        # Plane position
        qz = jnp.linspace(0.9, 1, 10)
        
        # Compute integral
        result = tetra_integration(
            grid_points, values, direction, qz
        )
        # For f(x,y,z) = x + y + z and plane at x = x0:
        # The intersection is [0,1] x [0,1], and on this plane: f = x0 + y + z
        # The average value over the plane is: x0 + 0.5 + 0.5 = x0 + 1.0
        # The area is 1.0, so expected integral ≈ (x0 + 1.0) * 1.0
        expected = qz + 1.0
        for i, (result, expected) in enumerate(zip(result, expected)):
            assert jnp.abs(result - expected) < 1e-4, f"Plane {i}: Expected {expected}, got {result}"

    def test_batch_processing(self):
        """
        Test that batch processing gives the same result as non-batched processing.
        """
        # Create a moderate-sized grid
        n = 4
        coords = jnp.linspace(0, 1, n)
        X, Y, Z = jnp.meshgrid(coords, coords, coords, indexing='ij')
        grid_points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Simple function
        values = grid_points[:, 0] * grid_points[:, 1]
        
        # Integration parameters
        direction = jnp.array([1.0, 0.0, 0.0])
        qz = jnp.linspace(0.1, 0.9, 7)
        
        # Compute without batching
        result_no_batch = tetra_integration(
            grid_points, values, direction, qz, batch_size=None
        )
        
        # Compute with batching
        result_batch = tetra_integration(
            grid_points, values, direction, qz, batch_size=5
        )
        
        # Results should be very close (relaxed tolerance for floating point differences)
        max_diff = jnp.max(jnp.abs(result_no_batch - result_batch))
        assert max_diff < 1e-6, f"Batched and non-batched results differ by {max_diff}"
    
    def test_edge_cases(self):
        """
        Test edge cases and error handling.
        """
        # Minimal valid input (5 points for 3D Delaunay - need at least 5 points)
        grid_points = jnp.array([
            [0, 0, 0],
            [1, 0, 0], 
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]  # Added fifth point
        ])
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        direction = jnp.array([1.0, 0.0, 0.0])
        qz = jnp.array([0.5])
        
        # This should work without error
        result = tetra_integration(
            grid_points, values, direction, qz
        )
        assert result.shape == qz.shape
        
        # Test error cases
        with pytest.raises(ValueError):
            # Mismatched grid_points and values shapes
            tetra_integration(
                grid_points, jnp.array([1.0, 2.0]), direction, qz
            )
        
        with pytest.raises(ValueError):
            # Wrong direction shape
            tetra_integration(
                grid_points, values, jnp.array([1.0, 0.0]), qz
            )
        
        with pytest.raises(ValueError):
            # Too few points (need at least 5 for 3D Delaunay)
            tetra_integration(
                grid_points[:3], values[:3], direction, qz
            )


def run_convergence_test():
    """
    Additional test to demonstrate convergence with grid refinement.
    """
    print("Running convergence test...")
    
    # Test function: f(x,y,z) = x*y
    def test_function(points):
        return points[:, 0] * points[:, 1]
    
    # Analytical solution for f(x,y,z) = x*y integrated over plane x=x0
    # The plane intersection is [0,1] x [0,1], so integral = x0 * ∫∫ y dy dz = x0 * 1 * 0.5 = 0.5*x0
    def analytical_solution(x0):
        return 0.5 * x0
    
    x0 = 3.0/7.0
    qz = jnp.array([x0])
    direction = jnp.array([1.0, 0.0, 0.0])
    
    grid_sizes = [3, 4, 6, 9, 13, 37]
    errors = []
    
    for n in grid_sizes:
        # Create grid
        coords = jnp.linspace(0, 1, n)
        X, Y, Z = jnp.meshgrid(coords, coords, coords, indexing='ij')
        grid_points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        values = test_function(grid_points)
        
        # Compute numerical integral
        result = tetra_integration(
            grid_points, values, direction, qz
        )
        
        # Compute error
        analytical = analytical_solution(x0)
        error = abs(result[0] - analytical)
        errors.append(error)
        
        print(f"Grid size {n}x{n}x{n}: Numerical = {result[0]:.6f}, "
              f"Analytical = {analytical:.6f}, Error = {error:.6f}")
    
    print("Convergence test completed.")
    return errors


if __name__ == "__main__":
    # Run individual tests
    test_instance = TestTetrahedralizationIntegration()
    
    print("Running unit tests...")
    
    try:
        test_instance.test_linear_function_integration()
        print("✓ Linear function test passed")
    except Exception as e:
        print(f"✗ Linear function test failed: {e}")
    
    try:
        test_instance.test_quadratic_function_integration()
        print("✓ Quadratic function test passed")
    except Exception as e:
        print(f"✗ Quadratic function test failed: {e}")
    
    try:
        test_instance.test_constant_function_integration()
        print("✓ Constant function test passed")
    except Exception as e:
        print(f"✗ Constant function test failed: {e}")
    
    try:
        test_instance.test_polynomial_function_integration()
        print("✓ Polynomial function test passed")
    except Exception as e:
        print(f"✗ Polynomial function test failed: {e}")
    
    try:
        test_instance.test_batch_processing()
        print("✓ Batch processing test passed")
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
    
    try:
        test_instance.test_edge_cases()
        print("✓ Edge cases test passed")
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
    
    # Run convergence test
    print("\n" + "="*50)
    run_convergence_test()
    
    print("\nAll tests completed!")
