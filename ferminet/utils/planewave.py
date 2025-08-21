# Planewave utilities
import jax.numpy as jnp
import numpy as np

def initgrids(
        lattice_vectors: jnp.ndarray,
        ecut: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize the grid for plane wave calculations.

    Args:
        lattice_vectors: Array of lattice vectors. Shape: (3, 3)
        ecut: Energy cutoff for the plane wave basis set in Hartree (Ha).

    Returns:
        A tuple containing:
        - grid_points: Array of G vectors in reciprocal space. dims: (n_points, 3)
        - g_magnitudes: Array of |G| magnitudes for each G vector. dims: (n_points,)
    """
    # Initialize lattice parameters
    tpiba = 2.0 * jnp.pi
    tpiba2 = tpiba * tpiba
    
    # Convert energy cutoff to lattice units
    # ecut is in Hartree (Ha), convert to lattice units
    # Note: 1 Ha = 2 Ry, so multiply by 2.0 to convert from Ha to Ry units
    gridecut_lat = ecut * 2.0 / tpiba2
    
    # Calculate reciprocal lattice vectors
    # GT = latvec.Inverse(), G = GT.Transpose(), GGT = G * GT
    GT = jnp.linalg.inv(lattice_vectors)  # Inverse of lattice vectors
    G = GT.T  # Transpose
    GGT = G @ GT  # Matrix product
    
    # Calculate initial ibox estimates (vectorized)
    # Following C++ code: ibox[i] = int(sqrt(gridecut_lat) * sqrt(lat * lat)) + 1
    # where lat is the i-th lattice vector
    lat_magnitudes = jnp.sqrt(jnp.sum(lattice_vectors**2, axis=1))  # |a_i|
    
    # Calculate ibox estimates: ibox[i] = int(sqrt(gridecut_lat) * |a_i|) + 1
    ibox_estimates = (jnp.sqrt(gridecut_lat) * lat_magnitudes + 1).astype(int)
    ibox = ibox_estimates
    
    # Create ranges for grid search
    igx_range = jnp.arange(-ibox[0], ibox[0] + 1)
    igy_range = jnp.arange(-ibox[1], ibox[1] + 1)
    igz_range = jnp.arange(-ibox[2], ibox[2] + 1)
    
    # Generate all combinations using meshgrid (vectorized)
    igx_grid, igy_grid, igz_grid = jnp.meshgrid(igx_range, igy_range, igz_range, indexing='ij')
    
    # Flatten and stack to get all grid indices
    grid_indices = jnp.stack([
        igx_grid.flatten(),
        igy_grid.flatten(), 
        igz_grid.flatten()
    ], axis=1).astype(float)
    
    # Calculate |G|^2 in lattice units for all points at once (vectorized)
    # modulus = f^T * GGT * f for each row f in grid_indices
    modulus = jnp.sum(grid_indices * (grid_indices @ GGT), axis=1)
    
    # Find valid points using vectorized comparison
    valid_mask = modulus <= gridecut_lat
    valid_indices = grid_indices[valid_mask]
    
    # Convert to Cartesian coordinates for all valid points at once
    if valid_indices.shape[0] > 0:
        grid_points = (valid_indices @ GT) * tpiba
        # Calculate |G| magnitudes for each G vector
        g_square = jnp.sum(grid_points**2, axis=1)
        
        # Sort by g_square from small to large
        sort_indices = jnp.argsort(g_square)
        grid_points = grid_points[sort_indices]
        g_square = g_square[sort_indices]
    else:
        grid_points = jnp.empty((0, 3))
        g_square = jnp.empty((0,))
    
    return grid_points, g_square