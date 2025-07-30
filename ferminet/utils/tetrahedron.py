# Delaunay tetrahedralization utilities
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import SphericalVoronoi, Delaunay
from typing import Tuple, Optional
import warnings


def delaunay_tetrahedralization(
    grid_points: jnp.ndarray,
    incremental: bool = False,
    qhull_options: Optional[str] = "QJ",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform Delaunay tetrahedralization on 3D grid points.
    
    Args:
        grid_points: Array of 3D grid points. Shape: (n_points, 3)
        qhull_options: Options for Qhull (default: "QJ" - jitter points to avoid degeneracies)
        incremental: Whether to use incremental construction (default: False)
    
    Returns:
        A tuple containing:
        - tetrahedra: Array of tetrahedra indices. Shape: (n_tetrahedra, 4)
        - neighbors: Array of neighbor tetrahedra for each tetrahedron. Shape: (n_tetrahedra, 4)
    """
    if grid_points.shape[0] < 4:
        raise ValueError("Need at least 4 points for 3D Delaunay tetrahedralization")
    if grid_points.shape[1] != 3:
        raise ValueError("Grid points must be 3D (shape: (n_points, 3))")
    
    # Convert JAX array to numpy for scipy compatibility
    points_np = np.asarray(grid_points)
    
    # Perform Delaunay triangulation
    tri = Delaunay(points_np, incremental=incremental, qhull_options=qhull_options)

    # Convert results back to JAX arrays
    tetrahedra = jnp.array(tri.simplices)
    neighbors = jnp.array(tri.neighbors)
    
    return tetrahedra, neighbors


def integrate_over_single_tetrahedra(
    vertices: jnp.ndarray,
    values: jnp.ndarray,
    direction: jnp.ndarray,
    qz: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the intersection integral of a single tetrahedron with multiple planes.
    
    This function implements the tetrahedralization method described by Matsumoto et al. 
    [J. Phys. Soc. Jpn. 73, 1870 (2004)] for computing integrals over tetrahedron-plane 
    intersections. The tetrahedron is divided into three regions (I, II, III) based on 
    the plane position relative to sorted vertex projections along the given direction.
    
    Args:
        vertices: Coordinates of the 4 tetrahedron vertices, shape (4, 3)
        values: Values at the 4 vertices, shape (4,)
        direction: Projection direction vector, shape (3,), e.g., [1, 0, 0]
        qz: Projection coordinates array, positions of the planes along the direction, shape (n_planes,)
    
    Returns:
        Integral values array: intersection_area × interpolated_value, shape (n_planes,)
    """
    # Ensure direction vector is normalized
    direction = direction / jnp.linalg.norm(direction)
    
    # Compute projections of all 4 vertices onto the direction
    projections = jnp.dot(vertices, direction)  # shape (4,)

    # Sort vertices and values by projection coordinates
    sorted_indices = jnp.argsort(projections)
    vertices = vertices[sorted_indices]
    values = values[sorted_indices]
    projections = projections[sorted_indices]
    
    # Extract sorted coordinates and vertices (Matsumoto et al. notation)
    # Tetrahedron sorted by projection: q0 <= q1 <= q2 <= q3
    q0, q1, q2, q3 = projections[0], projections[1], projections[2], projections[3]
    v1, v4, v5, v8 = vertices[0], vertices[1], vertices[2], vertices[3]
    f1, f4, f5, f8 = values[0], values[1], values[2], values[3]

    # Compute S1: cross-sectional area at vertex v4 (q1 position)
    # The cross-section is a triangle formed by v4 and intersection points
    # on edges v1-v5 and v1-v8
    t1_15 = jnp.where(jnp.abs(q2 - q0) > 1e-12, (q1 - q0) / (q2 - q0), 0.0)
    t1_18 = jnp.where(jnp.abs(q3 - q0) > 1e-12, (q1 - q0) / (q3 - q0), 0.0)
    
    v2 = v1 + t1_15 * (v5 - v1)  # intersection on edge v1-v5 at q1
    v3 = v1 + t1_18 * (v8 - v1)  # intersection on edge v1-v8 at q1
    f2 = f1 + t1_15 * (f5 - f1)
    f3 = f1 + t1_18 * (f8 - f1)
    
    S1 = _compute_triangle_area(v2, v3, v4)

    # Compute S2: cross-sectional area at vertex v5 (q2 position)
    # The cross-section is a triangle formed by v5 and intersection points
    # on edges v1-v8 and v4-v8
    t2_18 = jnp.where(jnp.abs(q3 - q0) > 1e-12, (q2 - q0) / (q3 - q0), 0.0)
    t2_48 = jnp.where(jnp.abs(q3 - q1) > 1e-12, (q2 - q1) / (q3 - q1), 0.0)
    
    v6 = v1 + t2_18 * (v8 - v1)  # intersection on edge v1-v8 at q2
    v7 = v4 + t2_48 * (v8 - v4)  # intersection on edge v4-v8 at q2
    f6 = f1 + t2_18 * (f8 - f1)
    f7 = f4 + t2_48 * (f8 - f4)
    
    S2 = _compute_triangle_area(v5, v6, v7)

    # Compute geometric parameters for region II
    l34 = v4 - v3
    l56 = v6 - v5
    L1L2sintheta = jnp.linalg.norm(jnp.cross(l34, l56))

    # Vectorized integration over three regions based on plane positions
    # Region masks with degenerate region handling
    mask_region_I = (qz >= q0) & (qz <= q1) & (q1 > q0)
    mask_region_II = (qz > q1) & (qz <= q2) & (q2 > q1)
    mask_region_III = (qz > q2) & (qz <= q3) & (q3 > q2)

    # Initialize result array
    result = jnp.zeros_like(qz)
    
    # Region I: plane between q0 and q1
    result = jnp.where(mask_region_I, 
                      _f(qz, S1, f1, f2, f3, f4, q0, q1), 
                      result)
    
    # Region II: plane between q1 and q2 (most complex case)
    region_II_value = (_f(qz, S1, f5, f4, f3, f2, q2, q1) + 
                      _f(qz, S2, f4, f5, f6, f7, q1, q2) + 
                      _g(qz, L1L2sintheta, f4, f5, f6, f7, q1, q2))
    result = jnp.where(mask_region_II, region_II_value, result)
    
    # Region III: plane between q2 and q3
    result = jnp.where(mask_region_III, 
                      _f(qz, S2, f8, f7, f6, f5, q3, q2), 
                      result)
    
    return result


def _f(qz, S, a, b, c, d, s, t):
    """
    Compute the f-function contribution as defined in Matsumoto et al.
    
    This function calculates the area-weighted integral contribution
    for triangular cross-sections in regions I and III.
    
    Note: Since degenerate regions (t ≈ s) are now excluded at the mask level,
    this function can assume t > s and doesn't need division-by-zero checks.
    """
    return (S * (qz - s)**2 / (3 * (t - s)**3) * 
            ((b + c + d) * (qz - s) - 3 * a * (qz - t)))


def _g(qz, l1l2sintheta, a, b, c, d, s, t):
    """
    Compute the g-function contribution as defined in Matsumoto et al.
    
    This function calculates the additional geometric contribution
    for the complex quadrilateral cross-section in region II.
    
    Note: Since degenerate regions (t ≈ s) are now excluded at the mask level,
    this function can assume t > s and doesn't need division-by-zero checks.
    """
    return ((l1l2sintheta * (qz - s) * (t - qz)) / (2 * (t - s)**3) * 
            ((qz - s) * (c + d) + (t - qz) * (a + b)))


def _compute_triangle_area(a, b, c):
    """
    Compute the area of a triangle in 3D space.
    
    Args:
        a: First triangle vertex, shape (3,)
        b: Second triangle vertex, shape (3,)
        c: Third triangle vertex, shape (3,)
    
    Returns:
        Area of the triangle
    """
    # Compute area using cross product of edge vectors
    ab = b - a
    ac = c - a
    area = jnp.linalg.norm(jnp.cross(ab, ac)) / 2.0
    
    return area


def integrate_over_all_tetrahedra(
    grid_points: jnp.ndarray,
    values: jnp.ndarray,
    tetrahedra: jnp.ndarray,
    direction: jnp.ndarray,
    qz: jnp.ndarray,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """
    Efficiently compute the intersection integral over all tetrahedra with multiple planes.
    
    This function integrates over all tetrahedra in the mesh for multiple plane positions.
    It automatically chooses between full vectorization (for smaller meshes) and 
    batched processing (for larger meshes) to balance performance and memory usage.
    
    Args:
        grid_points: Array of 3D grid points, shape (n_points, 3)
        values: Values at grid points, shape (n_points,)
        tetrahedra: Array of tetrahedra indices, shape (n_tetrahedra, 4)
        direction: Projection direction vector, shape (3,), e.g., [1, 0, 0]
        qz: Projection coordinates array, positions of the planes along the direction, shape (n_planes,)
        batch_size: Number of tetrahedra to process in each batch. If None, automatically determined.
    
    Returns:
        Total integral values array summed over all tetrahedra, shape (n_planes,)
    """
    # Normalize direction vector
    direction = direction / jnp.linalg.norm(direction)
    n_tetrahedra = tetrahedra.shape[0]
    n_planes = qz.shape[0]
    
    # Create a vectorized version of the single tetrahedron integration
    @jax.vmap
    def integrate_single_tet(tet_indices):
        # Extract vertices and values for this tetrahedron
        tet_vertices = grid_points[tet_indices]  # shape (4, 3)
        tet_values = values[tet_indices]  # shape (4,)
        
        return integrate_over_single_tetrahedra(
            tet_vertices, tet_values, direction, qz
        )


    if batch_size is None:
        # Small mesh: use full vectorization for maximum performance
        all_integrals = integrate_single_tet(tetrahedra)
        total_integrals = jnp.sum(all_integrals, axis=0)
    else:
        # Initialize result array
        total_integrals = jnp.zeros(n_planes)
        
        # Process tetrahedra in batches
        for i in range(0, n_tetrahedra, batch_size):
            end_idx = min(i + batch_size, n_tetrahedra)
            batch_tetrahedra = tetrahedra[i:end_idx]
            
            # Apply vectorized integration over current batch
            batch_integrals = integrate_single_tet(batch_tetrahedra)
            
            # Accumulate contributions
            total_integrals += jnp.sum(batch_integrals, axis=0)
    
    return total_integrals


def tetra_integration(
    grid_points: jnp.ndarray,
    values: jnp.ndarray,
    direction: jnp.ndarray,
    qz: jnp.ndarray,
    batch_size: Optional[int] = None,
    incremental: bool = False,
    qhull_options: Optional[str] = "QJ"
) -> jnp.ndarray:
    """
    Complete pipeline for tetrahedralization and plane intersection integration.
    
    This is the main external interface function that performs Delaunay tetrahedralization
    on the input grid points and then computes the intersection integrals with multiple 
    planes along the specified direction.
    
    Args:
        grid_points: Array of 3D grid points, shape (n_points, 3)
        values: Values at grid points, shape (n_points,)
        direction: Projection direction vector, shape (3,), e.g., [1, 0, 0] for x-direction
        qz: Projection coordinates array, positions of the planes along the direction, shape (n_planes,)
        batch_size: Number of tetrahedra to process in each batch. If None, processes all at once.
                   Use this for large meshes to control memory usage.
        incremental: Whether to use incremental Delaunay construction (default: False)
        qhull_options: Options for Qhull algorithm (default: "QJ" - jitter points to avoid degeneracies)
    
    Returns:
        Integral values array corresponding to each plane position in qz, shape (n_planes,)
    """
    # Input validation
    if grid_points.shape[0] != values.shape[0]:
        raise ValueError(
            f"grid_points and values must have same number of points: "
            f"got {grid_points.shape[0]} vs {values.shape[0]}"
        )
    
    if direction.shape != (3,):
        raise ValueError(
            f"direction must be a 3D vector (shape: (3,)), got shape {direction.shape}"
        )
    
    # Step 1: Perform Delaunay tetrahedralization
    tetrahedra, _ = delaunay_tetrahedralization(
        grid_points, 
        incremental=incremental, 
        qhull_options=qhull_options
    )
    
    # Step 2: Compute intersection integrals
    integrals = integrate_over_all_tetrahedra(
        grid_points=grid_points,
        values=values,
        tetrahedra=tetrahedra,
        direction=direction,
        qz=qz,
        batch_size=batch_size
    )
    
    return integrals