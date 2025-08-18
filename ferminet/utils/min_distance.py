import jax.numpy as jnp

def min_image_distance_cubic(r_ij, lattice_vector):
    """
    Returns the minimum image displacement vectors for batch relative position vectors.
    
    Args:
        r_ij: (N, 3) array of relative position vectors
        lattice_vector: (3, 3) lattice vector matrix, each row is a lattice vector
    
    Returns:
        dr_min: (N, 3) minimum image displacement vectors
        dr_norm: (N,) norms of the minimum image displacement vectors
    """
    lattice_inv = jnp.linalg.inv(lattice_vector)
    ds = jnp.einsum('ij,kj->ik', r_ij, lattice_inv)
    ds = jnp.mod(ds + 0.5, 1) - 0.5

    dr_min = jnp.einsum('ij,kj->ik', ds, lattice_vector)
    dr_norm = jnp.linalg.norm(dr_min, axis=1)
    return dr_min, dr_norm

def min_image_distance_triclinic(r_ij, lattice_vector, radius=1):
    """
    Returns the minimum image displacement vectors for batch relative position vectors
    using triclinic lattice minimum image convention.
    
    Args:
        r_ij: (N, 3) array of relative position vectors
        lattice_vector: (3, 3) lattice vector matrix, each row is a lattice vector
        radius: search radius, enumerate integer offsets in [-radius, radius]^3 
                around the component-wise rounded solution
    
    Returns:
        dr_min: (N, 3) minimum image displacement vectors
        dr_norm: (N,) norms of the minimum image displacement vectors
    """
    if radius == 0:
        return min_image_distance_cubic(r_ij, lattice_vector)
    lattice_inv = jnp.linalg.inv(lattice_vector)
    
    # Convert to fractional coordinates and wrap to primary cell
    ds = jnp.einsum('ij,kj->ik', r_ij, lattice_inv)
    ds = jnp.mod(ds + 0.5, 1) - 0.5

    # Generate all possible integer offsets
    rng = jnp.arange(-radius, radius+1)
    lx, ly, lz = jnp.meshgrid(rng, rng, rng, indexing='ij')
    offsets = jnp.stack([lx, ly, lz], axis=-1).reshape(-1, 3)  # (num_offsets, 3)

    # Compute all candidate displacement vectors
    ds_cand = ds[:, None, :] + offsets[None, :, :]  # (N, num_offsets, 3)
    
    # Convert back to Cartesian coordinates
    dr_cand = jnp.einsum('ijk,lk->ijl', ds_cand, lattice_vector)
    
    # Find minimum distance candidates
    d2 = jnp.sum(dr_cand**2, axis=2)
    k = jnp.argmin(d2, axis=1)
    
    # Extract minimum displacement vectors and their norms
    dr_min = dr_cand[jnp.arange(r_ij.shape[0]), k]
    dr_norm = jnp.sqrt(d2[jnp.arange(r_ij.shape[0]), k])

    return dr_min, dr_norm


def find_neighbors_within_cutoff(r_ij, lattice_vector, rcut, radius=1):
    """
    Returns all image displacement vectors with distances less than rcut.
    
    Args:
        r_ij: (N, 3) array of relative position vectors
        lattice_vector: (3, 3) lattice vector matrix, each row is a lattice vector
        rcut: cutoff distance
        radius: search radius, enumerate integer offsets in [-radius, radius]^3 
                in fractional coordinates
    
    Returns:
        neighbors: (M, 3) all neighbor displacement vectors within cutoff
        distances: (M,) corresponding distances
    """
    lattice_inv = jnp.linalg.inv(lattice_vector)
    
    # Convert to fractional coordinates and wrap to primary cell
    ds = jnp.einsum('ij,kj->ik', r_ij, lattice_inv)
    ds = jnp.mod(ds + 0.5, 1) - 0.5

    # Generate all possible integer offsets
    rng = jnp.arange(-radius, radius+1)
    lx, ly, lz = jnp.meshgrid(rng, rng, rng, indexing='ij')
    offsets = jnp.stack([lx, ly, lz], axis=-1).reshape(-1, 3)  # (num_offsets, 3)

    # Compute all candidate displacement vectors
    ds_cand = ds[:, None, :] + offsets[None, :, :]  # (N, num_offsets, 3)
    dr_cand = jnp.einsum('ijk,lk->ijl', ds_cand, lattice_vector)  # (N, num_offsets, 3)
    
    # Calculate all distances
    distances_all = jnp.linalg.norm(dr_cand, axis=2)  # (N, num_offsets)
    
    # Flatten all candidates
    dr_flat = dr_cand.reshape(-1, 3)  # (N*num_offsets, 3)
    distances_flat = distances_all.flatten()  # (N*num_offsets,)
    mask = distances_flat < rcut
    
    # Extract valid neighbors
    neighbors = dr_flat[mask]
    distances = distances_flat[mask]
    
    return neighbors, distances
