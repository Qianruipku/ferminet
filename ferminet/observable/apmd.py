from ferminet.utils.tetrahedron import tetra_integration, delaunay_tetrahedralization
import jax.numpy as jnp
import numpy as np
import os
import jax

def cal_apmd_1d(
        crystal_direction: str,
        qz: jnp.ndarray,
        grid_points: jnp.ndarray,
        density: jnp.ndarray,
        tetrahedra: jnp.ndarray,
):
    """Compute the APMD observable in 1D."""
    dirlist = []
    if crystal_direction == '100':
        dirlist = [jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]), jnp.array([0.0, 0.0, 1.0])]
    elif crystal_direction == '110':
        dirlist = [jnp.array([1.0, 1.0, 0.0]), jnp.array([1.0, 0.0, 1.0]), jnp.array([0.0, 1.0, 1.0]),
                   jnp.array([-1.0, 1.0, 0.0]), jnp.array([-1.0, 0.0, 0.0]), jnp.array([0.0, -1.0, 1.0])]
    elif crystal_direction == '111':
        dirlist = [jnp.array([1.0, 1.0, 1.0]), jnp.array([-1.0, 1.0, 1.0]), jnp.array([1.0, -1.0, 1.0]),
                   jnp.array([1.0, 1.0, -1.0])]
    else:
        raise ValueError(f"Invalid crystal direction: {crystal_direction}")

    apmd_1d = jnp.zeros_like(qz)
    for direction in dirlist:
        tmp = tetra_integration(
            grid_points=grid_points,
            values=density,
            tetrahedra=tetrahedra,
            direction=direction,
            qz=qz,
        )
        apmd_1d += tmp
    apmd_1d /= len(dirlist)

    return apmd_1d

def write_apmd_1d(
        crystal_direction: str,
        ecut: float,
        dq: float,
        grid_points: jnp.ndarray, #(ntwist, npoints, 3)
        density: jnp.ndarray, #(ntwist, npoints)
        ckpt_save_path: str,
):
    """Write the APMD observable in 1D to a file."""
    filename = f'apmd_1d_{crystal_direction}.txt'
    apmd_file = open(os.path.join(ckpt_save_path, filename), 'w')
    qmax = (jnp.sqrt(2.0 * ecut) + dq) // dq

    qz_full = jnp.arange(-qmax, qmax, 1)
    qz_full = qz_full * dq
    
    # Static tetrahedralization for all twists
    tetrahedra, _ = delaunay_tetrahedralization(grid_points[0])
    
    batch_cal_apmd_1d = jax.vmap(
        cal_apmd_1d, 
        in_axes=(None, None, 0, 0, None),
        out_axes=0
    )
    
    apmd_1d_full = batch_cal_apmd_1d(
        crystal_direction,
        qz_full,
        grid_points,
        density,
        tetrahedra
    )
    apmd_1d_full = jnp.mean(apmd_1d_full, axis=0)  # average over twists
    
    # Symmetrize and take positive half
    n_points = len(qz_full)
    n_center = n_points // 2
    qz = qz_full[n_center:]  # positive half including zero
    
    # Average symmetric points for positive q values
    apmd_1d = jnp.zeros_like(qz)
    apmd_1d = apmd_1d.at[0].set(apmd_1d_full[n_center])  # q=0 point
    for i in range(1, len(qz)):
        # Average positive and negative q values
        apmd_1d = apmd_1d.at[i].set((apmd_1d_full[n_center + i] + apmd_1d_full[n_center - i]) / 2.0)
   
    apmd_data = jnp.array([qz, apmd_1d]).T
    np.savetxt(apmd_file, apmd_data, fmt='%.6f')
    apmd_file.close()

