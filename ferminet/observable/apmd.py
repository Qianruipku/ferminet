from ferminet.utils.tetrahedron import tetra_integration
import jax.numpy as jnp
import numpy as np
import os

def cal_apmd_1d(
        crystal_direction: str,
        qz: jnp.ndarray,
        grid_points: jnp.ndarray,
        density: jnp.ndarray,
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
        grid_points: jnp.ndarray,
        density: jnp.ndarray,
        ckpt_save_path: str,
):
    """Write the APMD observable in 1D to a file."""
    filename = f'apmd_1d_{crystal_direction}.txt'
    apmd_file = open(os.path.join(ckpt_save_path, filename), 'w')
    qmax = jnp.sqrt(2.0 * ecut)
    qz = jnp.arange(0.0, qmax + dq, dq)
    apmd_1d = cal_apmd_1d(
        crystal_direction=crystal_direction,
        qz=qz,
        grid_points=grid_points,
        density=density
    )
   
    apmd_data = jnp.array([qz, apmd_1d]).T
    np.savetxt(apmd_file, apmd_data, fmt='%.6f')
    apmd_file.close()

