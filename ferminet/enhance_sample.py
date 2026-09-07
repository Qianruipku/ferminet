import jax.numpy as jnp
import jax
from ferminet.utils.min_distance import Lattice
from ferminet.utils.min_distance import min_image_distance_triclinic
def make_enhance_sample_fn(logabs_network, lat: Lattice, cfg):
  """Make a function that enhances the sample distribution.
    P = |psi|^2 * (1 + \sum_i factor * exp(-(r_{e_i, p} / rcut)^2))
  """
  factor = cfg.mcmc.enhance.factor
  rcut = cfg.mcmc.enhance.rcut

  def enhance_factor_fn(positions):
    """Compute the enhancement factor for the sample distribution."""
    positions = jnp.reshape(positions, (-1, 3))
    pos_positron = positions[-1]
    pos_electrons = positions[:-1]
    dr = pos_electrons - pos_positron
    _, dr_norm = min_image_distance_triclinic(dr, lat)
    enhance_exp = jnp.exp(-(dr_norm / rcut) ** 2)
    enhance_factor = 1.0 + factor * jnp.sum(enhance_exp)
    return enhance_factor

  def enhance_sample(params, positions, spins, atoms, charges):
    """Enhance the sample distribution."""
    logabs_psi = logabs_network(params, positions, spins, atoms, charges)
    enhance_factor = enhance_factor_fn(positions)
    logabs_psi_enhanced = logabs_psi + 0.5 * jnp.log(enhance_factor)
    return logabs_psi_enhanced
  
  def batch_enhance_sample(params, positions, spins, atoms, charges):
    vmap_enhance_sample = jax.vmap(enhance_sample, in_axes=(None, 0, 0, 0, 0))
    return vmap_enhance_sample(params, positions, spins, atoms, charges)

  def batch_inverse_enhance_sample(positions):
    """Inverse the enhancement factor for the sample distribution."""
    def inverse_enhance_sample(positions):
      enhance_factor = enhance_factor_fn(positions)
      inverse_enhance_factor = 1.0 / enhance_factor
      return inverse_enhance_factor
    vmap_inverse_enhance_sample = jax.vmap(inverse_enhance_sample)
    return vmap_inverse_enhance_sample(positions)
    
  return batch_enhance_sample, batch_inverse_enhance_sample