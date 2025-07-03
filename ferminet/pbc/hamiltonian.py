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

"""Ewald summation of Coulomb Hamiltonian in periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

import itertools
from typing import Callable, Optional, Sequence, Tuple

import chex
from ferminet import hamiltonian
from ferminet import networks
from ferminet.utils import utils
import jax
import jax.numpy as jnp


def make_ewald_potential_3d(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    atom_charges: jnp.ndarray,
    particle_charges: jnp.ndarray,
    truncation_limit: int = 5,
) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
  """Creates a function to evaluate infinite Coulomb sum for periodic lattice.

  Args:
    lattice: Shape (3, 3). Matrix whose columns are the primitive lattice
      vectors.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    atom_charges: Shape (natoms). Nuclear charges of the atoms.
    truncation_limit: Integer. Half side length of cube of nearest neighbours
      to primitive cell which are summed over in evaluation of Ewald sum.
      Must be large enough to achieve convergence for the real and reciprocal
      space sums.
    include_heg_background: bool. When True, includes cell-neutralizing
      background term for homogeneous electron gas.

  Returns:
    Callable with signature f(ae, ee), where (ae, ee) are atom-electon and
    electron-electron displacement vectors respectively, which evaluates the
    Coulomb sum for the periodic lattice via the Ewald method.
  """
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  volume = jnp.abs(jnp.linalg.det(lattice))
  # the factor gamma tunes the width of the summands in real / reciprocal space
  # and this value is chosen to optimize the convergence trade-off between the
  # two sums. See CASINO QMC manual.
  gamma = (2.8 / volume**(1 / 3))**2
  ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=3)))
  lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals)
  rec_vectors = jnp.einsum('jk,ij->ik', rec, ordinals[1:])
  rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)
  lat_vec_norm = jnp.linalg.norm(lat_vectors[1:], axis=-1)

  madelung_const = (
      jnp.sum(jax.scipy.special.erfc(gamma**0.5 * lat_vec_norm) / lat_vec_norm)
      - 2 * gamma**0.5 / jnp.pi**0.5)
  madelung_const += (
      (4 * jnp.pi / volume) *
      jnp.sum(jnp.exp(-rec_vec_square / (4 * gamma)) / rec_vec_square) -
      jnp.pi / (volume * gamma))

  def real_space_ewald(separation: jnp.ndarray):
    """Real-space Ewald potential between charges seperated by separation."""
    displacements = jnp.linalg.norm(
        separation - lat_vectors, axis=-1)  # |r - R|
    return jnp.sum(
        jax.scipy.special.erfc(gamma**0.5 * displacements) / displacements)

  def recp_space_ewald(separation: jnp.ndarray):
    """Returns reciprocal-space Ewald potential between charges."""
    return (4 * jnp.pi / volume) * jnp.sum(
        jnp.exp(1.0j * jnp.dot(rec_vectors, separation)) *
        jnp.exp(-rec_vec_square / (4 * gamma)) / rec_vec_square)

  def ewald_sum(separation: jnp.ndarray):
    """Evaluates combined real and reciprocal space Ewald potential."""
    return (real_space_ewald(separation) + recp_space_ewald(separation) -
            jnp.pi / (volume * gamma))

  batch_ewald_sum = jax.vmap(ewald_sum, in_axes=(0,))

  def potential(positions: jnp.ndarray, nspins: jnp.ndarray):
    """Callable which returns the Ewald potential

    Args:
      positions: electron positions, shape (nelectrons * ndim, )
    """

    # Make sure all distances are a within the same unit cell
    phase_particles = jnp.einsum('il,jl->ji', rec / (2 * jnp.pi), positions.reshape(-1, 3))
    phase_particles = phase_particles % 1
    positions = jnp.einsum('il,jl->ji', lattice, phase_particles)

    # Treat all particles equally, perform the Ewald sum symmetrically
    all_positions = jnp.concatenate([atoms, positions])
    all_charges = jnp.concatenate([atom_charges, jnp.repeat(particle_charges, nspins)])

    diff_pos = all_positions[:, None, ...] - all_positions[None, :, ...]
    charge_prod = all_charges[:, None] * all_charges[None, :]
    n = all_positions.shape[0]

    sum_charges_squared = jnp.sum(all_charges**2)

    # Remove diagonal elements and flatten to pass to batch_ewald_sum
    diff_pos = utils.remove_diagonal(diff_pos)
    charge_prod = utils.remove_diagonal(charge_prod)

    pot = 0.5 * (jnp.sum(charge_prod * batch_ewald_sum(diff_pos)) +
                 madelung_const * sum_charges_squared)
    
    return jnp.real(pot)

  return potential


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    particle_charges: Sequence[int],
    particle_masses: Sequence[float],
    ndim: int = 3,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    states: int = 0,
    state_specific: bool = False,
    pp_type: str = 'ccecp',
    pp_symbols: Sequence[str] | None = None,
    lattice_vectors: Optional[jnp.ndarray] = None,
    convergence_radius: int = 5,
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function in periodic boundary conditions.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian
    states: Number of excited states to compute. If 0, compute ground state with
      default machinery. If 1, compute ground state with excited state machinery
    state_specific: Only used for excited states (states > 0). If true, then
      the local energy is computed separately for each output from the network,
      instead of the local energy matrix being computed.
    pp_type: type of pseudopotential to use. Only used if ecp_symbols is
      provided.
    pp_symbols: sequence of element symbols for which the pseudopotential is
      used.
    lattice_vectors: Shape (ndim, ndim). Matrix of lattice vectors. Default: identity
      matrix.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  if states:
    raise NotImplementedError('Excited states not implemented with PBC.')
  
  if ndim != 3:
    raise NotImplementedError(f'{ndim}-dimensional Ewald summation not implemented')
  else:
    ewald_function = make_ewald_potential_3d
  
  if pp_symbols is not None:
    raise NotImplementedError("Pseudopotentials not implemented for PBCs")

  if lattice_vectors is None:
    lattice_vectors = jnp.eye(3)

  nspins = jnp.asarray(nspins)
  particle_charges = jnp.asarray(particle_charges)
  particle_masses = jnp.asarray(particle_masses)

  ke = hamiltonian.local_kinetic_energy(f,
                                        nspins=nspins,
                                        particle_masses=particle_masses,
                                        use_scan=use_scan,
                                        complex_output=complex_output,
                                        laplacian_method=laplacian_method,
                                        ndim=ndim)

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    del key  # unused
    potential_energy = ewald_function(
        lattice_vectors, data.atoms, charges, particle_charges, convergence_radius
    )
    potential = potential_energy(data.positions, nspins)
    kinetic = ke(params, data)
    return potential + kinetic, None

  return _e_l
