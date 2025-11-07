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

"""Monte Carlo evaluation of quantum observables."""
from typing import Any, Callable, Optional, Tuple

import chex
from ferminet import constants
from ferminet import density
from ferminet import mcmc
from ferminet import networks
from ferminet import network_blocks
from ferminet.utils import scf
from ferminet.utils import planewave
import jax
import jax.numpy as jnp
import kfac_jax
import functools
import ml_collections
import numpy as np
from ferminet.utils.min_distance import min_image_distance_triclinic, Lattice


@chex.dataclass
class DensityState:
  """The state of the MCMC chain used to compute the density matrix.

  Attributes:
    t: time step, not distributed over devices
    positions: walker positions, shape (devices, nelectrons, ndim).
    probabilities: walker probabilities, shape (devices, nelectrons)
    move_width: MCMC move width, shape (devices)
    pmove: probability of move being accepted, shape (update_frequency)
    mo_coeff: coefficients of Scf approximation, needed for checkpointing. Not
      distributed over devices.
  """

  # Similarly to networks.FermiNetData, this dataclass needs to be used for
  # vmap/pmap in_axes arguments, and so the types need to be flexible.
  t: Any  # int
  positions: Any  # array
  probabilities: Any  # array
  move_width: Any  # array
  pmove: Any  # array
  mo_coeff: Any  # array


Observable = Callable[[networks.ParamTree,
                       networks.FermiNetData,
                       Optional[DensityState]],
                      jnp.ndarray]

DensityUpdate = Callable[[Any,
                          networks.ParamTree,
                          networks.FermiNetData,
                          DensityState],
                         DensityState]


def make_observable_fns(fns, pmap: bool = True):
  """Transform batchless functions to functions averaged over a batch.

  Args:
    fns: arbitrary structure of functions with signatures fn(params, x) ->
      results, where params is the network parameters, x is a configuration of
      electron positions, and result is a scalar.
    pmap: if true, also apply a pmap transformation and take the mean over the
      pmap.

  Returns:
    corresponding structure of functions which act on a batch of electron
    configurations and return the mean over the batch.
  """

  def transform(fn):
    data_axes = networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)
    batch_fn = jax.vmap(fn, in_axes=(None, data_axes, 0))
    if pmap:
      def mean_fn(params, data, state):
        batch_vals = batch_fn(params, data, state)
        return constants.pmean(jnp.nanmean(batch_vals, axis=0))

      # only return from first device, since pmean makes them all the same
      return lambda *args: constants.pmap(mean_fn)(*args)[0]
    else:
      return lambda *args: jnp.nanmean(batch_fn(*args), axis=0)

  return jax.tree_util.tree_map(transform, fns)


def make_s2(
    signed_network: networks.FermiNetLike,
    nspins: Tuple[int, ...],
    assign_spin: bool = True,
    states: int = 0,
) -> Observable:
  """Evaluates the S^2 operator of a wavefunction.

  See Wang et al, J. Chem. Phys. 102, 3477 (1995).

  Args:
    signed_network: network callable which takes the network parameters and a
      single (batchless) configuration of electrons and returns the sign and log
      of the network.
    nspins: tuple of spin up and down electrons.
    assign_spin: True if the spin configuration (S_z) is fixed, false if sampled
    states: Number of excited states (0 if doing conventional ground state VMC,
      1 if doing ground state VMC using machinery for excited states).

  Returns:
    callable with same arguments as the network and returns the contribution to
    the Monte Carlo estimate of the S^2 expectation value of the wavefunction at
    the given configuration of electrons.
  """

  def s2_estimator(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: None = None,
  ) -> jnp.ndarray:
    """Returns the S^2 contribution from a single electron configuration x."""
    del state  # only included for consistency in the interface

    if sum(nspins) == 1:  # one-electron case is trivial - always singlet
      return jnp.eye(states) * 0.75 if states else jnp.asarray(0.75)

    if assign_spin:
      # Without loss of generality, N_a >= N_b.
      # <S^2> = (N_a - N_b)/2 ((N_a - N_b)/2 +1) + N_b
      #          + 2 \int \Gamma^{abba}(r_1 r_2|r_1 r_2) dr_1 dr_2
      # For S=M_s, M_s = (N_a - N_b)/2, the first term is the exact value of
      # S^2. The spin contamination is hence the incomplete cancellation of
      # the second and third terms.
      # Compute the contribution to S^2 from the diagonal pair densities.
      na, nb = sorted(nspins, reverse=True)
      s2_diagonal = (na - nb) / 2 * ((na - nb) / 2 + 1) + nb
      s2 = s2_diagonal
    else:
      # Following Löwdin 1955, the action of S^2 on psi with generic spins is
      # given by -N(N-4)/4 psi + \sum_{i<j} psi_ij, where psi_ij is the
      # wavef'n with spins of i and j swapped.
      n = sum(nspins)
      s2 = -n * (n - 4) / 4

    if states:
      state_matrix_network = networks.make_state_matrix(signed_network, states)
      sign_psi, log_psi = state_matrix_network(
          params, data.positions, data.spins, data.atoms, data.charges)
      log_psi_max = jnp.max(log_psi)
      psi = sign_psi * jnp.exp(log_psi - log_psi_max)  # avoid underflow
    else:
      sign_psi, log_psi = signed_network(params, data.positions, data.spins,
                                         data.atoms, data.charges)

    if assign_spin:
      if states:
        # Instead of evaluating local operator E_x[\psi^-1(x) O \psi(x)],
        # evaluate the matrices A_ij = O \psi_i(x_j) and B_ij = \psi_i(x_j) and
        # return E[A @ B^-1], analogous to how the matrix of local energies is
        # computed.
        s2 = s2 * psi  # promote s2 from a scalar to a matrix
        xa, xb = jnp.split(
            jnp.reshape(data.positions, (states, sum(nspins), -1)),
            nspins[:1], axis=-2)
        def _inner(ib, val):
          ia, s2 = val
          xx = xa.at[:, ia].set(xb[:, ib]), xb.at[:, ib].set(xa[:, ia])
          xx = jnp.reshape(jnp.concatenate(xx, axis=1), -1)
          sign_psi_swap, log_psi_swap = state_matrix_network(
              params, xx, data.spins, data.atoms, data.charges)
          # Minus sign from reordering electron positions such that alpha
          # electrons come first.
          # out to be numerically unstable.
          s2 -= sign_psi_swap * jnp.exp(log_psi_swap - log_psi_max)
          return ia, s2
      else:
        # Convert to (nalpha, ndim) and (nbeta, ndim) tensors.
        xa, xb = jnp.split(
            jnp.reshape(data.positions, (sum(nspins), -1)), nspins[:1], axis=-2)
        def _inner(ib, val):
          ia, s2 = val
          xx = xa.at[ia].set(xb[ib]), xb.at[ib].set(xa[ia])
          xx = jnp.reshape(jnp.concatenate(xx), -1)
          sign_psi_swap, log_psi_swap = signed_network(params, xx, data.spins,
                                                       data.atoms, data.charges)
          # Minus sign from reordering electron positions such that alpha
          # electrons come first.
          s2 -= sign_psi * sign_psi_swap * jnp.exp(log_psi_swap - log_psi)
          return ia, s2

      def _outer(ia, s2):
        return jax.lax.fori_loop(0, nspins[1], _inner, (ia, s2))[1]

      s2 = jax.lax.fori_loop(0, nspins[0], _outer, s2)

      if states:
        s2 = jnp.linalg.solve(psi, s2)
    else:
      if states:
        raise NotImplementedError('S^2 estimation for excited states is only '
                                  'implemented for spin-assigned wavefunctions')
      def _inner(ib, val):
        ia, s2 = val
        xx = data.spins.at[ia].set(data.spins[ib])
        xx = xx.at[ib].set(data.spins[ia])
        sign_psi_swap, log_psi_swap = signed_network(
            params, data.positions, xx, data.atoms, data.charges)
        # Unlike in the fixed-spin case, the wavefunction has no privileged
        # ordering of electrons due to their spin.
        s2 += sign_psi * sign_psi_swap * jnp.exp(log_psi_swap - log_psi)
        return ia, s2

      def _outer(ia, s2):
        return jax.lax.fori_loop(0, ia, _inner, (ia, s2))[1]

      s2 = jax.lax.fori_loop(0, n, _outer, s2)

    return s2

  return s2_estimator


def make_dipole(
    signed_network: networks.FermiNetLike,
    states: int = 0,
) -> Observable:
  """Evaluates the dipole moment of a wavefunction.

  Args:
    signed_network: network callable which takes the network parameters and a
      single (batchless) configuration of electrons and returns the sign and log
      of the network.
    states: Number of excited states (0 if doing conventional ground state VMC,
      1 if doing ground state VMC using machinery for excited states).

  Returns:
    callable with same arguments as the network and returns the contribution to
    the Monte Carlo estimate of the dipole moment of the wavefunction at the
    given configuration of electrons.
  """

  def dipole_estimator(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: None = None,
  ) -> jnp.ndarray:
    """Returns the dipole moment from a single electron configuration x."""
    del state  # only included for consistency in the interface
    if states:
      state_matrix_network = networks.make_state_matrix(signed_network, states)
      sign_psi, log_psi = state_matrix_network(
          params, data.positions, data.spins, data.atoms, data.charges)
      log_psi_max = jnp.max(log_psi)
      psi = sign_psi * jnp.exp(log_psi - log_psi_max)  # avoid underflow
      mean_pos = jnp.sum(jnp.reshape(data.positions, (states, -1, 3)), axis=1)
      moment = jnp.linalg.solve(
          psi[None], jnp.einsum('ij,ik->jik', mean_pos, psi))
    else:
      # the dipole moment is trivial in this case - it's just the expected
      # position of an electron, or the center of mass of the electron density
      moment = jnp.sum(jnp.reshape(data.positions, (-1, 3)), axis=0)

    return moment

  return dipole_estimator


def make_density_matrix(
    signed_network: networks.FermiNetLike,
    pos: jnp.ndarray,
    cfg: ml_collections.ConfigDict,
    ckpt_state: Optional[DensityState] = None) -> Tuple[
        DensityState, DensityUpdate, Observable]:
  """Evaluates the density matrix of a wavefunction.

  Args:
    signed_network: network callable which takes the network parameters and a
      single (batchless) configuration of electrons and returns the sign and log
      of the network.
    pos: Position of the electron walkers, used to initialize a parallel chain
      of walkers needed for Monte Carlo estimates of the one-electron reduced
      density matrix.
    cfg: Config for the experiment.
    ckpt_state: Optional state loaded from checkpoint

  Returns:
    The initial state of the parallel MCMC chain, a function to update the chain
    and a function that computes a single sample of the Monte Carlo estimate at
    a given electron position and parallel chain position.
  """

  nstates = cfg.system.states or 1
  device_batch_size = pos.shape[1]

  scf_approx = scf.Scf(
      molecule=cfg.system.molecule,
      restricted=False,  # compute dm for both spins
      nelectrons=cfg.system.electrons,
      basis=cfg.observables.density_basis)
  scf_approx.run()

  if ckpt_state is not None:
    t = ckpt_state.t
    rprime_pos = ckpt_state.positions
    rprime_prob = ckpt_state.probabilities
    move_width = ckpt_state.move_width
    pmove = ckpt_state.pmove
    scf_approx.mo_coeff = ckpt_state.mo_coeff
  else:
    t = 0
    pos = pos.copy().reshape(-1, 3)
    data_shape = (jax.local_device_count(), device_batch_size * nstates)
    idx = np.random.randint(
        low=0,
        high=pos.shape[0] - 1,
        size=np.prod(data_shape),
    )

    # r' positions
    # In the case of excited states, keep the array flat
    rprime_pos = pos[idx].reshape(*data_shape, -1)
    rprime_prob = jnp.ones(rprime_pos.shape[:-1])
    # MCMC move width for r' Monte Carlo sampling
    move_width = kfac_jax.utils.replicate_all_local_devices(jnp.asarray([0.1]))
    pmove = np.zeros(cfg.mcmc.adapt_frequency)

  density_state = DensityState(t=t,
                               positions=rprime_pos,
                               probabilities=rprime_prob,
                               move_width=move_width,
                               pmove=pmove,
                               mo_coeff=scf_approx.mo_coeff)

  rprime_step = density.make_rprime_mcmc_step(
      steps=cfg.mcmc.steps,
      ndim=cfg.system.ndim,
      blocks=1,
      nspins=cfg.system.electrons,
      device_batch_size=device_batch_size * nstates,
      scf_approx=scf_approx,
  )

  if cfg.system.states:
    signed_net = networks.make_state_matrix(signed_network, cfg.system.states)
  else:
    signed_net = signed_network
  batch_signed_net = jax.vmap(
      signed_net, in_axes=(None, 0, 0, 0, 0), out_axes=0,)

  def density_update(key,
                     params: networks.ParamTree,
                     data: networks.FermiNetData,
                     state: DensityState) -> DensityState:
    for _ in range(1000 if state.t == 0 else 1):  # burn-in on the first step
      key, mcmc_key = kfac_jax.utils.p_split(key)

      rprime_data = networks.FermiNetData(
          positions=state.positions,
          spins=data.spins,
          atoms=data.atoms,
          charges=data.charges,
      )

      rprime_data, rprime_probs, rprime_pmove = rprime_step(
          params=params,
          data=rprime_data,
          mcmc_key=mcmc_key,
          mcmc_width=state.move_width)

      move_width, pmoves = mcmc.update_mcmc_width(state.t,
                                                  state.move_width,
                                                  cfg.mcmc.adapt_frequency,
                                                  rprime_pmove,
                                                  state.pmove)
    return DensityState(t=state.t+1,
                        positions=rprime_data.positions,
                        probabilities=rprime_probs,
                        move_width=move_width,
                        pmove=pmoves,
                        mo_coeff=state.mo_coeff)

  def density_estimator(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: DensityState,
  ) -> jnp.ndarray:
    return density.get_rho(
        batch_signed_net,
        params,
        cfg.system.ndim,
        data.positions,
        data.spins,
        data.charges,
        cfg.system.electrons,
        data.atoms,
        state.positions,
        state.probabilities,
        scf_approx)

  return density_state, density_update, density_estimator

def cal_rho_r(
    nspins: Tuple[int, ...],
    ndim: int,
    lim: float,
    nbins: int,
    apply_pbc: bool,
    lattice_vectors: jnp.ndarray,
) -> Tuple[jnp.ndarray, Observable]:
  """Evaluates the single-particle density of each species.

  Args:
    nspins: Tuple containing the number of particles of each species
    ndim: Number of spatial dimensions
    lim: Limits of the binning
    nbins: How many bins to use
    apply_pbc: Whether or not we are on periodic boundary conditions
    lattice_vectors: Array of lattice vectors. Unused if apply_pbc is False

  Returns:
    callable with same arguments as the network and returns the contribution to
    the Monte Carlo estimate of the single-particle density.
  """

  if lattice_vectors is None:
    lattice_vectors = jnp.eye(ndim)

  rec = 2 * jnp.pi * jnp.linalg.inv(lattice_vectors)

  init_state = jnp.zeros((len(nspins),) + (nbins,) * ndim)

  def density_estimator_obc(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the single-particle density from electron configurations x."""
    del params # unused

    data = data.positions.reshape(-1, sum(nspins), ndim)
    inds = network_blocks.array_partitions(nspins)
    data = jnp.split(data, inds, axis = 1)

    for i in range(len(nspins)):
      hist, _ = jnp.histogramdd(data[i].reshape(-1, ndim), 
                                bins = nbins, 
                                range = lim * jnp.tile(jnp.array([-1, 1]), (ndim, 1)))
      state = state.at[i].add(hist)

    return state

  def density_estimator_pbc(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the single-particle density from electron configurations x."""
    del params # unused

    # Make sure all distances are a within the same unit cell
    phase = jnp.einsum('il,wjl->wji', rec / (2 * jnp.pi), data.positions.reshape(-1, sum(nspins), ndim))
    phase = phase % 1
    positions = jnp.einsum('il,wjl->wji', lattice_vectors, phase)

    # Split by particle species
    inds = network_blocks.array_partitions(nspins)
    data = jnp.split(data, inds, axis = 1)

    # Sample the walkers
    for i in range(len(nspins)):
      hist, _ = jnp.histogramdd(data[i], 
                                bins = nbins, 
                                range = lim * jnp.tile(jnp.array([0, 1]), (ndim, 1)))
      state = state.at[i].add(hist)

    return state

  return init_state, density_estimator_obc if not apply_pbc else density_estimator_pbc


def cal_pcf(
    nspins: Tuple[int, ...],
    rmax: float,
    nbins: int,
    elements: int,
    apply_pbc: bool,
    r_search: int,
    lattice_vectors: jnp.ndarray,
) -> Tuple[jnp.ndarray, Observable]:
  """Evaluates the pair distribution function of each species.
  Args:
    nspins: Tuple containing the number of particles of each species
    rmax: Maximum distance to consider
    nbins: How many bins to use
    target_species: Species to compute the pair distribution function for
    apply_pbc: Whether or not we are on periodic boundary conditions
    r_search: Number of nearest neighbors to search for. Unused if apply_pbc is False
    lattice_vectors: Array of lattice vectors. Unused if apply_pbc is False
  Returns:
    callable with same arguments as the network and returns the contribution to
    the Monte Carlo estimate of the pair distribution function.
  """ 
  grids = jnp.linspace(0, rmax, nbins + 1)
  dr = grids[1] - grids[0]
  bin_volume = 4 * jnp.pi / 3.0 * (grids[1:]**3 - grids[:-1]**3)
  init_state = jnp.zeros(nbins)
  lat = Lattice(lattice_vectors)


  def pcf_estimator(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the pair distribution function from electron configurations x."""
    del params  # unused

    n_particles = sum(nspins)
    n_devices_local = data.positions.shape[0]
    pos = data.positions.reshape(n_devices_local, -1, n_particles, 3)
    target_species = (elements + n_particles) % n_particles
    def pos_to_rabs(pos):
      """Computes the absolute distances between the target species and all others."""
      # Compute the distance vectors from the target species to all others
      rvec = jnp.concatenate([pos[:target_species, :], pos[target_species+1:, :]], axis=0) - pos[target_species, :]
      if apply_pbc:
        _, rabs = min_image_distance_triclinic(rvec, lat, radius=r_search)
      else:
        rabs = jnp.linalg.norm(rvec, axis=-1)
      return rabs
    
    batch_pos_to_rabs = jax.vmap(pos_to_rabs, in_axes=0, out_axes=0)
    para_pos_to_rabs = constants.pmap(batch_pos_to_rabs)
    rabs = para_pos_to_rabs(pos)

    rho_0 = (n_particles - 1) / jnp.linalg.det(lattice_vectors) if apply_pbc else 1.0
    nwalker_per_device = pos.shape[1]

    def compute_hist_per_device(rabs_device):
      hist, _ = jnp.histogram(rabs_device.flatten(), bins=nbins, range=(0, rmax))
      hist = hist / bin_volume
      hist /= rho_0 * nwalker_per_device
      hist = constants.pmean(hist)
      return hist
    hist_all = constants.pmap(compute_hist_per_device)(rabs)

    state += hist_all[0]

    return state

  return grids[:-1] + dr / 2, (init_state, pcf_estimator)

def cal_apmd(
    signed_network: networks.FermiNetLike,
    nspins: Tuple[int, ...],
    ecut: float,
    elements: int,
    apply_pbc: bool,
    lattice_vectors: jnp.ndarray,
    complex_output: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, Observable]:
  """Evaluates the annihilating pair moment density.

  Args:
    nspins: Tuple containing the number of particles of each species
    ecut: Energy cutoff for the plane wave basis set
    elements: Species to compute the annihilating pair moment density
    apply_pbc: Whether or not we are on periodic boundary conditions
    lattice_vectors: Array of lattice vectors. Unused if apply_pbc is False

  Returns:
    callable with same arguments as the network and returns the contribution to
    the Monte Carlo estimate of the annihilating pair moment density.
  """

  if not apply_pbc:
    raise NotImplementedError(
        'Annihilating pair moment density is only implemented for periodic boundary '
        'conditions.')

  #Get the grid points - plane wave basis set
  # Use the initgrids function to get plane wave G vectors and their magnitudes
  pwgrids, g_square = planewave.initgrids(lattice_vectors, ecut)
  g_magnitudes = jnp.sqrt(g_square)
  
  # Update the actual number of plane waves found
  n_planewaves = pwgrids.shape[0]
  
  # Initialize state with proper dimensions
  init_state = jnp.zeros((jax.local_device_count(), n_planewaves))
  
  @functools.partial(constants.pmap)
  def apmd_estimator(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the angular pair moment distribution from electron configurations x."""

    n_particles = sum(nspins)
    n_electrons = n_particles - 1  # exclude the positron
    pos = data.positions.reshape(-1, n_particles, 3)
    nwalker_per_device = pos.shape[0]
    # target_species = (elements + n_particles) % n_particles
    target_species = -1

    rdiff = pos[:, :-1, :] - pos[:, -1:, :]

    batch_network = jax.vmap(signed_network, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0))
    phase_psi, log_psi = batch_network(params, data.positions, data.spins, data.atoms, data.charges)

    def loop_electron_real(j, val):
      # For electron j, create modified positions for all walkers
      # double positrons for electron j across all walkers
      pos_dpositron_j = pos.copy()  # [nwalker_per_device, n_particles, 3]
      positron_coords = pos[:, -1, :]  # [nwalker_per_device, 3]
      pos_dpositron_j = pos_dpositron_j.at[:, j, :].set(positron_coords)

      # double electrons for electron j across all walkers  
      pos_delectron_j = pos.copy()  # [nwalker_per_device, n_particles, 3]
      electron_j_coords = pos[:, j, :]  # [nwalker_per_device, 3]
      pos_delectron_j = pos_delectron_j.at[:, -1, :].set(electron_j_coords)

      # Reshape for network input
      pos_dpositron_flat = jnp.reshape(pos_dpositron_j, (nwalker_per_device, n_particles*3))
      pos_delectron_flat = jnp.reshape(pos_delectron_j, (nwalker_per_device, n_particles*3))
      
      # Compute wave functions for all walkers with electron j modified
      sign_psi_dpositron_j, log_psi_dpositron_j = batch_network(params, pos_dpositron_flat, data.spins, data.atoms, data.charges)
      sign_psi_delectron_j, log_psi_delectron_j = batch_network(params, pos_delectron_flat, data.spins, data.atoms, data.charges)
      
      # Calculate factors for all walkers
      sign = sign_psi_dpositron_j * sign_psi_delectron_j
      factor = jnp.exp(log_psi_dpositron_j + log_psi_delectron_j - log_psi*2) * sign
      
      rdiff_j = rdiff[:, j, :]  # [nwalker_per_device, 3]
      # rdiff_j: [nwalker_per_device, 3], pwgrids: [n_planewaves, 3]
      # Compute dot product to get [nwalker_per_device, n_planewaves] matrix
      phase = jnp.dot(rdiff_j, pwgrids.T)  # [nwalker_per_device, n_planewaves]
      
      # Use matrix multiplication: cos(phase) * factor gives [nwalker_per_device, n_planewaves]
      contribution_per_walker = jnp.cos(phase) * factor[:, None]  # [nwalker_per_device, n_planewaves]
      contribution = jnp.sum(contribution_per_walker, axis=0)  # [n_planewaves]

      return val + contribution
    def loop_electron_complex(j, val):
      # For electron j, create modified positions for all walkers
      # double positrons for electron j across all walkers
      pos_dpositron_j = pos.copy()  # [nwalker_per_device, n_particles, 3]
      positron_coords = pos[:, -1, :]  # [nwalker_per_device, 3]
      pos_dpositron_j = pos_dpositron_j.at[:, j, :].set(positron_coords)

      # double electrons for electron j across all walkers  
      pos_delectron_j = pos.copy()  # [nwalker_per_device, n_particles, 3]
      electron_j_coords = pos[:, j, :]  # [nwalker_per_device, 3]
      pos_delectron_j = pos_delectron_j.at[:, -1, :].set(electron_j_coords)

      # Reshape for network input
      pos_dpositron_flat = jnp.reshape(pos_dpositron_j, (nwalker_per_device, n_particles*3))
      pos_delectron_flat = jnp.reshape(pos_delectron_j, (nwalker_per_device, n_particles*3))
      
      # Compute wave functions for all walkers with electron j modified
      phase_psi_dpositron_j, log_psi_dpositron_j = batch_network(params, pos_dpositron_flat, data.spins, data.atoms, data.charges)
      phase_psi_delectron_j, log_psi_delectron_j = batch_network(params, pos_delectron_flat, data.spins, data.atoms, data.charges)
      
      # Calculate factors for all walkers
      psi_phase = phase_psi_dpositron_j + phase_psi_delectron_j - 2 * phase_psi
      factor = jnp.exp(log_psi_dpositron_j + log_psi_delectron_j - log_psi*2) * jnp.exp(1.0j*psi_phase)
      
      rdiff_j = rdiff[:, j, :]  # [nwalker_per_device, 3]
      # rdiff_j: [nwalker_per_device, 3], pwgrids: [n_planewaves, 3]
      # Compute dot product to get [nwalker_per_device, n_planewaves] matrix
      phase = jnp.dot(rdiff_j, pwgrids.T)  # [nwalker_per_device, n_planewaves]
      
      # Use matrix multiplication: cos(phase) * factor gives [nwalker_per_device, n_planewaves]
      contribution_per_walker = jnp.exp(1.0j*phase) * factor[:, None]  # [nwalker_per_device, n_planewaves]
      contribution = jnp.sum(jnp.real(contribution_per_walker), axis=0)  # [n_planewaves]

      return val + contribution

    if complex_output:
      state += constants.pmean(jax.lax.fori_loop(0, n_electrons, loop_electron_complex, jnp.zeros((n_planewaves),))) / (n_electrons * nwalker_per_device)
    else:
      state += constants.pmean(jax.lax.fori_loop(0, n_electrons, loop_electron_real, jnp.zeros((n_planewaves),))) / (n_electrons * nwalker_per_device)

    return state
  return pwgrids, g_magnitudes, (init_state, apmd_estimator)