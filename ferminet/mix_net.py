# Copyright 2026 The FermiNet Authors.
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

"""Mixed target/contact sampling functions for electron-positron systems."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def make_mix_batch_network_fn(network_fn, cfg):
  """Builds mixed-sampling and importance-ratio functions.

  `network_fn` must return log|psi| and accept the twist vector as its last
  argument. The final particle is treated as the positron.
  """
  sample_type = cfg.mcmc.mix_sample.type
  alpha = cfg.mcmc.mix_sample.alpha
  index = cfg.mcmc.mix_sample.contact_index
  ndim = cfg.system.ndim
  ntwist = cfg.system.pbc.twist_vectors.shape[0]
  n_particles = sum(cfg.system.particles)
  n_electrons = n_particles - 1
  posi_on_elec = cfg.mcmc.mix_sample.posi_on_elec
  if sample_type not in ('none', 'contact', 'contact_i', 'contact_all'):
    raise ValueError(f'Invalid mix_sample type: {sample_type}')
  if sample_type != 'none':
    if not 0.0 < alpha < 1.0:
      raise ValueError('mix_sample.alpha must be strictly between 0 and 1.')
    if len(cfg.system.particles) != 3 or cfg.system.particles[-1] != 1:
      raise ValueError(
          'Contact sampling requires two electron spin groups and one '
          'positron as the final particle.')
    if posi_on_elec not in (0, 1, 2):
      raise ValueError('mix_sample.posi_on_elec must be 0, 1, or 2.')
    if sample_type == 'contact' and not 0 <= index < n_electrons:
      raise ValueError(
          'mix_sample.contact_index must identify an electron.')
    if sample_type == 'contact_i':
      n_alpha, n_beta = cfg.system.particles[:2]
      if n_alpha != n_beta or not 0 <= index < n_alpha:
        raise ValueError(
            'contact_i requires equal electron spin groups and a '
            'contact_index in the first spin group.')

  batch_network = jax.vmap(
      network_fn, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)

  def contact_log_prob(params, j, positions, spins, atoms, charges, twist):
    pos = positions.reshape((positions.shape[0], n_particles, ndim))
    electron = pos[:, j, :]
    positron = pos[:, -1, :]
    if posi_on_elec == 1:
      contact_pos = pos.at[:, -1, :].set(electron)
      return batch_network(
          params, contact_pos.reshape(positions.shape), spins, atoms, charges,
          twist)
    if posi_on_elec == 0:
      contact_pos = pos.at[:, j, :].set(positron)
      return batch_network(
          params, contact_pos.reshape(positions.shape), spins, atoms, charges,
          twist)
    first = pos.at[:, -1, :].set(electron)
    second = pos.at[:, j, :].set(positron)
    log_first = batch_network(
        params, first.reshape(positions.shape), spins, atoms, charges, twist)
    log_second = batch_network(
        params, second.reshape(positions.shape), spins, atoms, charges, twist)
    return 0.5 * logsumexp(
        jnp.stack((2.0 * log_first, 2.0 * log_second), axis=0), axis=0
    ) - 0.5 * jnp.log(2.0)

  def contact_log_prob_i(params, j, positions, spins, atoms, charges, twist):
    n_alpha = cfg.system.particles[0]
    log_probs = jnp.stack([
        contact_log_prob(params, j, positions, spins, atoms, charges, twist),
        contact_log_prob(
            params, j + n_alpha, positions, spins, atoms, charges, twist),
    ])
    return 0.5 * logsumexp(2.0 * log_probs, axis=0) - 0.5 * jnp.log(2.0)

  def all_contact_log_probs(params, positions, spins, atoms, charges, twist):
    def body_fn(_, j):
      return None, contact_log_prob(
          params, j, positions, spins, atoms, charges, twist)

    _, contacts = jax.lax.scan(body_fn, None, jnp.arange(n_electrons))
    return contacts

  def selected_contact_log_prob(
      params, positions, spins, atoms, charges, twist):
    if sample_type == 'contact':
      return contact_log_prob(
          params, index, positions, spins, atoms, charges, twist)
    if sample_type == 'contact_i':
      return contact_log_prob_i(
          params, index, positions, spins, atoms, charges, twist)
    contacts = all_contact_log_probs(
        params, positions, spins, atoms, charges, twist)
    log_mean_density = (
        logsumexp(2.0 * contacts, axis=0) -
        jnp.log(jnp.asarray(n_electrons, dtype=contacts.dtype)))
    return 0.5 * log_mean_density

  def mixture_log_density(log_prob, log_contact):
    return logsumexp(jnp.stack((
        jnp.log1p(-alpha) + 2.0 * log_prob,
        jnp.log(alpha) + 2.0 * log_contact,
    )), axis=0)

  def mean_per_twist(values, contact_axis=False):
    if contact_axis:
      values = values.reshape((values.shape[0], ntwist, -1))
      return jnp.mean(values, axis=(0, 2))
    values = values.reshape((ntwist, -1))
    return jnp.mean(values, axis=1)

  def sample_function(params, positions, spins, atoms, charges, twist):
    log_prob = batch_network(params, positions, spins, atoms, charges, twist)
    if sample_type == 'none':
      return log_prob
    log_contact = selected_contact_log_prob(
        params, positions, spins, atoms, charges, twist)
    return 0.5 * mixture_log_density(log_prob, log_contact)

  def contact_prob_fn(params, positions, spins, atoms, charges, twist):
    log_prob = batch_network(params, positions, spins, atoms, charges, twist)
    if sample_type == 'none':
      all_contacts = all_contact_log_probs(
          params, positions, spins, atoms, charges, twist)
      ratios = jnp.exp(2.0 * (all_contacts - log_prob[None, :]))
      return mean_per_twist(ratios, contact_axis=True)[None, :]
    log_contact = selected_contact_log_prob(
        params, positions, spins, atoms, charges, twist)
    log_density = mixture_log_density(log_prob, log_contact)
    if sample_type == 'contact_i':
      log_contact_density = 2.0 * log_contact - log_density
    else:
      all_contacts = all_contact_log_probs(
          params, positions, spins, atoms, charges, twist)
      log_contact_density = 2.0 * all_contacts - log_density[None, :]
    contact_ratios = jnp.exp(log_contact_density)
    target_ratios = jnp.exp(2.0 * log_prob - log_density)
    if sample_type == 'contact_i':
      mean_contact_ratio = mean_per_twist(contact_ratios)
    else:
      mean_contact_ratio = mean_per_twist(
          contact_ratios, contact_axis=True)
    mean_target_ratio = mean_per_twist(target_ratios)
    return jnp.stack((mean_contact_ratio, mean_target_ratio))

  def distribution_fn(params, positions, spins, atoms, charges, twist):
    log_prob = batch_network(params, positions, spins, atoms, charges, twist)
    if sample_type == 'none':
      all_contacts = all_contact_log_probs(
          params, positions, spins, atoms, charges, twist)
      contact_log_ratio = 2.0 * (all_contacts - log_prob[None, :])
      return contact_log_ratio.reshape(-1), jnp.zeros_like(log_prob)
    log_contact = selected_contact_log_prob(
        params, positions, spins, atoms, charges, twist)
    log_density = mixture_log_density(log_prob, log_contact)
    if sample_type == 'contact_i':
      contact_log_ratio = 2.0 * log_contact - log_density
    else:
      all_contacts = all_contact_log_probs(
          params, positions, spins, atoms, charges, twist)
      contact_log_ratio = 2.0 * all_contacts - log_density[None, :]
    target_log_ratio = 2.0 * log_prob - log_density
    return contact_log_ratio.reshape(-1), target_log_ratio.reshape(-1)

  return sample_function, contact_prob_fn, distribution_fn
