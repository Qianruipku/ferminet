import numpy as np
import jax.numpy as jnp
import jax

def make_mix_batch_network_fn(network_fn, cfg):
    """Constructs a batch network function that mixes two sampling methods."""
    sample_type = cfg.mcmc.mix_sample.type
    alpha = cfg.mcmc.mix_sample.alpha
    index=cfg.mcmc.mix_sample.contact_index
    ndim = cfg.system.ndim
    n_particles = sum(cfg.system.particles)
    n_electrons = n_particles - 1
    batch_network = jax.vmap(
            network_fn, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )
    def batch_contact_network(params, j, positions, spins, atoms, charges):
        pos = positions.reshape((positions.shape[0], -1, ndim))
        posj = pos[:, j ,:]
        new_pos = pos.at[:, -1, :].set(posj)
        pos_contact = new_pos.reshape(positions.shape)
        log_contact = batch_network(params, pos_contact, spins, atoms, charges)
        return log_contact
    
    def scan_contact_network(params, positions, spins, atoms, charges):
        def body_fn(carry, j):
            out = batch_contact_network(params, j, positions, spins, atoms, charges)
            return carry, out
        xs = jnp.arange(n_electrons)
        _, outs = jax.lax.scan(body_fn, None, xs)
        return outs

    
    if sample_type == 'none':
        def sample_function(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            return log_prob
        def contact_prob_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            ratio_log = all_contact_log - log_prob[None, :]
            ratio = jnp.exp(2 * ratio_log)
            mean_contact_ratio = jnp.mean(ratio)
            return jnp.array([mean_contact_ratio])
            
    elif sample_type == 'contact':
        lattice_vectors = cfg.system.pbc.lattice_vectors
        inv_volume = 1.0 / np.linalg.det(lattice_vectors) if cfg.system.pbc.apply_pbc else 1.0

        def sample_function(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            pos = positions.reshape((positions.shape[0], -1, ndim))
            pos_index = pos[:, index ,:]
            new_pos = pos.at[:, -1, :].set(pos_index)
            pos_contact = new_pos.reshape(positions.shape)
            log_contact = batch_network(params, pos_contact, spins, atoms, charges)
            mix_prob = (1-alpha) * jnp.exp(2*log_prob) + alpha * jnp.exp(2*log_contact) * inv_volume
            return 0.5 * jnp.log(mix_prob)

        def contact_prob_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            

            prob_index = all_contact_log[index]
            mix_log_prob = 2 * prob_index + jnp.log((1-alpha) * jnp.exp(2 * (log_prob - prob_index)) + alpha * inv_volume)

            ratio_contact_log = 2.0 * all_contact_log - mix_log_prob[None, :]
            ratio_contact_prob = jnp.exp(ratio_contact_log)
            mean_contact_ratio = jnp.mean(ratio_contact_prob)

            ratio_prob_log = 2.0 * log_prob - mix_log_prob
            ratio_prob = jnp.exp(ratio_prob_log)
            mean_prob_ratio = jnp.mean(ratio_prob)

            return jnp.vstack([mean_contact_ratio, mean_prob_ratio])
        
    elif sample_type == 'contact_all':
        lattice_vectors = cfg.system.pbc.lattice_vectors
        inv_volume = 1.0 / np.linalg.det(lattice_vectors) if cfg.system.pbc.apply_pbc else 1.0

        def sample_function(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_prob = scan_contact_network(params, positions, spins, atoms, charges)
            mean_contact_prob = jnp.mean(all_contact_prob, axis=0)
            mix_prob = (1-alpha) * jnp.exp(2*log_prob) + alpha * jnp.exp(2*mean_contact_prob) * inv_volume
            return 0.5 * jnp.log(mix_prob)

        def contact_prob_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            all_contact_prob = jnp.exp(2 * (all_contact_log - log_prob[None, :]))
            mean_contact_prob = jnp.mean(all_contact_prob, axis=0)

            mix_log_prob = 2 * log_prob + jnp.log((1-alpha) + alpha * inv_volume * mean_contact_prob)

            ratio_contact_log = 2.0 * all_contact_log - mix_log_prob[None, :]
            ratio_contact_prob = jnp.exp(ratio_contact_log)
            mean_contact_ratio = jnp.mean(ratio_contact_prob)

            ratio_prob_log = 2.0 * log_prob - mix_log_prob
            ratio_prob = jnp.exp(ratio_prob_log)
            mean_prob_ratio = jnp.mean(ratio_prob)

            return jnp.vstack([mean_contact_ratio, mean_prob_ratio])
    elif sample_type == 'constant':
        log_a = cfg.mcmc.mix_sample.log_alpha
        def sample_function(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            mix_prob = (1-jnp.exp(log_a)) * jnp.exp(2*log_prob-log_a) + 1
            return 0.5 * jnp.log(mix_prob) + 0.5 * log_a

        def contact_prob_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            mix_log_prob = jnp.log((1-jnp.exp(log_a)) * jnp.exp(2 * log_prob - log_a) + 1) + log_a

            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)

            ratio_contact_log = 2.0 * all_contact_log - mix_log_prob[None, :]
            ratio_contact_prob = jnp.exp(ratio_contact_log)
            mean_contact_ratio = jnp.mean(ratio_contact_prob)

            ratio_prob_log = 2.0 * log_prob - mix_log_prob
            ratio_prob = jnp.exp(ratio_prob_log)
            mean_prob_ratio = jnp.mean(ratio_prob)
            return jnp.vstack([mean_contact_ratio, mean_prob_ratio])
    else:
        raise ValueError(f"Invalid mix_sample type: {sample_type}")

    return sample_function, contact_prob_fn