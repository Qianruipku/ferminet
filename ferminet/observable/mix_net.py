import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp

def make_mix_batch_network_fn(network_fn, cfg):
    """Constructs a batch network function that mixes two sampling methods."""
    sample_type = cfg.mcmc.mix_sample.type
    alpha = cfg.mcmc.mix_sample.alpha
    index=cfg.mcmc.mix_sample.contact_index
    ndim = cfg.system.ndim
    posi_on_elec = cfg.mcmc.mix_sample.posi_on_elec
    n_particles = sum(cfg.system.particles)
    n_electrons = n_particles - 1
    start_idx = n_electrons // 2
    batch_network = jax.vmap(
            network_fn, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )

    def batch_contact_both_network(params, j, positions, spins, atoms, charges):
        pos = positions.reshape((positions.shape[0], -1, ndim))
        posj = pos[:, j ,:]
        posj2 = pos[:, j+start_idx, :]
        poslast = pos[:, -1, :]
        new_pos1 = pos
        new_pos2 = pos
        new_pos3 = pos
        new_pos4 = pos
        new_pos1 = new_pos1.at[:, -1, :].set(posj)
        new_pos2 = new_pos2.at[:, j, :].set(poslast)
        new_pos3 = new_pos3.at[:, -1, :].set(posj2)
        new_pos4 = new_pos4.at[:, j+start_idx, :].set(poslast)
        pos_contact1 = new_pos1.reshape(positions.shape)
        pos_contact2 = new_pos2.reshape(positions.shape)
        pos_contact3 = new_pos3.reshape(positions.shape)
        pos_contact4 = new_pos4.reshape(positions.shape)
        log_contact1 = batch_network(params, pos_contact1, spins, atoms, charges)
        log_contact2 = batch_network(params, pos_contact2, spins, atoms, charges)
        log_contact3 = batch_network(params, pos_contact3, spins, atoms, charges)
        log_contact4 = batch_network(params, pos_contact4, spins, atoms, charges)

        logs = jnp.stack([log_contact1, log_contact2, log_contact3, log_contact4], axis=0)
        max_log = jnp.max(logs, axis=0)
        logs_shifted = logs - max_log
        inside = jnp.mean(jnp.exp(2.0 * logs_shifted), axis=0)
        log_contact = 0.5 * jnp.log(inside) + max_log
        return log_contact

    def batch_contact_network(params, j, positions, spins, atoms, charges):
        pos = positions.reshape((positions.shape[0], -1, ndim))
        if posi_on_elec == 1:
            posj = pos[:, j ,:]
            new_pos = pos.at[:, -1, :].set(posj)
            pos_contact = new_pos.reshape(positions.shape)
            log_contact = batch_network(params, pos_contact, spins, atoms, charges)
        elif posi_on_elec == 0:
            poslast = pos[:, -1, :]
            new_pos = pos.at[:, j, :].set(poslast)
            pos_contact = new_pos.reshape(positions.shape)
            log_contact = batch_network(params, pos_contact, spins, atoms, charges)
        else:
            posj = pos[:, j ,:]
            poslast = pos[:, -1, :]
            new_pos1 = pos
            new_pos2 = pos
            new_pos1 = new_pos1.at[:, -1, :].set(posj)
            new_pos2 = new_pos2.at[:, j, :].set(poslast)
            pos_contact1 = new_pos1.reshape(positions.shape)
            pos_contact2 = new_pos2.reshape(positions.shape)
            log_contact1 = batch_network(params, pos_contact1, spins, atoms, charges)
            log_contact2 = batch_network(params, pos_contact2, spins, atoms, charges)
            max_log = jnp.maximum(log_contact1, log_contact2)
            log_contact1 = log_contact1 - max_log
            log_contact2 = log_contact2 - max_log
            log_contact = 0.5 * jnp.log(0.5 * jnp.exp(2*(log_contact1)) + 0.5 * jnp.exp(2*(log_contact2))) + max_log
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
        def distribution_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            ratio_log = 2.0 * (all_contact_log - log_prob[None, :])
            ratio_log = ratio_log.reshape(-1)
            ratio_prob_log = jnp.ones_like(ratio_log)
            return ratio_log, ratio_prob_log
            
    elif sample_type == 'contact':

        def sample_function(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            log_contact = batch_contact_network(params, index, positions, spins, atoms, charges)
            mix_log_prob = log_contact + 0.5 * jnp.log((1-alpha) * jnp.exp(2 * (log_prob- log_contact)) + alpha)
            return mix_log_prob

        def contact_prob_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            

            prob_index = all_contact_log[index]
            mix_log_prob = 2 * prob_index + jnp.log((1-alpha) * jnp.exp(2 * (log_prob - prob_index)) + alpha)

            ratio_contact_log = 2.0 * all_contact_log - mix_log_prob[None, :]
            ratio_contact_prob = jnp.exp(ratio_contact_log)
            mean_contact_ratio = jnp.mean(ratio_contact_prob)

            ratio_prob_log = 2.0 * log_prob - mix_log_prob
            ratio_prob = jnp.exp(ratio_prob_log)
            mean_prob_ratio = jnp.mean(ratio_prob)

            return jnp.vstack([mean_contact_ratio, mean_prob_ratio])

        def distribution_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            prob_index = all_contact_log[index]
            mix_log_prob = 2 * prob_index + jnp.log((1-alpha) * jnp.exp(2 * (log_prob - prob_index)) + alpha)
            ratio_contact_log = 2.0 * all_contact_log - mix_log_prob[None, :]
            ratio_contact_log = ratio_contact_log.reshape(-1)
            ratio_prob_log = 2.0 * log_prob - mix_log_prob
            ratio_prob_log = ratio_prob_log.reshape(-1)
            return ratio_contact_log, ratio_prob_log

    elif sample_type == 'contact_i':

        def sample_function(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            log_contact = batch_contact_both_network(params, index, positions, spins, atoms, charges)
            mix_log_prob = log_contact + 0.5 * jnp.log((1-alpha) * jnp.exp(2*(log_prob-log_contact)) + alpha)
            return mix_log_prob

        def contact_prob_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            prob_index = batch_contact_both_network(params, index, positions, spins, atoms, charges)
            

            mix_log_prob = 2 * prob_index + jnp.log((1-alpha) * jnp.exp(2 * (log_prob - prob_index)) + alpha)

            ratio_contact_log = 2.0 * prob_index - mix_log_prob
            ratio_contact_prob = jnp.exp(ratio_contact_log)
            mean_contact_ratio = jnp.mean(ratio_contact_prob)

            ratio_prob_log = 2.0 * log_prob - mix_log_prob
            ratio_prob = jnp.exp(ratio_prob_log)
            mean_prob_ratio = jnp.mean(ratio_prob)

            return jnp.vstack([mean_contact_ratio, mean_prob_ratio])

        def distribution_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            prob_index = batch_contact_both_network(params, index, positions, spins, atoms, charges)
            mix_log_prob = 2 * prob_index + jnp.log((1-alpha) * jnp.exp(2 * (log_prob - prob_index)) + alpha)
            ratio_contact_log = 2.0 * prob_index - mix_log_prob
            ratio_contact_log = ratio_contact_log.reshape(-1)
            ratio_prob_log = 2.0 * log_prob - mix_log_prob
            ratio_prob_log = ratio_prob_log.reshape(-1)
            return ratio_contact_log, ratio_prob_log
        
    elif sample_type == 'contact_all':

        def sample_function(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            # Compute log of mixture probability in a numerically stable way.
            # p = (1-alpha)*exp(2*log_prob) + alpha * mean_j exp(2*all_contact_log_j)
            A = 2.0 * log_prob
            B = 2.0 * all_contact_log  # shape (n_contacts, batch)
            # log mean_j exp(B_j) = logsumexp(B, axis=0) - log(n_contacts)
            log_mean_B = logsumexp(B, axis=0) - jnp.log(jnp.array(B.shape[0], dtype=B.dtype))
            log_term1 = jnp.log(1.0 - alpha) + A
            log_term2 = jnp.log(alpha) + log_mean_B
            log_p = logsumexp(jnp.stack([log_term1, log_term2], axis=0), axis=0)
            # return 0.5 * log p (log amplitude)
            return 0.5 * log_p

        def contact_prob_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            A = 2.0 * log_prob
            B = 2.0 * all_contact_log
            # log mean_j exp(B_j)
            log_mean_B = logsumexp(B, axis=0) - jnp.log(jnp.array(B.shape[0], dtype=B.dtype))
            log_term1 = jnp.log(1.0 - alpha) + A
            log_term2 = jnp.log(alpha) + log_mean_B
            mix_log = logsumexp(jnp.stack([log_term1, log_term2], axis=0), axis=0)

            ratio_contact_log = 2.0 * all_contact_log - mix_log[None, :]
            ratio_contact_prob = jnp.exp(ratio_contact_log)
            mean_contact_ratio = jnp.mean(ratio_contact_prob)

            ratio_prob_log = 2.0 * log_prob - mix_log
            ratio_prob = jnp.exp(ratio_prob_log)
            mean_prob_ratio = jnp.mean(ratio_prob)

            return jnp.vstack([mean_contact_ratio, mean_prob_ratio])
        def distribution_fn(params, positions, spins, atoms, charges):
            log_prob = batch_network(params, positions, spins, atoms, charges)
            all_contact_log = scan_contact_network(params, positions, spins, atoms, charges)
            A = 2.0 * log_prob
            B = 2.0 * all_contact_log
            log_mean_B = logsumexp(B, axis=0) - jnp.log(jnp.array(B.shape[0], dtype=B.dtype))
            log_term1 = jnp.log(1.0 - alpha) + A
            log_term2 = jnp.log(alpha) + log_mean_B
            mix_log = logsumexp(jnp.stack([log_term1, log_term2], axis=0), axis=0)

            ratio_contact_log = 2.0 * all_contact_log - mix_log[None, :]
            ratio_contact_log = ratio_contact_log.reshape(-1)
            ratio_prob_log = 2.0 * log_prob - mix_log
            ratio_prob_log = ratio_prob_log.reshape(-1)
            return ratio_contact_log, ratio_prob_log
    else:
        raise ValueError(f"Invalid mix_sample type: {sample_type}")

    return sample_function, contact_prob_fn, distribution_fn