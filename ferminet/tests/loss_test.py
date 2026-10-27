"""Tests for ferminet.loss."""

from absl.testing import absltest
from ferminet import loss
from ferminet import networks
import jax
import jax.numpy as jnp


def _network(params, positions, spins, atoms, charges):
  del spins, atoms, charges
  return params * positions[0]


def _local_energy(params, key, data):
  del key
  return params * data.positions[0], jnp.array(0.0)


class LossTest(absltest.TestCase):

  def _run(self, positions, target_e, **kwargs):
    n = positions.shape[0]
    data = networks.FermiNetData(
        positions=positions[:, None],
        spins=jnp.zeros((n, 0)),
        atoms=jnp.zeros((n, 0)),
        charges=jnp.zeros((n, 0)))
    evaluate_loss = loss.make_loss(
        _network, _local_energy, target_e=target_e, **kwargs)
    return jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)(
        jnp.array(2.0), jax.random.PRNGKey(0), data)

  def test_target_energy_objective_and_gradient(self):
    x = jnp.array([1.0, 2.0, -1.0])
    target_e = 0.5
    (value, aux_data), gradient = self._run(x, target_e)

    local_energy = 2.0 * x
    residual = local_energy - target_e
    squared = residual**2
    expected = (2.0 * jnp.mean((squared - jnp.mean(squared)) * x)
                + 2.0 * jnp.mean(residual * x))

    self.assertAlmostEqual(float(value), float(jnp.mean(squared)), places=5)
    self.assertAlmostEqual(
        float(aux_data.energy), float(jnp.mean(local_energy)), places=5)
    self.assertAlmostEqual(float(gradient), float(expected), places=4)

  def test_clipping_only_limits_the_weight(self):
    x = jnp.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.0, 10.0])
    target_e = 0.5
    scale = 1.0
    (_, aux_data), gradient = self._run(
        x, target_e, clip_local_energy=scale, clip_from_median=False,
        center_at_clipped_energy=True)

    e_l = 2.0 * x
    center = jnp.mean(e_l)
    tv = jnp.mean(jnp.abs(e_l - center))
    clipped = jnp.clip(e_l, center - scale * tv, center + scale * tv)
    self.assertFalse(bool(clipped[-1] == e_l[-1]))  # the outlier is clipped

    residual = clipped - target_e
    squared = residual**2
    # The clipped sample keeps contributing through its full dE_L/dparams.
    expected = (2.0 * jnp.mean((squared - jnp.mean(squared)) * x)
                + jnp.mean(2.0 * residual * x))

    self.assertTrue(bool(jnp.all(aux_data.clipped_energy == clipped)))
    self.assertAlmostEqual(float(gradient), float(expected), places=3)

  def test_abs_and_pseudo_huber_gradients(self):
    x = jnp.array([1.0, 2.0, -1.0, 0.3])
    target_e = 0.5
    delta = 0.7
    for name, rho, drho in (
        ('abs', lambda r: r, lambda r: jnp.sign(r)),
        ('pseudo_huber',
         lambda r: jnp.sqrt(delta**2 + r**2) - delta,
         lambda r: r / jnp.sqrt(delta**2 + r**2))):
      (value, _), gradient = self._run(
          x, target_e, target_loss=name, target_delta=delta)
      residual = 2.0 * x - target_e
      per_sample = rho(jnp.abs(residual))
      # d/dparams of E_L = x, and d|X|/dE_L = sign(X).
      expected = (2.0 * jnp.mean((per_sample - jnp.mean(per_sample)) * x)
                  + jnp.mean(drho(jnp.abs(residual)) * jnp.sign(residual) * x))
      self.assertAlmostEqual(float(value), float(jnp.mean(per_sample)),
                             places=5, msg=name)
      self.assertAlmostEqual(float(gradient), float(expected), places=4,
                             msg=name)

  def test_unknown_target_loss_raises(self):
    with self.assertRaises(ValueError):
      loss.make_loss(_network, _local_energy, target_e=0.0, target_loss='x')


if __name__ == '__main__':
  absltest.main()
