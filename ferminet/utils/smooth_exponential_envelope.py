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
# limitations under the License.

"""Multiplicative envelope functions."""

import enum
from typing import Any, Mapping, Sequence, Union

import attr
from ferminet import curvature_tags_and_blocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol

from ferminet.envelopes import Envelope, EnvelopeType


def make_smooth_exponential_envelope() -> Envelope:
  """Creates an isotropic smooth exponential multiplicative envelope."""

  def init(
      natom: int, output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del ndim  # unused
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.ones(shape=(natom, output_dim))
      })
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes an isotropic gaussian multiplicative envelope."""
    del ae, r_ee  # unused
    smooth_r = jnp.sqrt(1 + r_ae**2)
    return jnp.sum(jnp.exp(-smooth_r * sigma) * pi, axis=1)

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)
