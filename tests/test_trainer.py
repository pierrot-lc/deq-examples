import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray
from src.trainer import Trainer


@pytest.mark.parametrize(
    "hidden_dim, batch_size, key",
    [
        (128, 256, jr.key(1)),
        (128, 256, jr.key(2)),
    ],
)
def test_hutchinson_estimate(hidden_dim: int, batch_size: int, key: PRNGKeyArray):
    keys = iter(jr.split(key, 3))
    model = nn.Linear(hidden_dim, hidden_dim, key=next(keys))
    x = jr.normal(next(keys), (batch_size, hidden_dim))

    # Model is linear so the jac is the same for every points.
    jac = jax.jacobian(model)(jnp.zeros((hidden_dim,)))
    trace = jnp.linalg.trace(jac @ jac.T)

    estimate = jax.vmap(Trainer.hutchinson_estimate, in_axes=(None, 0, 0))(
        model, x, jr.split(next(keys), batch_size)
    )
    estimate = jnp.mean(estimate, axis=0)

    assert jnp.allclose(trace, estimate, rtol=1e-1, atol=1e-1)
