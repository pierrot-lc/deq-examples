import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray
from src.solvers.gradient import lstsq_gd, root_lbfgs


@pytest.mark.parametrize(
    "n_iterations, lr, hidden_dim, key",
    [
        (100_000, 0.01, 16, jr.key(0)),
        (100_000, 0.01, 16, jr.key(1)),
    ],
)
def test_lstsq_gd(n_iterations: int, lr: float, hidden_dim: int, key: PRNGKeyArray):
    keys = iter(jr.split(key, 2))
    weight = jr.normal(next(keys), (hidden_dim, hidden_dim))
    x = jr.normal(next(keys), (hidden_dim,))
    b = weight @ x

    def f(z):
        return weight @ z

    x_ = lstsq_gd(f, b, n_iterations, lr)
    assert jnp.allclose(b, weight @ x_, rtol=1e-1, atol=1e-3)


@pytest.mark.parametrize(
    "hidden_dim, n_iterations, key",
    [
        (16, 100, jr.key(1)),
        (64, 100, jr.key(2)),
        (256, 100, jr.key(3)),
    ],
)
def test_root_lbfgs(hidden_dim: int, n_iterations: int, key: PRNGKeyArray):
    model = nn.Linear(hidden_dim, hidden_dim, key=key)

    def f_root(x):
        return model(x) - x

    x = jnp.zeros((hidden_dim,))
    x = root_lbfgs(f_root, x, n_iterations)
    assert jnp.allclose(f_root(x), jnp.zeros_like(x), rtol=1e-4, atol=1e-4)
