import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray
from src.solvers.anderson import anderson_acceleration, lstsq_qr


@pytest.mark.parametrize(
    "hidden_dim_1, hidden_dim_2, key",
    [
        (100, 1000, jr.key(1)),
        (100, 100, jr.key(2)),
        (10, 100, jr.key(3)),
        (100, 10, jr.key(4)),
    ],
)
def test_lstsq_qr(hidden_dim_1: int, hidden_dim_2: int, key: PRNGKeyArray):
    keys = iter(jr.split(key, 2))
    weight = jr.normal(next(keys), (hidden_dim_2, hidden_dim_1))
    x = jr.normal(next(keys), (hidden_dim_1,))
    b = weight @ x
    Q, R = jnp.linalg.qr(weight)
    x_ = lstsq_qr(Q, R, b)
    assert jnp.allclose(b, weight @ x_, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize(
    "hidden_dim, n_iterations, m, beta, key",
    [
        (16, 50, 3, 0.5, jr.key(1)),
        (64, 100, 5, 0.5, jr.key(2)),
        (256, 200, 5, 0.5, jr.key(3)),
    ],
)
def test_anderson_acceleration(
    hidden_dim: int, n_iterations: int, m: int, beta, key: PRNGKeyArray
):
    keys = iter(jr.split(key, 2))
    model = nn.Sequential(
        [
            nn.Linear(hidden_dim, hidden_dim, key=next(keys)),
            nn.LayerNorm(hidden_dim),
        ]
    )
    x_init = jr.normal(next(keys), (hidden_dim,))

    def f(x):
        return model(x) + x_init

    x = jnp.zeros((hidden_dim,))
    x = anderson_acceleration(f, x, n_iterations=n_iterations, m=m, beta=beta)
    assert jnp.allclose(x, f(x), rtol=1e-3, atol=1e-5)
