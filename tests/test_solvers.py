import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray
from src.solvers import anderson_acceleration, lstsq_gd, lstsq_qr, neumann_series


@pytest.mark.parametrize(
    "order, hidden_dim, key",
    [
        (1, 32, jr.key(0)),
        (0, 32, jr.key(0)),
        (2, 16, jr.key(0)),
        (3, 16, jr.key(0)),
    ],
)
def test_neumann(order: int, hidden_dim: int, key: PRNGKeyArray):
    keys = iter(jr.split(key, 2))
    weight = jr.normal(next(keys), (hidden_dim, hidden_dim))
    x = jr.normal(next(keys), (hidden_dim,))

    def f(z):
        return weight @ z

    y = neumann_series(f, x, order)
    y_true = [jnp.linalg.matrix_power(weight, o) @ x for o in range(1, order + 1)]
    y_true = sum(y_true) + x
    assert jnp.allclose(y, y_true)


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
    "hidden_dim, n_iterations, m, key",
    [
        (16, 100, 3, jr.key(1)),
        (64, 200, 10, jr.key(2)),
        (256, 1_000, 10, jr.key(3)),
    ],
)
def test_anderson_acceleration(
    hidden_dim: int, n_iterations: int, m: int, key: PRNGKeyArray
):
    model = nn.Sequential(
        [
            nn.Linear(hidden_dim, hidden_dim, key=key),
            nn.LayerNorm(hidden_dim),
        ]
    )

    x = jnp.zeros((hidden_dim,))
    x = anderson_acceleration(model, x, n_iterations=n_iterations, m=m)
    assert jnp.allclose(x, model(x), rtol=1e-1, atol=1e-3)
