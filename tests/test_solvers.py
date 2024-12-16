import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray
from src.solvers import lstsq_gd, neumann_series


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
    y = weight @ x

    def f(z):
        return weight @ z

    assert jnp.allclose(x, lstsq_gd(f, y, n_iterations, lr), rtol=1e-1, atol=1e-4)
