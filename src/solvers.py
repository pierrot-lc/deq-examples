from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array


def fixed_point_iterations(f: Callable, x: Array, n_iterations: int) -> Array:
    def body_fn(_: Any, x: Array) -> Array:
        return f(x)

    return jax.lax.fori_loop(0, n_iterations, body_fn, init_val=x)


def neumann_series(f: Callable, x: Array, n_iterations: int) -> Array:
    """Compute the Neumann series of f at x.

    ---
    Sources:
        https://en.wikipedia.org/wiki/Neumann_series
    """

    def body_fn(_: Any, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        x, sum = carry
        x = f(x)
        return x, sum + x

    _, x = jax.lax.fori_loop(
        0,
        n_iterations,
        body_fn,
        init_val=(x, x),
    )
    return x


def lstsq_gd(f: Callable, y: Array, n_iterations: int, lr: float) -> Array:
    """Solve the least-squares problem using gradient descent iterations."""
    fT = jax.linear_transpose(f, y)

    def body_fn(_: Any, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        y, x = carry
        (grad,) = fT(f(x) - y)
        return y, x - lr * grad

    _, x = jax.lax.fori_loop(
        0,
        n_iterations,
        body_fn,
        init_val=(y, jnp.zeros_like(y)),
    )
    return x
