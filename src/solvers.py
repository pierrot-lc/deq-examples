from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped


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


@jaxtyped(typechecker=beartype)
def lstsq_qr(
    Q: Float[Array, "M K"], R: Float[Array, "K N"], b: Float[Array, " M"]
) -> Float[Array, " N"]:
    """Solve the least-squares problem using an available QR decomposition of the
    problem.

    If the matrix R is not square, it will be padded with the identity matrix before
    solving the problem.

    ---
    Args:
        Q: The orthogonal matrix.
        R: The upper-triangular matrix.
        b: The right-side of the problem.

    ---
    Returns:
        The solution `x` to `Rx = Q.T @ b`.
    """
    if R.shape[0] == R.shape[1]:
        return jsp.linalg.solve_triangular(R, Q.T @ b)

    # R is not square. Multiple solutions are possible so we must provide some
    # additional constraints.
    K, N = R.shape
    (M, _) = Q.shape
    max_ = max(K, N)

    # Pad R to be a square matrix.
    R_ = jnp.eye(max_, dtype=R.dtype)
    R_ = R_.at[:K, :N].set(R)

    # Extend b to match the size of R.
    b = Q.T @ b
    b_ = jnp.zeros((max_,), b.dtype)
    b_ = b_.at[:M].set(b)

    # Solve the triangular problem.
    x_ = jsp.linalg.solve_triangular(R_, b_, lower=False)
    return x_[:N]
