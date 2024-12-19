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
    Q: Float[Array, "M K"],
    R: Float[Array, "K N"],
    b: Float[Array, " M"],
    lambda_: float = 0.0,
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
        lambda_: Regularizer term to enhance the conditioning of R.

    ---
    Returns:
        The solution `x` to `(R + lambda_ I) x = Q^T b`.
    """
    if lambda_ != 0:
        # How should this be done?
        R = R + jnp.eye(R.shape[0], R.shape[1]) * lambda_

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


def anderson_acceleration(
    f: Callable, x0: Float[Array, " hidden_dim"], n_iterations: int, m: int
) -> Float[Array, " hidden_dim"]:
    """Use Anderson acceleration to find the fixed point of f.

    ---
    Args:
        f: The function for which we want to find the fixed point.
        x0: The initial guess.
        n_iterations: Number of Anderson steps.
        m: Size of the history.

    ---
    Returns:
        The last estimation of the fixed point.

    ---
    Sources:
        See https://en.wikipedia.org/wiki/Anderson_acceleration.
    """
    assert m > 2

    def body_fn(k: int, carry: tuple[Array, Array, Array, Array]) -> tuple:
        X, G, X_k, G_k = carry
        xk, gk = X[k % m], G[k % m]

        gammas, *_ = jnp.linalg.lstsq(G_k.T, G[k % m])
        xkp1 = xk + gk - (X_k + G_k).T @ gammas
        gkp1 = f(xkp1) - xkp1

        X = X.at[(k + 1) % m].set(xkp1)
        G = G.at[(k + 1) % m].set(gkp1)
        X_k = X_k.at[(k + 1) % m].set(xkp1 - xk)
        G_k = G_k.at[(k + 1) % m].set(gkp1 - gk)

        return X, G, X_k, G_k

    (h,) = x0.shape
    X = jnp.zeros((m, h), x0.dtype)
    X = X.at[0].set(x0)
    X = X.at[1].set(f(x0))

    G = jnp.zeros((m, h), x0.dtype)
    G = G.at[1].set(f(x0) - x0)

    X_k = jnp.zeros((m, h), x0.dtype)
    X_k = X_k.at[1].set(X[1] - X[0])

    G_k = jnp.zeros((m, h), x0.dtype)
    G_k = G_k.at[1].set(G[1] - G[0])

    X, _, _, _ = jax.lax.fori_loop(
        1, n_iterations + 1, body_fn, init_val=(X, G, X_k, G_k)
    )
    return X[(n_iterations + 1) % m]
