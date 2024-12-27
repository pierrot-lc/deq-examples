from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from beartype import beartype
from jaxtyping import Array, Float, PyTree, Scalar, jaxtyped


def fixed_point_iterations(
    f: Callable, x: Array, n_iterations: int, *args: list[Any]
) -> Array:
    def body_fn(_: Any, x: Array) -> Array:
        return f(x, *args)

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
    x_ = jsp.linalg.solve_triangular(R_, b_)
    return x_[:N]


def anderson_acceleration(
    f: Callable,
    x0: Float[Array, " hidden_dim"],
    n_iterations: int,
    m: int,
    *args: list[Any],
) -> Float[Array, " hidden_dim"]:
    """Use Anderson acceleration to find the fixed point of f.

    ---
    Args:
        f: The function for which we want to find the fixed point.
        x0: The initial guess.
        n_iterations: Number of Anderson steps.
        m: Size of the history.
        *args: Extra args for `f`. Typically coming from `jax.closure_convert`.

    ---
    Returns:
        The last estimation of the fixed point.

    ---
    Sources:
        Paper: https://epubs.siam.org/doi/10.1137/10078356X
        Pseudo-code: https://en.wikipedia.org/wiki/Anderson_acceleration#Example_MATLAB_implementation
    """
    assert m > 2

    def body_fn(k: int, carry: tuple[Array, Array, Array, Array]) -> tuple:
        X, G, Xk, Gk = carry
        xk, gk = X[k % m], G[k % m]

        Q, R = jnp.linalg.qr(Gk)
        gammas = lstsq_qr(Q, R, gk, lambda_=0.01)
        # gammas, *_ = jnp.linalg.lstsq(Gk, gk)
        xkp1 = xk + gk - (Xk + Gk) @ gammas
        gkp1 = f(xkp1, *args) - xkp1

        X = X.at[(k + 1) % m].set(xkp1)
        G = G.at[(k + 1) % m].set(gkp1)
        Xk = Xk.at[:, (k + 1) % m].set(xkp1 - xk)
        Gk = Gk.at[:, (k + 1) % m].set(gkp1 - gk)

        return X, G, Xk, Gk

    (h,) = x0.shape
    X = jnp.zeros((m, h), x0.dtype)
    X = X.at[0].set(x0)
    X = X.at[1].set(f(x0, *args))

    G = jnp.zeros((m, h), x0.dtype)
    G = G.at[1].set(f(x0, *args) - x0)

    Xk = jnp.zeros((h, m), x0.dtype)
    Xk = Xk.at[:, 1].set(X[1] - X[0])

    Gk = jnp.zeros((h, m), x0.dtype)
    Gk = Gk.at[:, 1].set(G[1] - G[0])

    X, *_ = jax.lax.fori_loop(1, n_iterations + 1, body_fn, init_val=(X, G, Xk, Gk))
    return X[(n_iterations + 1) % m]


def root_lbfgs(f: Callable, x: PyTree, n_iterations: int, *args: list[Any]) -> PyTree:
    """Root solver using L-BFGS.

    ---
    Args:
        f: The function for which we want to find the root.
        x: Initial guess.
        n_iterations: Number of L-BFGS steps.
        *args: Extra args for `f`. Typically coming from `jax.closure_convert`.

    ---
    Returns:
        The estimated root.

    ---
    Sources:
        Optax tutorial: https://optax.readthedocs.io/en/stable/_collections/examples/lbfgs.html#l-bfgs-solver
    """
    optimizer = optax.lbfgs()

    def loss_fn(x: PyTree) -> Scalar:
        x_root = f(x, *args)
        return optax.tree_utils.tree_l2_norm(x_root)

    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn)

    def body_fn(_: Any, carry: tuple[Array, optax.OptState]) -> tuple:
        x, opt_state = carry

        value, grad = value_and_grad_fn(x, state=opt_state)
        updates, opt_state = optimizer.update(
            grad, opt_state, x, value=value, grad=grad, value_fn=loss_fn
        )
        x = optax.apply_updates(x, updates)

        return x, opt_state

    x_star, _ = jax.lax.fori_loop(
        0, n_iterations, body_fn, init_val=(x, optimizer.init(x))
    )
    return x_star


class Solver(eqx.Module):
    n_iterations: int = eqx.field(static=True)
    anderson_m: int = eqx.field(static=True)

    def __call__(self, f: Callable, x: Array, *args: Any) -> Array:
        return anderson_acceleration(f, x, self.n_iterations, self.anderson_m, *args)
