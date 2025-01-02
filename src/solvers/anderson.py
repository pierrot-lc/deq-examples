from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..implicit import SolverStats


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
    f: Callable, x0: Array, n_iterations: int, m: int, beta: float
) -> tuple[Array, SolverStats]:
    """Use Anderson acceleration to find the fixed point of f.

    ---
    Args:
        f: The function for which we want to find the fixed point.
        x0: The initial guess.
        n_iterations: Number of Anderson steps.
        m: Size of the history.
        beta: Damping factor.

    ---
    Returns:
        The last estimation of the fixed point.

    ---
    Sources:
        Paper: https://epubs.siam.org/doi/10.1137/10078356X
        Implem details from the same author: https://users.wpi.edu/~walker/Papers/anderson_accn_algs_imps.pdf
        Easier pseudo-code: https://ctk.math.ncsu.edu/TALKS/Anderson.pdf
    """
    # Flattened version of f.
    x_shape = x0.shape
    f_flatten = lambda x: f(x.reshape(x_shape)).flatten()
    x0 = x0.flatten()

    def body_fn(k: int, carry: tuple[Array, Array, Array, SolverStats]) -> tuple:
        X, G, F, stats = carry
        gk, fk = G[k % m], F[k % m]

        dG = jnp.roll(G, shift=-1, axis=0) - G
        dF = jnp.roll(F, shift=-1, axis=0) - F

        # TODO: Better Q/R update.
        Q, R = jnp.linalg.qr(dF.T)
        gammas = lstsq_qr(Q, R, fk, lambda_=1e-3)
        # gammas, *_ = jnp.linalg.lstsq(dF.T, fk)
        xkp1 = gk - dG.T @ gammas
        xkp1 = xkp1 - (1 - beta) * (fk - dF.T @ gammas)
        gkp1 = f_flatten(xkp1)

        X = X.at[(k + 1) % m].set(xkp1)
        G = G.at[(k + 1) % m].set(gkp1)
        F = F.at[(k + 1) % m].set(gkp1 - xkp1)
        stats["roots"] = stats["roots"].at[k].set(jnp.linalg.norm(gkp1 - xkp1))

        return X, G, F, stats

    (h,) = x0.shape
    X = jnp.zeros((m, h), x0.dtype)
    X = X.at[0].set(x0)
    G = jax.vmap(f_flatten)(X)
    F = G - X
    stats: SolverStats = {"roots": jnp.zeros((n_iterations,), float)}
    X, *_, stats = jax.lax.fori_loop(
        0, n_iterations, body_fn, init_val=(X, G, F, stats)
    )
    return X[n_iterations % m].reshape(x_shape), stats


class AndersonSolver(eqx.Module):
    n_iterations: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def init_stats(self) -> SolverStats:
        return {"roots": jnp.zeros((self.n_iterations,), float)}

    def __call__(self, f: Callable, x: Array) -> tuple[Array, SolverStats]:
        return anderson_acceleration(f, x, self.n_iterations, self.m, self.beta)

    def __post_init__(self):
        assert self.n_iterations > 0
        assert self.m > 2
        assert 0.0 < self.beta <= 1.0
