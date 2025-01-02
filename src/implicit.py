from collections.abc import Callable
from functools import partial
from typing import Protocol, TypedDict

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

SolverStats = dict[str, Array]


class ImplicitStats(TypedDict):
    forward: SolverStats
    backward: SolverStats


class FixedPointSolver(Protocol):
    """Abstract implementation of what a fixed point solver should look like."""

    def init_stats(self) -> SolverStats:
        """Initialize empty stats, to be filled by the solver."""
        ...

    def __call__(self, f: Callable, x: Array) -> tuple[Array, SolverStats]:
        """Find the fixed point of `f`.

        ---
        Args:
            f: Function for which the solver should find the fixed point.
            x: Initial guess.

        ---
        Returns:
            A tuple containing:
                x_star: Estimated fixed point by the solver.
                stats: A dictionnary of statistics from the solver.
        """
        ...


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def fixed_point(
    f: Callable, solver: FixedPointSolver, x: Array, a: PyTree, stats: ImplicitStats
) -> Array:
    """Compute the fixed point of `f`.
    Use implicit differientation for the reverse vjp.

    ---
    Args:
        f: The function for which we want to find the fixed point. Can close over static
            paramters.
        solver: The solver used for the finding the fixed point.
        x: Initial guess for the fixed point.
        a: PyTree of differentiable parameters. All dynamic parameters must be passed
            here.
        stats: Empty stats to be filled by the solver so that it is possible.
            Fetch the backward stats by asking for the vjp/grad of this function w.r.t.
            the stats.

    ---
    Returns:
        The estimated fixed point.

    ---
    Sources:
        https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations
    """
    x_star, _ = solver(lambda x: f(x, a), x)
    return x_star


def fixed_point_fwd(
    f: Callable, solver: FixedPointSolver, x: Array, a: PyTree, stats: ImplicitStats
) -> tuple[Array, tuple[Array, PyTree, ImplicitStats]]:
    x_star, forward_stats = solver(lambda x: f(x, a), x)
    stats = ImplicitStats(forward=forward_stats, backward=stats["backward"])
    return x_star, (x_star, a, stats)


def fixed_point_bwd(
    f: Callable,
    solver: FixedPointSolver,
    res: tuple[Array, PyTree, ImplicitStats],
    v: Array,
) -> tuple[Array, PyTree, ImplicitStats]:
    """Use the maths notations from the jax tutorial on implicit diff."""
    x_star, a, stats = res
    _, A = jax.vjp(lambda x: f(x, a), x_star)
    _, B = jax.vjp(lambda a: f(x_star, a), a)

    w, backward_stats = solver(lambda w: v + A(w)[0], v)
    stats = ImplicitStats(forward=stats["forward"], backward=backward_stats)
    (a_bar,) = B(w)
    return jnp.zeros_like(x_star), a_bar, stats


fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd)
