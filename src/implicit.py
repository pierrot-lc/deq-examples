from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .solvers import Solver


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def fixed_point(f: Callable, solver: Solver, x: Array, a: PyTree) -> Array:
    """Compute the fixed point of `f`.
    Use implicit differientation for the reverse vjp.

    ---
    Args:
        f: The function for which we want to find the fixed point.
        solver: The solver used for the finding the fixed point.
        x: Initial guess for the fixed point.
        a: PyTree of differentiable parameters.

    ---
    Returns:
        The estimated fixed point.

    ---
    Sources:
        https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations
    """
    return solver(lambda x: f(x, a), x)


def fixed_point_fwd(
    f: Callable, solver: Solver, x: Array, a: PyTree
) -> tuple[Array, tuple[Array, PyTree]]:
    x_star = solver(lambda x: f(x, a), x)
    return x_star, (x_star, a)


def fixed_point_bwd(
    f: Callable, solver: Solver, res: tuple[Array, PyTree], v: Array
) -> tuple[Array, PyTree]:
    """Use the maths notations from the jax tutorial on implicit diff."""
    x_star, a = res
    _, A = jax.vjp(lambda x: f(x, a), x_star)
    _, B = jax.vjp(lambda a: f(x_star, a), a)

    w = solver(lambda w: v + A(w)[0], v)
    a_bar = B(w)[0]
    return jnp.zeros_like(x_star), a_bar


fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd)
