from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree, Scalar


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


def root_lbfgs(f: Callable, x: PyTree, n_iterations: int) -> PyTree:
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
        x_root = f(x)
        return optax.tree_utils.tree_l2_norm(x_root)

    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn)

    def body_fn(_: Any, carry: tuple[PyTree, optax.OptState]) -> tuple:
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
