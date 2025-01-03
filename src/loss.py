from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Scalar


def jac_reg(f: Callable, x: Array, key: PRNGKeyArray) -> Scalar:
    """Estimate the trace of the jacobian of `f` evaluated at `x`.
    Use the Hutchinson estimator.

    ---
    Args:
        f: Function f.
        x: Point at which the jacobian is evaluated.
        key: Random key used for the estimate.

    ---
    Returns:
        The trace estimate: tr(Jf(x) @ Jf(x).T).

    ---
    Sources:
        https://proceedings.mlr.press/v139/bai21b.html
    """
    eps = jr.normal(key, x.shape)
    (_, estimate) = jax.jvp(f, primals=(x,), tangents=(eps,))
    n_elements = jnp.prod(jnp.array(estimate.shape))
    return jnp.linalg.norm(estimate.flatten()) ** 2 / n_elements
