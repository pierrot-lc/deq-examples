from collections.abc import Callable
from typing import Any

import jax
from jaxtyping import Array


def fixed_point_iterations(f: Callable, x: Array, n_iterations: int) -> Array:
    def body_fn(_: Any, x: Array) -> Array:
        return f(x)

    return jax.lax.fori_loop(0, n_iterations, body_fn, init_val=x)
