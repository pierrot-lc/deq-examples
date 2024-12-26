from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .solvers import (
    anderson_acceleration,
    fixed_point_iterations,
    lstsq_qr,
    neumann_series,
    root_lbfgs,
)


class FixedPointSolver(eqx.Module):
    solve_method: str = eqx.field(static=True)
    tangent_solve_method: str = eqx.field(static=True)

    def __call__(self, f: Callable, x: Array) -> Array:
        x_shape = x.shape

        def f_flatten(x_flatten: Array) -> Array:
            y = f(x_flatten.reshape(x_shape))
            return y.flatten()

        def f_root(x_flatten: Array) -> Array:
            return f_flatten(x_flatten) - x_flatten

        def solve_fn(_: Callable, x_flatten: Array) -> Array:
            match self.solve_method:
                case "l-bfgs":
                    return root_lbfgs(f_root, x_flatten, n_iterations=1000)
                case "anderson":
                    return anderson_acceleration(
                        f_flatten, x_flatten, n_iterations=100, m=10
                    )
                case "fixed-point":
                    return fixed_point_iterations(
                        f_flatten, x_flatten, n_iterations=1000
                    )
                case _:
                    raise ValueError("Unknown solve method")

        def tangent_solve_fn(g: Callable, y: Array) -> Array:
            match self.tangent_solve_method:
                case "exact":
                    A = jax.jacobian(g)(y)
                    x = jnp.linalg.solve(A, y)
                    return x
                case "qr":
                    A = jax.jacobian(g)(y)
                    Q, R = jnp.linalg.qr(A)
                    x = lstsq_qr(Q, R, y)
                    return x
                case "neumann":

                    def f(x):
                        return x - g(x)

                    return neumann_series(f, y, n_iterations=3)
                case "anderson":

                    def f(x):
                        """Ax = y <=> Ax + x - y = x."""
                        return g(x) + x - y

                    return anderson_acceleration(g, y, n_iterations=10, m=3)
                case _:
                    raise ValueError("Unknown tangent solve method")

        x_star = jax.lax.custom_root(
            f_root,
            initial_guess=x.flatten(),
            solve=solve_fn,
            tangent_solve=tangent_solve_fn,
        )
        return x_star.reshape(x_shape)
