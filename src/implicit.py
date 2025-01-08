from collections.abc import Callable
from functools import partial
from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from .solvers import (
    anderson_acceleration,
    fixed_point_iterations,
    neumann_series,
    root_lbfgs,
)


class ImplicitStats(TypedDict):
    forward: Array
    backward: Array


class FixedPointSolver(eqx.Module):
    """Implements the full implicit fixed point logic.

    This class is purely static, so that it counts as a nondiff argument. This is useful
    to pass the hyperparameter values around during both forward and backward passes.
    """

    fwd_solver: str = eqx.field(static=True)
    fwd_iterations: int = eqx.field(static=True)
    fwd_init: str = eqx.field(static=True)

    bwd_solver: str = eqx.field(static=True)
    bwd_iterations: int = eqx.field(static=True)

    # Anderson specifics.
    anderson_m: int = eqx.field(static=True)
    anderson_b: float = eqx.field(static=True)

    def __call__(
        self,
        f: Callable,
        x: Array,
        a: PyTree,
        key: PRNGKeyArray,
        solver_stats: bool = False,
    ) -> tuple[Array, Scalar, ImplicitStats | None]:
        key1, key2 = jr.split(key)
        stats = ImplicitStats(forward=jnp.array(0.0), backward=jnp.array(0.0))

        match self.fwd_init:
            case "zero":
                x = jnp.zeros_like(x)
            case "random":
                x = jr.normal(key1, x.shape)
            case _:
                raise ValueError("Unknown init")

        x_star = FixedPointSolver._fixed_point(self, f, x, a, stats)
        jac_reg = FixedPointSolver.jac_reg(lambda x: f(x, a), x_star, key2)
        stats = self._solver_stats(f, x, a) if solver_stats else None
        return x_star, jac_reg, stats

    @partial(jax.custom_vjp, nondiff_argnums=(0, 1))
    def _fixed_point(
        self, f: Callable, x: Array, a: PyTree, _stats: ImplicitStats
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
            stats: Empty stats to be filled by the solver so that it is possible to
                fetch the backward stats by asking for the vjp/grad of this function
                w.r.t. the stats.

        ---
        Returns:
            The estimated fixed point.

        ---
        Sources:
            https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations
        """
        match self.fwd_solver:
            case "anderson":
                return anderson_acceleration(
                    lambda x: f(x, a),
                    x,
                    self.fwd_iterations,
                    self.anderson_m,
                    self.anderson_b,
                )
            case "lbfgs":
                return root_lbfgs(lambda x: f(x, a) - x, x, self.fwd_iterations)
            case "picard":
                return fixed_point_iterations(lambda x: f(x, a), x, self.fwd_iterations)
            case _:
                raise ValueError("Unknown forward solver")

    def _fixed_point_fwd(
        self, f: Callable, x: Array, a: PyTree, stats: ImplicitStats
    ) -> tuple[Array, tuple[Array, PyTree, ImplicitStats]]:
        x_star = FixedPointSolver._fixed_point(self, f, x, a, stats)
        # Compute the relative error as is done in torchdeq.
        # https://torchdeq.readthedocs.io/en/latest/torchdeq/solver.html#solver-stat
        eps = x_star - f(x_star, a)
        eps = jnp.linalg.norm(eps.flatten()) / jnp.linalg.norm(x_star.flatten())
        stats = ImplicitStats(forward=eps, backward=stats["backward"])
        return x_star, (x_star, a, stats)

    def _fixed_point_bwd(
        self, f: Callable, ctx: tuple[Array, PyTree, ImplicitStats], v: Array
    ) -> tuple[Array, PyTree, ImplicitStats]:
        x_star, a, stats = ctx
        _, A = jax.vjp(lambda x: f(x, a), x_star)
        _, B = jax.vjp(lambda a: f(x_star, a), a)

        # Compute the vector w^T = v^T @ (I - A)^-1.
        # Equivalent to the fixed point w^T = v^T + w^T @ A.
        # NOTE: The initial guess of fixed point solvers is 0 as is done in torchdeq.
        # https://github.com/locuslab/torchdeq/blob/main/torchdeq/grad.py#L143
        match self.bwd_solver:
            case "anderson":
                w = anderson_acceleration(
                    lambda w: v + A(w)[0],
                    jnp.zeros_like(v),
                    self.bwd_iterations,
                    self.anderson_m,
                    self.anderson_b,
                )
            case "lbfgs":
                return root_lbfgs(
                    lambda w: v + A(w)[0] - w, jnp.zeros_like(v), self.bwd_iterations
                )
            case "neumann":
                w = neumann_series(
                    lambda v: A(v)[0], v, n_iterations=self.bwd_iterations
                )
            case "picard":
                w = fixed_point_iterations(
                    lambda w: v + A(w)[0], jnp.zeros_like(v), self.bwd_iterations
                )
            case _:
                raise ValueError("Unknown backward solver")

        # Compute the solver's performance.
        eps = w - (v + A(w)[0])
        eps = jnp.linalg.norm(eps.flatten()) / jnp.linalg.norm(w.flatten())
        stats = ImplicitStats(forward=stats["forward"], backward=eps)

        # Finish the implicit derivation.
        (a_bar,) = B(w)

        return jnp.zeros_like(x_star), a_bar, stats

    def _solver_stats(self, f: Callable, x: Array, a: PyTree) -> ImplicitStats:
        """Extract solver statistics of the given sample.

        Do the full implicit forward and backward

        The solver statistics are extracted by asking for the vjp w.r.t. the input
        statistics. This is a hack that allows us to pass values computed during both
        forward and backward pass.
        """
        stats = ImplicitStats(forward=jnp.array(0.0), backward=jnp.array(0.0))
        (x_star, stats_vjp) = jax.vjp(
            lambda s: FixedPointSolver._fixed_point(self, f, x, a, s), stats
        )
        (stats,) = stats_vjp(x_star)
        return stats

    @staticmethod
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


FixedPointSolver._fixed_point.defvjp(
    FixedPointSolver._fixed_point_fwd, FixedPointSolver._fixed_point_bwd
)
