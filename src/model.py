import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from .implicit import FixedPointSolver, ImplicitStats, fixed_point


class ConvNet(eqx.Module):
    project: nn.Sequential
    deq: nn.Sequential
    classify: nn.Linear

    def __init__(
        self, n_channels: int, kernel_size: int, n_classes: int, *, key: PRNGKeyArray
    ):
        keys = iter(jr.split(key, 4))
        self.project = nn.Sequential(
            [
                nn.Conv2d(1, n_channels, kernel_size=3, stride=3, key=next(keys)),
                nn.Lambda(jax.nn.gelu),
                nn.GroupNorm(groups=1, channels=n_channels),
            ]
        )
        self.deq = nn.Sequential(
            [
                nn.Conv2d(
                    n_channels, n_channels, kernel_size, padding="same", key=next(keys)
                ),
                nn.Lambda(jax.nn.gelu),
                nn.GroupNorm(groups=1, channels=n_channels),
            ]
        )
        self.classify = nn.Linear(n_channels, n_classes, key=next(keys))

    def solve_deq(
        self, x: Array, solver: FixedPointSolver, stats: ImplicitStats
    ) -> Array:
        params, static = eqx.partition(self.deq, eqx.is_array)

        def f(x: Array, a: PyTree) -> Array:
            params, x_init = a
            deq = eqx.combine(params, static)
            return deq(x) + x_init

        a = (params, x)
        return fixed_point(f, solver, x, a, stats)

    def solver_stats(
        self, x: Float[Array, "height width"], solver: FixedPointSolver
    ) -> ImplicitStats:
        """Extract solver statistics of the given sample.

        Do a forward and backward pass to extract both the forward values and the
        solver statistics. The solver statistics are extracted by asking for the vjp
        w.r.t. the input statistics. This is a hack that allows us to pass values
        computed during both forward and backward pass.
        """
        x = einops.rearrange(x, "h w -> 1 h w")
        x = self.project(x)
        stats = ImplicitStats(forward=jnp.array(0.0), backward=jnp.array(0.0))
        x_star, stats_vjp = jax.vjp(lambda s: self.solve_deq(x, solver, stats), stats)
        (stats,) = stats_vjp(x_star)
        return stats

    def __call__(
        self,
        x: Float[Array, "height width"],
        solver: FixedPointSolver,
    ) -> Float[Array, " n_classes"]:
        x = einops.rearrange(x, "h w -> 1 h w")
        x = self.project(x)
        x_star = self.solve_deq(
            x, solver, ImplicitStats(forward=jnp.array(0.0), backward=jnp.array(0.0))
        )
        x = einops.reduce(x_star, "c h w -> c", "mean")
        return self.classify(x)
