import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar

from .implicit import FixedPointSolver, ImplicitStats, fixed_point
from .loss import jac_reg


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
                nn.Conv2d(
                    1,
                    n_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    key=next(keys),
                ),
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

    def __call__(
        self,
        x: Float[Array, "height width"],
        solver: FixedPointSolver,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, " n_classes"], Scalar]:
        """Predict the class of the input image.

        ---
        Args:
            x: Input image.
            solver: Fixed point solver for DEQ forward/backward pass.
            key: Random key used for jacobian regularization.

        ---
        Returns:
            The classes probabilities and the jacobian regularization estimate.
        """
        x = einops.rearrange(x, "h w -> 1 h w")
        x = self.project(x)
        stats = ImplicitStats(forward=jnp.array(0.0), backward=jnp.array(0.0))
        x, reg = self.solve_deq(x, solver, stats, key)
        x = einops.reduce(x, "c h w -> c", "mean")
        return self.classify(x), reg

    def solve_deq(
        self,
        x: Array,
        solver: FixedPointSolver,
        stats: ImplicitStats,
        key: PRNGKeyArray,
    ) -> tuple[Array, Scalar]:
        """Find the DEQ fixed point and compute the related jacobian regularization."""
        params, static = eqx.partition(self.deq, eqx.is_array)

        def f(x: Array, a: PyTree) -> Array:
            params, x_init = a
            deq = eqx.combine(params, static)
            return deq(x) + x_init

        key1, key2 = jr.split(key)
        a = (params, x)
        x0 = jr.normal(key1, x.shape)
        x_star = fixed_point(f, solver, x0, a, stats)
        reg = jac_reg(lambda x: f(x, a), x_star, key2)
        return x_star, reg

    def solver_stats(
        self,
        x: Float[Array, "height width"],
        solver: FixedPointSolver,
        key: PRNGKeyArray,
    ) -> ImplicitStats:
        """Extract solver statistics of the given sample.

        The solver statistics are extracted by asking for the vjp w.r.t. the input
        statistics. This is a hack that allows us to pass values computed during both
        forward and backward pass.
        """
        x = einops.rearrange(x, "h w -> 1 h w")
        x = self.project(x)
        stats = ImplicitStats(forward=jnp.array(0.0), backward=jnp.array(0.0))
        (x_star, reg), stats_vjp = jax.vjp(
            lambda s: self.solve_deq(x, solver, s, key), stats
        )
        (stats,) = stats_vjp((x_star, reg))
        return stats
