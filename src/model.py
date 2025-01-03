import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar

from .implicit import FixedPointSolver, ImplicitStats


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
        solver_stats: bool = False,
    ) -> tuple[Float[Array, " n_classes"], Scalar, ImplicitStats | None]:
        """Predict the class of the input image.

        ---
        Args:
            x: Input image.
            solver: Fixed point solver for DEQ forward/backward pass.
            key: Random key used for jacobian regularization.
            solver_stats: Whether to return solver's statistics for this sample.
                Returning solver's statistics require vjp computations.

        ---
        Returns:
            The classes probabilities, the jacobian regularization estimate and
            optionally the solver's statistics.
        """
        x = einops.rearrange(x, "h w -> 1 h w")
        x = self.project(x)

        # NOTE: The function `f` needs to capture all static parameters in its closure.
        # The rest is passed as dynamic parameters in `a`.
        params, static = eqx.partition(self.deq, eqx.is_array)

        def f(x: Array, a: PyTree) -> Array:
            params, x_inject = a
            deq = eqx.combine(params, static)
            return deq(x) + x_inject

        x, reg, stats = solver(f, x, (params, x), key, solver_stats)

        x = einops.reduce(x, "c h w -> c", "mean")
        y = self.classify(x)
        return y, reg, stats
