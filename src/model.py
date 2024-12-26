import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from .implicit import FixedPointSolver


class ConvNet(eqx.Module):
    project: nn.Conv2d
    deq: nn.Sequential
    classify: nn.Linear

    def __init__(
        self, n_channels: int, kernel_size: int, n_classes: int, *, key: PRNGKeyArray
    ):
        keys = iter(jr.split(key, 4))
        self.project = nn.Conv2d(
            1, n_channels, kernel_size, padding="same", key=next(keys)
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

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, "height width"], solver: FixedPointSolver
    ) -> tuple[Float[Array, " n_classes"], Array]:
        x = einops.rearrange(x, "h w -> 1 h w")
        x = self.project(x)
        x_eq = solver(self.deq, x)
        x = einops.reduce(x_eq, "c h w -> c", "mean")
        return self.classify(x), x_eq
