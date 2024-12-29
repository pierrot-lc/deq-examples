import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, jaxtyped

from .implicit import fixed_point
from .solvers import FixedPointSolver


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

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, "height width"], solver: FixedPointSolver
    ) -> tuple[Float[Array, " n_classes"], Array, Array]:
        x = einops.rearrange(x, "h w -> 1 h w")
        x_proj = self.project(x)

        params, static = eqx.partition(self.deq, eqx.is_array)

        def f(x: Array, a: PyTree) -> Array:
            params, x_init = a
            deq = eqx.combine(params, static)
            return deq(x) + x_init

        a = (params, x_proj)
        x_star = fixed_point(f, solver, x_proj, a)
        x_root = f(x_star, a) - x_star
        x = einops.reduce(x_star, "c h w -> c", "mean")
        return self.classify(x), x_star, x_root
