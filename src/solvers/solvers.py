from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array

from .anderson import anderson_acceleration
from .gradient import root_lbfgs
from .others import fixed_point_iterations


class FixedPointSolver(eqx.Module):
    # General.
    name: str = eqx.field(static=True)
    n_iterations: int = eqx.field(static=True)

    # Anderson specifics.
    anderson_m: int = eqx.field(static=True)
    anderson_b: float = eqx.field(static=True)

    def __call__(self, f: Callable, x: Array) -> Array:
        match self.name:
            case "anderson":
                return anderson_acceleration(
                    f, x, self.n_iterations, self.anderson_m, self.anderson_b
                )
            case "fixed-point":
                return fixed_point_iterations(f, x, self.n_iterations)
            case "lbfgs":
                return root_lbfgs(lambda x: f(x) - x, x, self.n_iterations)
            case _:
                raise ValueError(f"Unknown solver: {self.name}")
