# Deep Equilibrium Models with Examples

This code is an attempt to implement small DEQs on simple tasks such as the
MNIST classification. A good implementation should use advanced solvers such as
the accelerated Anderson method or Broyden's method.

*The goal of this repository is not to provide a DEQ library but more to show
how to implement one yourself.*

The implicit differentiation is almost purely implemented with raw `jax`. I use
`equinox` to easily handle hyperparameters but it should be easy to translate
this code into any other jax-based DL framework.

Solvers are implemented using `jax` only. You should be able to take them as
they are.

## The good stuff

The main code is in `src/implicit.py` and `src/solvers/`. The implicit file
implements everything related to the implicit differentiation. You have to
instanciate the implicit solver `FixedPointSolver` by specifying every
hyperparameters, and then simply call the solver to find the fixed-point.

```py
from src.implicit import FixedPointSolver

solver = FixedPointSolver(
    fwd_solver="anderson",
    fwd_iterations=50,
    fwd_init="zero",
    bwd_solver="neumann",
    bwd_iterations=3,
    anderson_m=3,          # See `src/solvers/anderson.py`
    anderson_b=0.5,
)
f: Callable  # The function for which we search the fixed-point. Must take two positional arguments: x and a.
x: Array     # Any array having the same shape as the fixed-point.
a: PyTree    # Anything else that is a dynamic parameter and acts in `f`.
key: PRNGKeyArray  # Some random key to handle potential random initializations and the jacobian regularization estimate.

x_star, jac_reg, _ = solver(f, x, a, key)

# You can also ask for the solver statistics.
x_star, jac_reg, stats = solver(f, x, a, key, solver_stats=True)
```

The solver returns both the fixed-point and the associated jacobian
regularization estimation. It can also return the solver's statistics if you
specifically ask for it. Please know that the statistics require a full forward
and backward pass of the solver so it should only be used for tracking the
stability of your training.

You should also have a look at `src/model.py` to see how I declare the
arguments `f`, `x` and `a` for the solver.

## Implementation details

There are currently two ways of implementing DEQs:

- Using [`jax.lax.custom_vjp`](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations)
- Using [`jax.lax.custom_root`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.custom_root.html)

The second approach looks pretty because it directly target implicit
differentiation. But it is in fact a pain to use because it tends to easily
throw errors when trying to access variables captured by closures. This makes
the implementation really difficult, [with some cryptic error
messages](https://github.com/jax-ml/jax/issues/18311).

The first approach solves all those issues. Even if it takes a little bit more
code to properly setup the implicit differentiation, everything is clearer and
more control is given to the developer into how the implicit differentiation is
done.

### Solver's statistics

It is actually not easy to extract the performance of the solver for both the
forward and backward passes. The way I've implemented it is to add a special
PyTree dynamic argument to the implicit solver which is where I add the
solver's statistics for both forward and backward passes. I can then ask for
the derivative of the solver w.r.t. this argument to read the solver's
statistics.

Please have a look at the specific code of `FixedPointSolver._solver_stats` if
you want to see exactly how I implemented it.
