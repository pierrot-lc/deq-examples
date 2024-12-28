# Deep Equilibrium Models with Examples

This code is an attempt to implement small DEQs on simple tasks such as the
MNIST classification. A good implementation should use advanced solvers such as
the accelerated Anderson method or Broyden's method.

When I face an issue during custom-root backward pass: https://github.com/jax-ml/jax/issues/18311

Notes:
- Compiled code can remove unusued lines of code
- Captured arguments from variables used outside of a locally defined function
  makes the backward pass crash when the function is used within a fori loop.
  It turns out to be true only when using `jax.lax.custom_root`. If the
  implicit diff is implemented with `jax.custom_vjp` we can do whatever we want
  during backward pass!
