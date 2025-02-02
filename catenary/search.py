"""Main namespace for Catenary code.


Notes / Agenda for Stephan:

- Quick sketch from Henry of what we're trying to do.
- Code sketch;
  - goal is a function from N traces -> inner product matrix -> positivity check
  - then we root-find on correlators.
  - goal is to find values for the correlators such that the maximized min
    eigenvalue is 0, or justttt positive.

V(A) = A^2 + g/3 A^4

TODO
- send code + exact equations

We were thinking that we would be able to generate large inner product
matrices, and that our bottleneck would be solved by the lanczos solver.

Questions:
  - numerical instability + generating a bunch of functions in sympy -> jaxpr. Crazy?
  - how should we be thinking about root-finding with jax? any refs?
  - any recs in the JAX arena for code that does this well?

CURRENT GOAL:

- plot the min eigenvalue of upper-left matrices as we train
- dynamically adjust alpha whenever we hit a nan. What is the right interface,
  here?

- search over t1 and t2
- confirm with c2exact
- plot t2 as a function of g (as estimate). Pull in UV to make this happen.

IDEAS:

- can I pass a sparse vector to express the already-supplied terms? My parameters?
- create arange a single time?

what if we get stuck at a local min?

in ML, calculate gradient on random subset, nice. Here, we could take a random
submatrix, and that might help us get unstuck if that in fact happens.

"""

from functools import partial
from itertools import count
from typing import Callable, Tuple

import jax as j
import jax.experimental.optimizers as jo
import jax.numpy as np
import jax.numpy.linalg as la
import uv.reporter.store as r
from jax.config import config
from uv.fs.reporter import FSReporter

import catenary.jax_fns as cj
import catenary.single as cs
import catenary.util as u

# I'm doing this now because if we don't tune alpha properly, we get a blowup
# in the size of the largest elements of the inner product matrix. If we can
# get that under control early we can remove this requirements.
config.update('jax_enable_x64', True)

# Types

# This is the optimizer triple from Jax.
Optimizer = object


def LossGrad(t):
  """Returns a pair of the loss and the differential in the input parameters.

  """
  return Callable[t, Tuple[float, t]]


# Loss Function Wrapper


@partial(j.jit, static_argnums=(0, 1))
def min_eigenvalue(n, alpha, g, t1, t2):
  """Returns the minimum eigenvalue of the inner product matrix built out of the
  supplied parameters.

  """
  m = cs.inner_product_matrix(n, alpha, g, t1, t2)
  return la.eigvalsh(m)[0]


def gradient_ascent(initial,
                    loss_grad_fn,
                    stop_fn,
                    optimizer: Optimizer,
                    reporter=r.NullReporter()):
  """Generalized gradient ascent in JAX."""
  init_fn, update_fn, get_params = optimizer

  @j.jit
  def step(i, optimizer_state):
    """Calculates a single step of gradient ascent."""
    x = get_params(optimizer_state)
    loss, dx = loss_grad_fn(x)
    return loss, update_fn(i, -dx, optimizer_state)

  loss = None
  state = init_fn(initial)

  i = 0
  for i in count(start=1):
    old_loss, old_state = loss, state
    loss, state = step(i, state)
    metadata = {
        "loss.previous": old_loss,
        "state.previous": get_params(old_state),
        "loss": loss,
        "state": get_params(state)
    }
    reporter.report_all(i, metadata)

    if stop_fn(i, old_loss, loss):
      break

  ret = {"steps": i}
  ret.update(metadata)
  return ret


def main(g,
         t1=0.0,
         t2=1.0,
         n=8,
         alpha=0.5,
         steps=1000,
         step_size=1e-2,
         absolute_tolerance=1e-10,
         relative_tolerance=1e-10,
         log_path="notebooks/output/repl",
         reporter_outputs=20):
  """This seems to work up to n=7.

import catenary.single as s
from uv.fs.reader import FSReader
reader = FSReader("notebooks/cake/face1")
data=reader.read_all(["eigenvalue", "t2"])
s.plot_metrics(data, 2, 2)


  Some example calls:

  import catenary.search as s

  s.main(g=1.2, n=5, alpha=1, steps=1000)
  s.main(g=1.2, n=100, alpha=0.5, step_size=1e-2, steps=2000)
  s.main(g=1.2, alpha=0.6630955754113801, t2=0.48642791, n=1000, step_size=1e-5, absolute_tolerance=1e-8, relative_tolerance=1e-8, steps=100)

  """

  # Build the reporter that we'll use to generate results.
  fs = FSReporter(log_path).stepped()
  logging = u.LoggingReporter(digits=10)
  quot, _ = divmod(steps, reporter_outputs)
  # reporter = fs.plus(logging).report_each_n(quot)
  reporter = logging.report_each_n(quot)

  loss_fn = partial(cj.jacfwd(min_eigenvalue, argnums=4), n, alpha, g, t1)

  stop_fn = u.either(
      u.max_step(steps),
      u.either(u.positive_loss,
               u.stop_below(relative_tolerance, absolute_tolerance)))

  return gradient_ascent(initial=t2,
                         loss_grad_fn=loss_fn,
                         stop_fn=stop_fn,
                         optimizer=optimizer,
                         reporter=reporter)


def wrapper(g, t2, step_size=0.0001):
  m = main(g, t2=t2, step_size=step_size, steps=10000)
  return m['loss'], m['state']


def bisection_search(f, a, a_state, b, b_state, tol=1e-3):
  """Bisection search smart enough to pass the previously used state along.

  - TODO convert this to use the same stop function interface.
  - TODO properly thread the state through this thing.
  - TODO maybe instead we just need an abstraction around narrowing the bounds,
         instead of explicit state?

  """
  fa, a_state = f(a, a_state)
  print(f"a: {a}, fa: {fa}, a_state: {a_state}")

  fb, b_state = f(b, b_state)
  print(f"b: {b}, fb: {fb}, a_state: {b_state}")

  if fa * fb > 0:
    print("No root found.")

  else:
    i = 0
    while abs((b - a) / 2.0) > tol:
      mid = (a + b) / 2.0

      # bias left to start.
      mid_state = a_state

      # we need to pass ONE of the states...
      fmid, mid_state = f(mid, mid_state)
      print(f"mid: {mid}, fmid: {fmid}, mid_state: {mid_state}")

      if fa * fmid < 0:
        # fa and fmid still have opposite signs, so move in from b.
        b, b_state = mid, mid_state
      else:
        a, a_state = mid, mid_state

      i += 1
    return (mid, fmid, mid_state, i)


if __name__ == '__main__':
  a = -3.0
  b = 2.0

  # this bisection search starts with a large learning rate to find a nice
  # initial point, then backs off to a small step size.
  bisection_search(wrapper, a,
                   wrapper(a, 1.0, step_size=0.01)[1], b,
                   wrapper(b, 1.0, step_size=0.01)[1])
