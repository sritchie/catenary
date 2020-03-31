"""Main namespace for Catenary code.

Notes:

- Jax only takes the gradient with respect to the first argument. You don't
  have to curry; you can leave later arguments open for configuration.

CURRENT GOAL:

- modify reference implementation to include alpha
- fix bisection method code (minimize evaluations)
- get a notebook up with the solid single matrix code.
- read henry's code for multi-matrix, get the ideas imported.
- search over t1 and t2
- confirm with c2exact
- plot t2 as a function of g (as estimate). Pull in UV to make this happen.

IDEAS:

- can I pass a sparse vector to express the already-supplied terms? My parameters?
- create arange a single time?
- think about dynamically adjusting alpha.
- regularize.... figure out how to dynamically adjust alpha.

what if we get stuck at a local min?

in ML, calculate gradient on random subset, nice. Here, we could take a random
submatrix, and that might help us get unstuck if that in fact happens.

"""

from functools import partial

import jax as j
import jax.experimental.optimizers as jo
import jax.lax as jl
import jax.numpy as np
import jax.numpy.linalg as la
import scipy.optimize as o
from jax.config import config

import catenary.jax_fns as cj

config.update('jax_enable_x64', True)


def t_k_plus_3_reference(k, g_recip, xs):
  """Clearer version of what's happening to calculate t_{k+3}."""

  def term(i, acc):
    return acc + (xs[i] * xs[k - 1 - i])

  sum_term = jl.fori_loop(0, k, term, 0.0)
  return g_recip * (sum_term - xs[k + 1])


def t_k_plus_3(k, alpha, g_recip, xs):
  """Code-golfed version."""
  n = xs.shape[0]

  # effectively mask all but the first k elements to be 0, then reverse the
  # array.
  ys = np.where(np.arange(n) >= n - k, xs[::-1], 0)

  # get our alpha^4 term.
  ys = np.multiply(ys, np.power(alpha, 4))

  ## "subtract" the (k + 2)'th element in the list - which has index t_{k+1},
  ## if we're zero-indexing - by setting the SECOND element in the not-rolled
  ## list to -1.
  ys = j.ops.index_update(ys, j.ops.index[1], -alpha)

  # Roll the array, bringing all of the reversed elements to the head, the
  # (k+2)nd element into position, so that the subtraction happens during the
  # dot product.
  return g_recip * np.dot(xs, np.roll(ys, k - n))


# example(4, 2, 1.2, 0, 2.5)
# example(4, 2, 1.2, 0, 2.5)
# Out[177]:
# DeviceArray([[ 1.        ,  0.        ,  0.625     ,  0.        ],
#              [ 0.        ,  0.625     ,  0.        , -0.078125  ],
#              [ 0.625     ,  0.        , -0.078125  ,  0.        ],
#              [ 0.        , -0.078125  ,  0.        ,  0.08138021]],            dtype=float64)


def correlators(g, alpha, xs):
  """Populates the supplied vector with correlator values, assuming the first
  three entries have been populated.

  """
  n = xs.shape[0]

  if n <= 2:
    return xs

  g_recip = jl.reciprocal(g)

  def run(k, xs):
    new_v = t_k_plus_3(k - 3, alpha, g_recip, xs)
    return j.ops.index_update(xs, k, new_v)

  return jl.fori_loop(3, n + 1, run, xs)


def initial_state(t1, t2, max_idx):
  """Generates an initial state for the list of correlators; this is all zeroes
  except for the populated values of t1 and t2, and the initial value of t_0 =
  1.

  """
  s = np.zeros(max_idx, dtype=np.double)
  return j.ops.index_update(s, j.ops.index[:3], [1, t1, t2])


@partial(j.jit, static_argnums=(0, 1))
def single_matrix_correlators(n, alpha, g, t1, t2):
  """This is a the function the we want to abstract, so we can explore more
  interesting models.

  This function is jitted for all but the final arg.

  """
  s1 = np.multiply(t1, alpha)
  s2 = np.multiply(t2, np.power(alpha, 2.0))
  xs = initial_state(s1, s2, n)
  return correlators(g, alpha, xs)


@partial(j.jit, static_argnums=(0, 1))
def sliding_window_m(xs, window):
  """Returns matrix with `window` columns whose rows are a sliding window view
  onto the input array xs.

n  The returned matrix will have dimensions (n - window + 1, window).

  """
  rows = xs.shape[0] - window + 1
  wide = j.vmap(partial(np.roll, xs, axis=0))(-np.arange(rows))
  return wide[:, :window]


@partial(j.jit, static_argnums=(0, 1))
def inner_product_matrix(n, alpha, g, t1, t2):
  """Returns the !"""
  items = 2 * n - 1
  xs = single_matrix_correlators(items, alpha, g, t1, t2)
  return sliding_window_m(xs, n)


@partial(j.jit, static_argnums=(0, 1))
def min_eigenvalue(n, alpha, g, t1, t2):
  m = inner_product_matrix(n, alpha, g, t1, t2)
  return la.eigvalsh(m)[0]


# Optimization Code
#
# First attempt is to try this this myself, naively.


def stop_below(tolerance):
  """Returns a function that stops at a certain tolerance."""

  def f(old, new):
    if old is None:
      return False

    return np.abs((new - old) / old) < tolerance

  return f


@partial(j.jit, static_argnums=(0, 1, 2, 3))
def update(step_size, n, g, alpha, t2):
  ret, dt2 = cj.jacfwd(min_eigenvalue, argnums=4)(n, alpha, g, 0.0, t2)
  return ret, t2 + step_size * dt2


def f_naive(g, alpha, t2=2.5, n=5, steps=10000, step_size=1e-4, tolerance=1e-5):
  """This is the function whose roots we are trying to find. This does the
  optimization loop internally.

  """
  ev = None
  stop_fn = stop_below(tolerance)

  for _ in range(steps):
    new_ev, new_t2 = update(step_size, n, g, alpha, t2)

    if new_ev >= 0 or stop_fn(ev, new_ev):
      break

    t2 = new_t2
    ev = new_ev

  return new_t2, new_ev


# Then, another attempt using JAX primitives:


def f(g, alpha=1, t2=2.5, n=5, steps=1000, step_size=1e-3, tolerance=1e-4):
  print("Attempting with g={}, t2_initial={}, n={}".format(g, t2, n))

  init_fn, update_fn, get_params = jo.sgd(step_size)
  stop_fn = stop_below(tolerance)

  @j.jit
  def step(i, opt_state):
    """Calculates a single step of gradient ascent against the min eigenvalue."""
    x = get_params(opt_state)
    new_ev, dx = cj.jacfwd(min_eigenvalue, argnums=4)(n, alpha, g, 0.0, x)
    return new_ev, update_fn(i, -dx, opt_state)

  ev = None
  opt_state = init_fn(t2)

  for i in range(steps):
    old_ev = ev
    ev, opt_state = step(i, opt_state)
    if ev >= 0 or stop_fn(old_ev, ev):
      break

  t2_final = get_params(opt_state)

  print("steps={}, t2_final ={}, min_eigenvalue={}".format(i, t2_final, ev))

  return t2_final, ev


def optf(g, **kwargs):
  """Version that I can pass to my own bisect function."""
  return f(g, **kwargs)[1]


def main(**kwargs):
  """This seems to work up to n=7."""
  return o.brentq(partial(optf, **kwargs), -.2, -.001)


if __name__ == '__main__':
  main(n=5, alpha=1)