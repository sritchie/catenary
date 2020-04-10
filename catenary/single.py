"""Main namespace for Catenary code.

Notes:

- Jax only takes the gradient with respect to the first argument. You don't
  have to curry; you can leave later arguments open for configuration.

CURRENT GOAL:


- plot loss and accuracy as a function of steps for various learning rates.
- plot the min eigenvalue of upper-left matrices as we train
- fix the alpha-readjustment

- get a notebook up with the solid single matrix code.
- read henry's code for multi-matrix, get the ideas imported.
- search over t1 and t2
- confirm with c2exact
- plot t2 as a function of g (as estimate). Pull in UV to make this happen.

QUESTIONS

- can you show me the multi-matrix stuff?
- job candidates, send them my way?
- what's the different loss fn?

IDEAS:

- can I pass a sparse vector to express the already-supplied terms? My parameters?
- create arange a single time?
- figure out how to dynamically adjust alpha.

what if we get stuck at a local min?

in ML, calculate gradient on random subset, nice. Here, we could take a random
submatrix, and that might help us get unstuck if that in fact happens.

"""

from functools import partial

import jax as j
import jax.lax as jl
import jax.numpy as np
import scipy.optimize as o
from jax.config import config

config.update('jax_enable_x64', True)

# Correlators


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
  ys = j.ops.index_update(ys, j.ops.index[1], -np.square(alpha))

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

  The returned matrix will have dimensions (n - window + 1, window).

  """
  rows = xs.shape[0] - window + 1
  wide = j.vmap(partial(np.roll, xs, axis=0))(-np.arange(rows))
  return wide[:, :window]


@partial(j.jit, static_argnums=(0, 1))
def inner_product_matrix(n, alpha, g, t1, t2):
  """Returns the... inner product matrix of correlators for the single matrix
  model.

  """
  items = 2 * n - 1
  xs = single_matrix_correlators(items, alpha, g, t1, t2)
  return sliding_window_m(xs, n)


# Exact Solution for t2


@j.jit
def t2_exact(g):
  a_squared = np.reciprocal(6 * g) * (np.sqrt(1 + (12 * g)) - 1)
  numerator = a_squared * (4 - a_squared)
  return np.divide(numerator, 3)


# # Alpha Tuning
#
#
# This section holds code that attempts to generate values of alpha that hold
# the inner product matrix down to some sane size.


def items_lte_trigger(xs, trigger):
  """Returns a pair of:

  bool: signals whether or not the entire list is returned.
  list: the list of elements <= the trigger value.
  """
  over = np.where(np.abs(xs) > trigger)[0]

  if over.size == 0:
    return True, xs

  return False, xs[:over[0] + 1]


def largest_elem(xs):
  """Returns a pair of k and the magnitude of the largest element in the list of
  correlators.

  """
  s = xs.size
  k = np.floor_divide(s, 2) * 2 - 1
  return k, np.abs(xs[s - k - 3])


def tune_alpha(f, alpha, target=1000, trigger=1e6):
  """Returns a pair of:

  - a suggested alpha,
  - the maximal size N of the inner product matrix that has a non-zero alpha.

  Parameters:
    f: function from alpha => list of correlators.
    alpha: starting value for the search.
    target: if the function has to recompute alpha, it aims for this value.
    trigger: recalculation won't trigger until some element of the list
             returned by f busts out beyond trigger.

  Example call:

  m.tune_alpha(
    lambda a: m.single_matrix_correlators(2 * n - 1, a, g, t1, t2),
    alpha=1,
    target=1e6,
    trigger=1e12
  )
  """

  xs = f(alpha)
  done, ys = items_lte_trigger(xs, trigger=trigger)

  if done:
    return alpha, np.floor_divide(xs.size + 1, 2)

  k, elem = largest_elem(ys)
  new_alpha = np.power(np.divide(target * np.power(alpha, k), elem),
                       np.reciprocal(k))

  if new_alpha == 0:
    return alpha, np.floor_divide(ys.size + 1, 2)

  return tune_alpha(f, new_alpha, target=target, trigger=trigger)
