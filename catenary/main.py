"""Main namespace for Catenary code.

Notes:

- Jax only takes the gradient with respect to the first argument. You don't
  have to curry; you can leave later arguments open for configuration.

CURRENT GOAL:

IDEAS:

- the matrix values are either super huge or super tiny at large N. Can we
  expand the N as we zoom in to the region?
- don't calculate the sum term so many times. return it and pass it along.
- can I pass a sparse vector to express the already-supplied terms? My parameters?
-

"""

import sys
from functools import partial
from time import time

import jax as j
import jax.interpreters.batching as jib
import jax.lax as jl
import jax.numpy as np
import jax.numpy.linalg as la
from jax.config import config

config.update('jax_enable_x64', True)


def t_k_plus_3_reference(k, g_recip, xs):
  """Clearer version of what's happening to calculate t_{k+3}."""

  def term(i, acc):
    return acc + (xs[i] * xs[k - 1 - i])

  sum_term = jl.fori_loop(0, k, term, 0.0)
  return g_recip * (sum_term - xs[k + 1])


def t_k_plus_3(k, g_recip, xs):
  """Code-golfed version."""
  n = xs.shape[0]

  # effectively mask all but the first k elements to be 0, then reverse the
  # array.
  ys = np.where(np.arange(n) >= n - k, xs[::-1], 0)

  ## "subtract" the (k + 2)'th element in the list - which has index t_{k+1},
  ## if we're zero-indexing - by setting the SECOND element in the not-rolled
  ## list to -1.
  ys = j.ops.index_update(ys, j.ops.index[1], -1)

  # Roll the array, bringing all of the reversed elements to the head, the
  # (k+2)nd element into position, so that the subtraction happens during the
  # dot product.
  return g_recip * np.dot(xs, np.roll(ys, k - n))


def correlators(g, xs):
  """Populates the supplied vector with correlator values, assuming the first
  three entries have been populated.

  """
  n = xs.shape[0]

  if n <= 2:
    return xs

  g_recip = jl.reciprocal(g)

  def run(k, xs):
    new_v = t_k_plus_3(k - 3, g_recip, xs)
    return j.ops.index_update(xs, k, new_v)

  return jl.fori_loop(3, n + 1, run, xs)


def initial_state(t1, t2, max_idx):
  """Generates an initial state for the list of correlators; this is all zeroes
  except for the populated values of t1 and t2, and the initial value of t_0 =
  1.

  """
  s = np.zeros(max_idx, dtype=np.double)
  return j.ops.index_update(s, j.ops.index[:3], [1, t1, t2])


@partial(j.jit, static_argnums=0)
def single_matrix_correlators(n, g, t1, t2):
  """This is a the function the we want to abstract, so we can explore more
  interesting models.

  This function is jitted for all but the final arg.

  """
  xs = initial_state(t1, t2, n)
  return correlators(g, xs)


def sliding_window_m(xs, window):
  """Returns matrix with `window` columns whose rows are a sliding window view
  onto the input array xs.

  The returned matrix will have dimensions (n - window + 1, window).

  """
  rows = xs.shape[0] - window + 1
  wide = j.vmap(partial(np.roll, xs, axis=0))(-np.arange(rows))
  return wide[:, :window]


@partial(j.jit, static_argnums=0)
def inner_product_matrix(n, g, t1, t2):
  """Returns the !"""
  xs = single_matrix_correlators((2 * n) - 1, g, t1, t2)
  return sliding_window_m(xs, n)


@partial(j.jit, static_argnums=0)
def min_eigenvalue(n, g, t1, t2):
  m = inner_product_matrix(n, g, t1, t2)
  return la.eigvalsh(m)[0]


def cake(n):
  # THIS gets it done!
  return j.jacfwd(min_eigenvalue, argnums=(0, 2))(1.2, 0.0, 2.5, n)


def jacfwd(fun, argnums=0, holomorphic=False):
  """Identical to j.jacfwd, except it also returns the calculated value."""

  def jacfun(*args, **kwargs):
    f = j.lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = j.api._argnums_partial(f, argnums, args)
    holomorphic or j.tree_util.tree_map(j.api._check_real_input_jacfwd,
                                        dyn_args)
    pushfwd = partial(j.jvp, f_partial, dyn_args)
    y, jac = j.vmap(pushfwd,
                    out_axes=(None, jib.last))(j.api._std_basis(dyn_args))
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = j.tree_util.tree_map(
        partial(j.api._unravel_array_into_pytree, example_args, -1), jac)
    return y, jac

  return jacfun


@partial(j.jit, static_argnums=(0, 1))
def update(n, g, t2):
  step_size = 0.0001
  ret, dt2 = jacfwd(min_eigenvalue, argnums=3)(n, g, 0, t2)
  return ret, t2 + step_size * dt2


def f(g, t2=2.5, n=5, steps=10000):
  t2 = 2.5
  ev = None

  for step in range(steps):
    new_ev, new_t2 = update(n, g, t2)

    if ev is not None and (np.abs((new_ev - ev) / ev) < 0.00001 or new_ev >= 0):
      break

    t2 = new_t2
    ev = new_ev

  return new_t2, new_ev


def bisection(f, l, r, n_steps):
  if n_steps <= 0:
    return

  fl = f(l)
  fr = f(r)

  if fl * fr >= 0:
    print(f"Bisection method fails for {l} and {r}.")
    return None

  for n in range(1, n_steps + 1):
    m_n = (a_n + b_n) / 2
    f_m_n = f(m_n)

    print(f"a_n: {a_n}, b_n: {b_n}")
    print(f"f_a_n: {f_a_n}, f_b_n: {f_b_n}")

    print(f"m_n: {m_n}")
    print(f"f_m_n: {f_m_n}")

    if f_a_n * f_m_n < 0:
      a_n = a_n
      b_n = m_n
      f_b_n = f_m_n
    elif f_b_n * f_m_n < 0:
      a_n = m_n
      f_a_n = f_m_n
      b_n = b_n
    elif f_m_n == 0:
      print("Found exact solution.")
      return m_n
    else:
      print(f"Bisection method fails for {a} and {b}.")
      return None
  return (a_n + b_n) / 2


def main():
  args = sys.argv[1:]
  print(time(f(float(args[0]), n=7)))


if __name__ == '__main__':
  print(bisection(partial(f, n=7), -1, -0.05, 20))
  #main()
