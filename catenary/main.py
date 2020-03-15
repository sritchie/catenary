"""Main namespace for Catenary code.

Notes:

- Jax only takes the gradient with respect to the first argument. You don't
  have to curry; you can leave later arguments open for configuration.

CURRENT GOAL:

- Get a test going that can generate and solve these equations. We want to
  solve them all... the zeros compose, remember. How can I bake that in?


IDEAS:

- the matrix values are either super huge or super tiny at large N. Can we
  expand the N as we zoom in to the region?
- don't calculate the sum term so many times. return it and pass it along.

"""

import functools

import jax as j
import jax.lax as jl
import jax.numpy as np
import jax.numpy.linalg as la
from jax.config import config

config.update('jax_enable_x64', True)


def t_k_plus_3(k, g_recip, xs):
  """Takes an array of state and calculates just single term."""

  def term(i, acc):
    return acc + (xs[i] * xs[k - 1 - i])

  sum_term = jl.fori_loop(0, k, term, 0.0)

  return g_recip * (sum_term - xs[k + 1])


def correlators(g, xs):
  """Populates the supplied vector with correlator values."""
  k = xs.shape[0]

  if k <= 2:
    return xs

  g_recip = jl.reciprocal(g)

  def run(i, xs):
    new_v = t_k_plus_3(i - 3, g_recip, xs)
    return j.ops.index_update(xs, i, new_v)

  return jl.fori_loop(3, k + 1, run, xs)


def initial_state(t1, t2, max_idx):
  s = np.zeros(max_idx, dtype=np.double)
  return j.ops.index_update(s, j.ops.index[:3], [1, t1, t2])


@functools.partial(j.jit, static_argnums=3)
def single_matrix_correlators(g, t1, t2, max_idx):
  """This is a the function the we want to abstract, so we can explore more
  interesting models.

  This function is jitted for all but the final arg.

  """
  state = initial_state(t1, t2, max_idx)
  return correlators(g, state)


def rolling_rows(xs, window):
  """"""
  top = xs.shape[0] - window + 1
  wide = j.vmap(functools.partial(np.roll, xs, axis=0))(-np.arange(top))
  return wide[:, :window]


@functools.partial(j.jit, static_argnums=3)
def correlator_matrix(g, t1, t2, n):
  """The jitted correlator matrix producer, so good!"""
  correlators = single_matrix_correlators(g, t1, t2, (2 * n) - 1)
  return rolling_rows(correlators, n)


@functools.partial(j.jit, static_argnums=3)
def smallest_eigenvalue(g, t1, t2, n):
  """This is the final fucntion...

  this gets us a forward pass derivative, NICE


  """

  m = correlator_matrix(g, t1, t2, n)
  mev = la.eigvalsh(m)[0]
  return np.abs(mev)


def cake(n):
  # THIS gets it done!
  return j.jacfwd(smallest_eigenvalue, argnums=(0, 2))(1.2, 0.0, 2.5, n)


@functools.partial(j.jit, static_argnums=3)
def update(g, t1, t2, n):
  step_size = 0.0001
  (dg, dt1, dt2) = j.jacfwd(smallest_eigenvalue, argnums=(0, 2))(g, t1, t2, n)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]


# So I THINK I can do this:
#
# j.jvp(test_out, (1.2,), (1.,))
#
# that gets me the sum...

# 1. build the list of correlators using a recursive definition DONE
# 2. build a scipy matrix out of these correlators DONE
#
# 3a. calculate smallest eigenvalue and do gradient descent toward zero
# 3b. calculate determinant, push it toward 0
#
# challenge in multi-matrix case...

# grad(outer_fn(10))


def main():
  cake(100)


if __name__ == '__main__':
  main()
