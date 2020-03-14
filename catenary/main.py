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

import jax as j
import jax.lax as l
import jax.numpy as np
import functools

import numpy as onp


def t_k_plus_3(k, g_recip, state):
  """Takes an array of state and calculates just single term. I

  """
  top = k - 1

  def cake(i, acc):
    return acc + (state[i] * state[top - i])

  sum_term = l.fori_loop(0, top + 1, cake, 0.0)

  return g_recip * (sum_term - state[k + 1])


def correlators(g, state):
  """Populates the supplied vector with correlator values."""
  k = state.shape[0]

  if k <= 2:
    return state

  g_recip = l.reciprocal(g)

  def run(i, xs):
    new_v = t_k_plus_3(i - 3, g_recip, xs)
    return j.ops.index_update(xs, i, new_v)

  return l.fori_loop(3, k + 1, run, state)


def initial_state(t1, t2, max_idx):
  s = np.zeros(max_idx)
  return j.ops.index_update(s, j.ops.index[:3], [1, t1, t2])


@functools.partial(j.jit, static_argnums=3)
def single_matrix_correlators(g, t1, t2, max_idx):
  """This is a the function the we want to abstract, so we can explore more
  interesting models.

  This function is jitted for all but the final arg.

  """
  state = initial_state(t1, t2, max_idx)
  return correlators(g, state)


def sliding_window(xs, window):
  top = xs.shape[0] - window + 1
  wide = l.map(functools.partial(np.roll, xs, axis=0), -np.arange(top))
  return wide[:, :window]


@functools.partial(j.jit, static_argnums=3)
def correlator_matrix(g, t1, t2, n):
  """The jitted correlator matrix producer, so good!"""
  correlators = single_matrix_correlators(g, t1, t2, (2 * n) - 1)
  return sliding_window(correlators, n)


def smallest_eigenvalue(g, t1, t2):
  """This is the final fucntion..."""
  m = correlator_matrix(g, t1, t2, 100)
  return np.sum(m)


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
  a = np.zeros(shape=10,)
  key = j.random.PRNGKey(0)


if __name__ == '__main__':
  main()
