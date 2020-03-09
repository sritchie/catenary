"""Main namespace for Catenary code."""

import jax as j
import jax.numpy as np
import numpy as onp


def ranged_fold_left(acc, f):
  """Takes a list and an updating function of of type:

  (acc: List[a], i: int) -> List[a]

  And returns an updated list, updated by calling the function on every element
  of acc.

  """

  def run(xs, i):
    new_v = f(xs, i)
    updated = j.ops.index_update(xs, i, new_v)
    return (updated, [])

  return j.lax.scan(run, acc, np.arange(acc.shape[0]))[0]


# 1. build an array using a recursive definition
# 2. build a matrix out of these correlators
#
# formula relating log of det with trace of

# what if I have more parameters?


def mapf(f, xs):
  if xs.shape[0] == 0:
    return xs
  else:
    j.lax.map(f, xs)


def t_k(k, g_recip, t):
  """
  t_{k} = 1/g * sum_0^{k-4}(t_l * t_{k-4-l}) - t_{k-2}

  """
  top = j.lax.min(k - 4, 1)

  def f(l):
    return t[l]

  return np.sum(mapf(f, t[:top - 1])) - t[k - 2]


def outer_f(max_idx):

  def inner_f(g, t1, t2):
    acc = np.zeros(max_idx,)
    acc1 = j.ops.index_update(acc, 0, 1)
    acc2 = j.ops.index_update(acc1, 1, t1)
    acc3 = j.ops.index_update(acc2, 2, t2)
    return f(acc3, g, 3, max_idx)

  return inner_f


# grad(outer_fn(10))


def loop_jax(g, max_idx):
  """Loop equations in Jax."""
  acc = np.arange(max_idx)
  g_recip = j.lax.reciprocal(g)

  def run(acc, i):
    ret = single_set(acc, i, g_recip)
    return (ret, [i])

  return j.lax.scan(run, acc, acc)


def loop_sympy(start, end, acc):
  """Loop equations in Sympy"""


def main():
  a = np.zeros(shape=10,)
  key = j.random.PRNGKey(0)

  # set t_0
  a2 = j.ops.index_update(a, 0, 1)
  # t_1 stays 0...
  # set t_2 and calculate!
  a3 = j.ops.index_update(a2, 2, 1.3)

  return a3


if __name__ == '__main__':
  main()
