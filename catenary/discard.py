"""Functions that seemed useful at the time."""

import jax as j
import jax.lax as l
import jax.numpy as np


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

  return l.scan(run, acc, np.arange(acc.shape[0]))[0]


def safe_map(f, xs):
  """equivalent to jax.lax.map, but handles empty arrays too.

  """
  if xs.shape[0] == 0:
    return xs

  return l.map(f, xs)
