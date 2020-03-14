"""Tests for Catenary."""

import catenary.main as m
import jax.numpy as np

import hypothesis.strategies as st
from hypothesis import given


def small_lists():
  return st.lists(st.integers(min_value=-10000, max_value=10000),
                  min_size=1,
                  max_size=100)


@given(small_lists())
def test_ranged_fold_left(xs):

  # this does not currently work with lists.
  acc = np.array(xs)

  def squared_map(acc, i):
    return acc[i] * acc[i]

  result = m.ranged_fold_left(acc, squared_map)

  assert list(result) == list(map(lambda x: x * x, xs))


@given(small_lists())
def test_safe_map(xs):

  def square(x):
    return x * x

  result = m.safe_map(square, np.array(xs))
  assert list(result) == list(map(square, xs))
