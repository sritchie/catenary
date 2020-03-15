"""Tests for Catenary."""

import catenary.main as m
import jax.numpy as np

import hypothesis.strategies as st
from hypothesis import given


def small_lists():
  return st.lists(st.integers(min_value=-10000, max_value=10000),
                  min_size=1,
                  max_size=100)
