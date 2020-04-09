"""Utilities, original and stolen from other libraries."""

import sys
from functools import partial
from typing import Dict, Optional

import jax as j
import jax.numpy as np
import uv.types as t
import uv.util as u
from uv.reporter.base import AbstractReporter


class LoggingReporter(AbstractReporter):
  """Reporter that logs all data to the file handle you pass in using a fairly
  sane format. Compatible with tqdm, the python progress bar.

  """

  def __init__(self, file=sys.stdout, digits: int = 3):

    self._file = file
    self._digits = digits

  def _format(self, v: t.Metric) -> str:
    """Formats the value into something appropriate for logging."""
    if u.is_number(v):
      return "{num:.{digits}f}".format(num=float(v), digits=self._digits)

    return str(v)

  def report_all(self, step: int, m: Dict[t.MetricKey, t.Metric]) -> None:
    s = ", ".join(["{} = {}".format(k, self._format(v)) for k, v in m.items()])
    f = self._file
    print("Step {}: {}".format(step, s), file=f)


# Stopping Functions and stop function generators


def either(f1, f2):
  """Takes two stopping functions and returns a new function that stops if either
  stops."""

  def stop_fn(step, old_loss, loss):
    return f1(step, old_loss, loss) or f2(step, old_loss, loss)

  return stop_fn


def positive_loss(step, old_loss, loss) -> bool:
  """Returns true if the loss is positive, false otherwise."""
  return loss >= 0


@partial(j.jit, static_argnums=(0))
def max_step(n: int):
  """Returns a stopping function that stops of the index of steps increases
  beyond some n. (Assumes that the loop running is 1-indexed, so subtract 1 if
  that's not the case.)

  """
  return lambda i, old_loss, loss: i >= n


@partial(j.jit, static_argnums=(0, 1))
def stop_below(abs_tolerance: float, rel_tolerance: float):
  """Returns a function that stops at a certain tolerance."""

  def should_stop(_: int, old_loss: Optional[float], loss: float) -> bool:
    """Returns True if stop, false otherwise."""
    if old_loss is None:
      return False

    # Check absolute tolerance. If we pass this test, move to the relative
    # tolerance test.
    abs_diff = np.abs(loss - old_loss)
    if abs_diff < abs_tolerance:
      return True

    rel_diff = abs_diff / np.minimum(np.abs(loss), np.abs(old_loss))
    return rel_diff < rel_tolerance

  return should_stop
