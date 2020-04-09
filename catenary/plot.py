"""Plotting helpers."""

from typing import Dict, List

import matplotlib.pyplot as plt
import uv.types as t


def plot_metrics(m: Dict[t.MetricKey, List[t.Metric]],
                 nrows: int,
                 ncols: int,
                 every_nth: int = 1,
                 **kwargs):
  """This is a tighter way to do things that makes some assumptions."""
  assert nrows * ncols >= len(m), "You don't have enough spots!"

  fig, ax = plt.subplots(nrows, ncols, **kwargs)

  for i, (k, v) in enumerate(m.items()):
    row, col = divmod(i, ncols)
    xs, ys = zip(*[(m["step"], float(m["value"])) for m in v])
    ax[row, col].plot(xs, ys, '+-')
    ax[row, col].set_title(k)

    for n, label in enumerate(ax[row, col].get_yticklabels()):
      if n % every_nth != 0:
        label.set_visible(False)

  fig.tight_layout()
  plt.show()
