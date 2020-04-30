"""Attempt at staying in mpmath mode.

STEPS:

- get the correlator values printing correctly with something like floats
- form the matrix
- take the min eigenvalue
- see if I can diff through iter

s.derive_by_array

writeup -

It works pretty well now... I can use this crazy block of code to verify that
for 20x20, for 39 correlators, I get the correct answer. Things break down when
I try and use numpy to get my final derivative.

[(lambda g, t2: la.eigvalsh(sliding_window_m_np(onp.array([x if isinstance(x, int) else x.evalf(100)  for x in f(g, 0, t2)[:39]], dtype=onp.float64), 20))[0])(g, t2_exact_sympy(g)) for g in [s.Rational(-1, 12) + s.Rational(x, 10) for x in range(1, 3)]]


So I need to stay in mpmath, OR do a doubledouble thing in jax. I guess that is fairly straightfoward.

# MPMath

"""

# First bits are copied over from symbolic.

import catenary.symbolic as cs
import sympy as s
import numpy as onp
from functools import partial

# This is how we keep high precision:
# t2_exact_sympy(s.Float(-0.02).evalf(100))
# This is the actual way: s.Float('-0.02', 100)

# correlator_m = cs.load_from_pickle()


def generate_functions(m, n):
  g = s.symbols('g')
  t = s.IndexedBase('t')

  expr = [v for i, v in sorted(m.items()) if i < n]
  f = s.lambdify([g, t[1], t[2]], expr)

  return f


def sliding_window_m_np(xs, window):
  """Returns matrix with `window` columns whose rows are a sliding window view
  onto the input array xs.

  The returned matrix will have dimensions (n - window + 1, window).

  """

  rows = xs.shape[0] - window + 1

  f = partial(onp.roll, xs, axis=0)
  wide = onp.vstack(list(map(f, -onp.arange(rows))))
  return wide[:, :window]


def sw(xs, window):
  return [xs[0 + x:window + x] for x in range(len(xs) - window + 1)]


def inner_product_matrix(xs, n):
  """Returns the... inner product matrix of correlators for the single matrix
  model.

  We do NOT need to do the transformation of t2 by alpha for this, vs the
  recursive solution. Since that happens already inside the equations.

  """
  return sliding_window_m_np(xs, n)


f = generate_functions(subm, 39)
# mpmath.diff(lambda g: inner_product_matrix(onp.array(f(g, 0, t2_exact_sympy(g))), 13), s.Rational(-2, 100)).shape

# mpmath.diff(lambda g: f(g, 0, t2_exact_sympy(g))[-1], 1.2)
