"""Sympy version!


Next steps:
- Move this sympy investigation to a different file.
- look up how to diff etc in mpmath

"""

from functools import partial

import jax as j
import jax.numpy as np
import jax.numpy.linalg as la
import sympy as s
from sympy.utilities.lambdify import (MODULES, NUMPY, NUMPY_DEFAULT,
                                      NUMPY_TRANSLATIONS)

import catenary.single as cs
import catenary.symbolic as sym


def print_scinote(xs):
  """Print all entries in scientific notation for easy comparison.

  """
  return ['{:.15e}'.format(x) for x in xs]


# Report on what I've tried.

# This is the JAX recursive code:
#
# cs.single_matrix_correlators(39, 1, -0.02, 0, cs.t2_exact(-0.02))
recursive = [
    1.000000000000000e+00, 0.000000000000000e+00, 1.044103676767703e+00,
    -0.000000000000000e+00, 2.205183838385150e+00, -0.000000000000000e+00,
    5.848824242487183e+00, -0.000000000000000e+00, 1.741520389385238e+01,
    -0.000000000000000e+00, 5.563371508323574e+01, -0.000000000000000e+00,
    1.863456870879013e+02, -0.000000000000000e+00, 6.458117350396407e+02,
    -0.000000000000000e+00, 2.296471528260753e+03, -0.000000000000000e+00,
    8.331877595935770e+03, -0.000000000000000e+00, 3.072098803228173e+04,
    -0.000000000000000e+00, 1.147952789537119e+05, -0.000000000000000e+00,
    4.343122918507295e+05, -0.000000000000000e+00, 1.686594434338086e+06,
    -0.000000000000000e+00, 7.936485499678843e+06, -0.000000000000000e+00,
    1.002148079939317e+08, -0.000000000000000e+00, 3.714576365185739e+09,
    -0.000000000000000e+00, 1.735783390256434e+11, -0.000000000000000e+00,
    8.291489993052295e+12, -0.000000000000000e+00, 3.967906943184581e+14
]

# Generated this with
#
# single_matrix_correlators(1, -0.02, 0, cs.t2_exact(-0.02))
sympy_numeric = [
    1.000000000000000e+00, 0.000000000000000e+00, 1.044103676767703e+00,
    -0.000000000000000e+00, 2.205183838385150e+00, 0.000000000000000e+00,
    5.848824242487183e+00, 0.000000000000000e+00, 1.741520389386300e+01,
    0.000000000000000e+00, 5.563371508293669e+01, 0.000000000000000e+00,
    1.863456870844050e+02, 0.000000000000000e+00, 6.458117355773950e+02,
    0.000000000000000e+00, 2.296471463033545e+03, 0.000000000000000e+00,
    8.331875557629688e+03, 0.000000000000000e+00, 3.072097213772779e+04,
    0.000000000000000e+00, 1.147915533525885e+05, 0.000000000000000e+00,
    4.343336795641596e+05, 0.000000000000000e+00, 1.696420507462035e+06,
    0.000000000000000e+00, 8.618730314790826e+06, 0.000000000000000e+00,
    1.036617249891930e+08, 0.000000000000000e+00, 9.817873300747375e+08,
    0.000000000000000e+00, 3.185243111321702e+11, 0.000000000000000e+00,
    8.784782650953000e+12, 0.000000000000000e+00, 2.841086340720340e+14
]

# Generated in Mathematica from correlators entries using g = -0.02.
mathematica_numeric = [
    1.000000000000000e+00, 0.000000000000000e+00, 1.044100000000000e+00,
    0.000000000000000e+00, 2.205180000000000e+00, 0.000000000000000e+00,
    5.848820000000000e+00, 0.000000000000000e+00, 1.741520000000000e+01,
    0.000000000000000e+00, 5.563370000000000e+01, 0.000000000000000e+00,
    1.863460000000000e+02, 0.000000000000000e+00, 6.458120000000000e+02,
    0.000000000000000e+00, 2.296470000000000e+03, 0.000000000000000e+00,
    8.331870000000001e+03, 0.000000000000000e+00, 3.072090000000000e+04,
    0.000000000000000e+00, 1.147920000000000e+05, 0.000000000000000e+00,
    4.347600000000000e+05, 0.000000000000000e+00, 1.676410000000000e+06,
    0.000000000000000e+00, 8.211530000000000e+06, 0.000000000000000e+00,
    1.443230000000000e+08, 0.000000000000000e+00, 2.352270000000000e+09,
    0.000000000000000e+00, 1.776550000000000e+11, 0.000000000000000e+00,
    1.319990000000000e+13, 0.000000000000000e+00, 2.097390000000000e+14
]

# This entry is what happens when I do a fully symbolic evaluation of the sympy
# entries:


def get_sympy_symbolic(m, n, g, alpha, t1, t2):
  """Takes a dict of index to expression, like the one generated by
  load_from_pickle().

  """
  alpha_sym, g_sym = s.symbols('alpha g')
  t = s.IndexedBase('t')

  def term(expr):
    return s.simplify(
        expr.subs({
            t[1]: alpha * t1,
            t[2]: alpha**2 * t2,
            g_sym: g,
            alpha_sym: alpha
        }))

  return {k: term(v) for k, v in m.items() if k < n}


# And this is what I get with a purely symbolic evaluation in Mathematica.
# Generated from the `correlators` entry.
mathematica_sym = [
    1.000000000000000e+00, 0.000000000000000e+00, 1.044103676767700e+00,
    0.000000000000000e+00, 2.205183838385140e+00, 0.000000000000000e+00,
    5.848824242486660e+00, 0.000000000000000e+00, 1.741520389382730e+01,
    0.000000000000000e+00, 5.563371508203750e+01, 0.000000000000000e+00,
    1.863456870305480e+02, 0.000000000000000e+00, 6.458117322945510e+02,
    0.000000000000000e+00, 2.296471396872550e+03, 0.000000000000000e+00,
    8.331871307302830e+03, 0.000000000000000e+00, 3.072068703946560e+04,
    0.000000000000000e+00, 1.147808725356120e+05, 0.000000000000000e+00,
    4.336227575119600e+05, 0.000000000000000e+00, 1.653591254709760e+06,
    0.000000000000000e+00, 6.356854429350010e+06, 0.000000000000000e+00,
    2.460893163750270e+07, 0.000000000000000e+00, 9.585257138642250e+07,
    0.000000000000000e+00, 3.753748264750530e+08, 0.000000000000000e+00,
    1.477127527122290e+09, 0.000000000000000e+00, 5.837729364778010e+09
]


def generate_sympy_symbolic(m=None, alpha=1):
  """Generates the symbolic evaluation of -0.02 with the supplied alpha."""
  if m is None:
    m = sym.load_from_pickle()

  g = s.Rational(-2, 100)
  n = 39
  t1 = 0
  t2 = sym.t2_exact_sympy(g)

  xs = get_sympy_symbolic(m, n, g, alpha, t1, t2)
  return [xs[i].evalf() for i in range(len(xs))]


sympy_symbolic = [
    1.000000000000000e+00, 0.000000000000000e+00, 1.044103676767703e+00,
    0.000000000000000e+00, 2.205183838385139e+00, 0.000000000000000e+00,
    5.848824242486660e+00, 0.000000000000000e+00, 1.741520389382735e+01,
    0.000000000000000e+00, 5.563371508203746e+01, 0.000000000000000e+00,
    1.863456870305482e+02, 0.000000000000000e+00, 6.458117322945513e+02,
    0.000000000000000e+00, 2.296471396872554e+03, 0.000000000000000e+00,
    8.331871307302832e+03, 0.000000000000000e+00, 3.072068703946557e+04,
    0.000000000000000e+00, 1.147808725356123e+05, 0.000000000000000e+00,
    4.336227575119596e+05, 0.000000000000000e+00, 1.653591254709761e+06,
    0.000000000000000e+00, 6.356854429350010e+06, 0.000000000000000e+00,
    2.460893163750270e+07, 0.000000000000000e+00, 9.585257138642249e+07,
    0.000000000000000e+00, 3.753748264750528e+08, 0.000000000000000e+00,
    1.477127527122289e+09, 0.000000000000000e+00, 5.837729364778011e+09
]


def ssanr_fn(m=None, alpha=1):
  """This is an attempt to jit-compile from the very beginning. How well can we
  do if we go from expressions and evaluate, with floating point inputs?

  """
  if m is None:
    m = sym.load_from_pickle()

  n = 39
  g = s.symbols('g')
  t1 = 0

  # let's keep t2 around.
  t = s.IndexedBase('t')
  xs = get_sympy_symbolic(m, n, g, alpha, t1, t[2])

  expr = [v.evalf() for i, v in sorted(xs.items())]
  f = s.lambdify([g, t[2]], expr, 'jax')

  def ret(g, t2):
    return np.array(f(g, t2))

  return j.jit(ret)


def generate_sympy_symbolic_alpha_numeric_rest(f):
  g = -0.02
  t2 = cs.t2_exact(g)
  return f(g, t2)


# Not that well!
sympy_symbolic_alpha_numeric_rest = [
    1.000000000000000e+00, 0.000000000000000e+00, 1.044103676767703e+00,
    0.000000000000000e+00, 2.205183838385150e+00, 0.000000000000000e+00,
    5.848824242487005e+00, 0.000000000000000e+00, 1.741520389383910e+01,
    0.000000000000000e+00, 5.563371508338076e+01, 0.000000000000000e+00,
    1.863456870754287e+02, 0.000000000000000e+00, 6.458117335177149e+02,
    0.000000000000000e+00, 2.296471519891430e+03, 0.000000000000000e+00,
    8.331876855116603e+03, 0.000000000000000e+00, 3.072065171694227e+04,
    0.000000000000000e+00, 1.147899050119028e+05, 0.000000000000000e+00,
    4.336808689942017e+05, 0.000000000000000e+00, 1.653408313040394e+06,
    0.000000000000000e+00, 8.131516293641280e+06, 0.000000000000000e+00,
    1.016439536705160e+08, 0.000000000000000e+00, 6.776263578034401e+09,
    0.000000000000000e+00, 8.470329472543001e+10, 0.000000000000000e+00,
    3.388131789017200e+13, 0.000000000000000e+00, 0.000000000000000e+00
]


# lets try a hardcore hand tuned alpha on the symbolic problem, then
# jit-compiling.
def highly_tuned(m=None):
  """I tuned alpha by calling

  generate_sympy_symbolic(correlator_m, alpha)

  a ton and trying for a max value of 1.

  alpha = s.Rational(5534, 10000) worked!

  """
  if m is None:
    m = sym.load_from_pickle()

  alpha = s.Rational(5534, 10000)
  f = ssanr_fn(m, alpha)
  g = -0.02
  t2 = cs.t2_exact(g)
  return f(g, t2)


highly_tuned_half_numeric = [
    1.000000000000000e+00, 0.000000000000000e+00, 3.197583798118448e-01,
    0.000000000000000e+00, 2.068242319008188e-01, 0.000000000000000e+00,
    1.679977633731789e-01, 0.000000000000000e+00, 1.531940161760719e-01,
    0.000000000000000e+00, 1.498751428626687e-01, 0.000000000000000e+00,
    1.537408235798958e-01, 0.000000000000000e+00, 1.631751794022531e-01,
    0.000000000000000e+00, 1.777000461866715e-01, 0.000000000000000e+00,
    1.974456767695500e-01, 0.000000000000000e+00, 2.229531028397529e-01,
    0.000000000000000e+00, 2.551441478392077e-01, 0.000000000000000e+00,
    2.951956679283261e-01, 0.000000000000000e+00, 3.390155017810941e-01,
    0.000000000000000e+00, 5.191201314232151e-01, 0.000000000000000e+00,
    2.649689167929410e+00, 0.000000000000000e+00, 4.057357205967418e+01,
    0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00,
    1.902698026949941e+04, 0.000000000000000e+00, 0.000000000000000e+00
]


def generate_highly_tuned_symbolic(g, t2, alpha=None, m=None):
  if m is None:
    m = sym.load_from_pickle()

  if alpha is None:
    alpha = s.Rational(5534, 10000)

  n = 39
  t1 = 0

  xs = get_sympy_symbolic(m, n, g, alpha, t1, t2)
  print(xs[38].evalf())
  return [float(xs[i].evalf()) for i in range(len(xs))]


# generate_highly_tuned_symbolic(
# s.Rational(-2, 100).evalf(), t2_exact_sympy(s.Rational(-2, 100)), m=correlator_m)
tuned_symbolic_g_evaluated = [
    1.000000000000000e+00, 0.000000000000000e+00, 3.197583798118447e-01,
    0.000000000000000e+00, 2.068242319007974e-01, 0.000000000000000e+00,
    1.679977633729028e-01, 0.000000000000000e+00, 1.531940161791548e-01,
    0.000000000000000e+00, 1.498751428444983e-01, 0.000000000000000e+00,
    1.537408233869445e-01, 0.000000000000000e+00, 1.631751664699474e-01,
    0.000000000000000e+00, 1.777000622513795e-01, 0.000000000000000e+00,
    1.974493008097425e-01, 0.000000000000000e+00, 2.229794253212150e-01,
    0.000000000000000e+00, 2.547209760346844e-01, 0.000000000000000e+00,
    2.491900984212964e-01, 0.000000000000000e+00, -1.510322991202021e-01,
    0.000000000000000e+00, -3.596553793078173e+00, 0.000000000000000e+00,
    -1.216629511593756e+02, 0.000000000000000e+00, 4.187657008750927e+02,
    0.000000000000000e+00, 8.186598465636956e+04, 0.000000000000000e+00,
    3.636514669884911e+05, 0.000000000000000e+00, 2.041890106427151e+05
]

# generate_highly_tuned_symbolic(
#  s.Rational(-2, 100), t2_exact_sympy(s.Rational(-2, 100)), m=correlator_m)
tuned_symbolic_all_symbols = [
    1.000000000000000e+00, 0.000000000000000e+00, 3.197583798118447e-01,
    0.000000000000000e+00, 2.068242319008178e-01, 0.000000000000000e+00,
    1.679977633731689e-01, 0.000000000000000e+00, 1.531940161759685e-01,
    0.000000000000000e+00, 1.498751428590502e-01, 0.000000000000000e+00,
    1.537408235428679e-01, 0.000000000000000e+00, 1.631751789836234e-01,
    0.000000000000000e+00, 1.777000366675201e-01, 0.000000000000000e+00,
    1.974455452995505e-01, 0.000000000000000e+00, 2.229533591906351e-01,
    0.000000000000000e+00, 2.551120235115831e-01, 0.000000000000000e+00,
    2.951561129028659e-01, 0.000000000000000e+00, 3.447038957510510e-01,
    0.000000000000000e+00, 4.058248163854691e-01, 0.000000000000000e+00,
    4.811355022078455e-01, 0.000000000000000e+00, 5.739270864342895e-01,
    0.000000000000000e+00, 6.883295398391586e-01, 0.000000000000000e+00,
    8.295213428590618e-01, 0.000000000000000e+00, 1.003995670356674e+00
]

# 11x11 matrix seems to be the only thing that still works:
#
# la.eigvalsh(cs.sliding_window_m(np.array(list(map(float, generate_highly_tuned_symbolic(s.Rational(-2, 100).evalf(), t2_exact_sympy(s.Rational(-2, 100)).evalf(), m=correlator_m)))), 20)[:11,:11])


def generate_partial_symbolic(g, alpha=None, m=None):
  """what happens if we force coefficient evaluation, AND rational g... then
  evaluate for t2? Can we keep precision all the way through to jax?

  """
  if m is None:
    m = sym.load_from_pickle()

  if alpha is None:
    alpha = s.Rational(5534, 10000)

  n = 39
  t1 = 0

  t = s.IndexedBase('t')
  xs = get_sympy_symbolic(m, n, g, alpha, t1, t[2])

  expr = [v.evalf(100) for i, v in sorted(xs.items())]
  print(expr[38])
  f = s.lambdify([t[2]], expr, 'jax')

  def ret(t2):
    return np.array(f(t2))

  return j.jit(ret)


def ssanr_fn2(m=None, alpha=1):
  """This is an attempt to jit-compile from the very beginning. How well can we
  do if we go from expressions and evaluate, with floating point inputs?

  """
  if m is None:
    m = sym.load_from_pickle()

  n = 39
  g = s.symbols('g')
  t1 = 0

  # let's keep t2 around.
  t = s.IndexedBase('t')
  xs = get_sympy_symbolic(m, n, g, alpha, t1, t[2])

  expr = [v.evalf() for i, v in sorted(xs.items())]
  f = s.lambdify([g, t[2]], expr, 'jax')

  def ret(g, t2):
    return np.array(f(g, t2))

  return j.jit(ret)


def checker(f, n):
  """Takes a function of g, t2 and prints the eigenvalues."""
  for g in np.arange(-1 / 12, 1, 0.05):
    t2 = cs.t2_exact(g)
    ipm = cs.sliding_window_m(np.array(f(g, t2)), 20)[:n, :n]
    ev = la.eigvalsh(ipm)

    print(
        f"for {g}, {t2}: {all([x >= 0 for x in ev])} with exact soln min ev: {min(ev)}"
    )


def run_checks(m, n):
  checker(partial(generate_highly_tuned_symbolic, m=m), n)


# f1 = partial(generate_highly_tuned_symbolic, m=correlator_m)
#

# Note that this does NOT work at all if you try and just hijack that value for
# the recursive solution.
# cs.single_matrix_correlators(39, float(alpha.evalf()), -0.02, 0, cs.t2_exact(-0.02))
