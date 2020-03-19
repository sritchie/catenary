"""Functions imported from Jax that haven't yet been updated upstream.

"""

from functools import partial

import jax as j
import jax.interpreters.batching as jib


def jacfwd(fun, argnums=0, holomorphic=False):
  """Identical to j.jacfwd, except it also returns the calculated value."""

  def jacfun(*args, **kwargs):
    f = j.lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = j.api._argnums_partial(f, argnums, args)
    holomorphic or j.tree_util.tree_map(j.api._check_real_input_jacfwd,
                                        dyn_args)
    pushfwd = partial(j.jvp, f_partial, dyn_args)
    y, jac = j.vmap(pushfwd,
                    out_axes=(None, jib.last))(j.api._std_basis(dyn_args))
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = j.tree_util.tree_map(
        partial(j.api._unravel_array_into_pytree, example_args, -1), jac)
    return y, jac

  return jacfun
