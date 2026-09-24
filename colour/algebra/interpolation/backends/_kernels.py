"""
Interpolation kernels.

The kernels are pure elementwise functions (``abs`` / ``where`` / ``sinc``) that
resolve their namespace from the input array type, exactly as ``detect_backend``
does. They perform no *NumPy* coercion so that they stay differentiable with
respect to the query when evaluated with *PyTorch* or *JAX* arrays: the kernel
weights of :class:`KernelInterpolator` depend smoothly on the evaluation points.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np

from colour.constants import DTYPE_FLOAT_DEFAULT

if TYPE_CHECKING:
    from types import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "kernel_nearest_neighbour",
    "kernel_linear",
    "kernel_sinc",
    "kernel_lanczos",
    "kernel_cardinal_spline",
]


def _namespace(x: Any) -> tuple[ModuleType, Any]:
    """
    Return the array namespace matching the input type and the input coerced
    into it.

    *PyTorch* / *JAX* arrays are returned untouched (preserving any autodiff
    graph); everything else (scalars, sequences, *NumPy* arrays) is coerced to a
    floating *NumPy* array.
    """

    root = type(x).__module__.partition(".")[0]

    if root == "torch":
        return importlib.import_module("torch"), x

    if root in ("jax", "jaxlib"):
        return importlib.import_module("jax.numpy"), x

    return np, np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)


def kernel_nearest_neighbour(x: Any) -> Any:
    """
    Return the *nearest-neighbour* kernel evaluated at specified samples.

    The kernel equals 1 for :math:`|x| < 0.5` and 0 elsewhere.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> import numpy as np
    >>> kernel_nearest_neighbour(np.linspace(0, 1, 10))
    array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    """

    xp, x = _namespace(x)

    return xp.where(xp.abs(x) < 0.5, 1, 0)


def kernel_linear(x: Any) -> Any:
    """
    Evaluate the *linear* (triangular) kernel at specified samples.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> import numpy as np
    >>> kernel_linear(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([1.        , 0.8888888..., 0.7777777..., 0.6666666..., 0.5555555...,
           0.4444444..., 0.3333333..., 0.2222222..., 0.1111111..., 0.        ])
    """

    xp, x = _namespace(x)

    return xp.where(xp.abs(x) < 1, 1 - xp.abs(x), 0)


def kernel_sinc(x: Any, a: float = 3) -> Any:
    """
    Evaluate the *sinc* kernel at specified samples over the support
    :math:`[-a, a]`.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> import numpy as np
    >>> kernel_sinc(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([1.00000000e+00, 9.7981553...e-01, 9.2072542...e-01, 8.2699334...e-01,
           7.0531659...e-01, 5.6425327...e-01, 4.1349667...e-01, 2.6306440...e-01,
           1.2247694...e-01, 3.8981718...e-17])
    """

    if a < 1:
        error = '"a" must be equal or superior to 1!'
        raise ValueError(error)

    xp, x = _namespace(x)

    return xp.where(xp.abs(x) < a, xp.sinc(x), 0)


def kernel_lanczos(x: Any, a: float = 3) -> Any:
    """
    Return the *Lanczos* kernel evaluated at specified samples.

    Defined as :math:`L(x) = \\text{sinc}(x)\\,\\text{sinc}(x/a)` for
    :math:`|x| < a`, and zero otherwise.

    References
    ----------
    :cite:`Wikipedia2005b`

    Examples
    --------
    >>> import numpy as np
    >>> kernel_lanczos(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([1.00000000e+00, 9.7760615...e-01, 9.1243770...e-01, 8.1030092...e-01,
           6.8012706...e-01, 5.3295773...e-01, 3.8071690...e-01, 2.3492839...e-01,
           1.0554054...e-01, 3.2237621...e-17])
    """

    if a < 1:
        error = '"a" must be equal or superior to 1!'
        raise ValueError(error)

    xp, x = _namespace(x)

    return xp.where(xp.abs(x) < a, xp.sinc(x) * xp.sinc(x / a), 0)


def kernel_cardinal_spline(x: Any, a: float = 0.5, b: float = 0.0) -> Any:
    """
    Return the *cardinal spline* kernel evaluated at specified samples.

    Notable :math:`(a, b)` parameterizations: *Catmull-Rom* :math:`(0.5, 0)`,
    *Cubic B-Spline* :math:`(0, 1)`, *Mitchell-Netravalli*
    :math:`(\\frac{1}{3}, \\frac{1}{3})`.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> import numpy as np
    >>> kernel_cardinal_spline(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([1.        , 0.9711934..., 0.8930041..., 0.7777777..., 0.6378600...,
           0.4855967..., 0.3333333..., 0.1934156..., 0.0781893..., 0.        ])
    """

    xp, x = _namespace(x)

    x_abs = xp.abs(x)
    y = xp.where(
        x_abs < 1,
        (-6 * a - 9 * b + 12) * x_abs**3 + (6 * a + 12 * b - 18) * x_abs**2 - 2 * b + 6,
        (-6 * a - b) * x_abs**3
        + (30 * a + 6 * b) * x_abs**2
        + (-48 * a - 12 * b) * x_abs
        + 24 * a
        + 8 * b,
    )
    y = xp.where(x_abs >= 2, 0, y)

    return 1 / 6 * y
