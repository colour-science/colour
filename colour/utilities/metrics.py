"""
Metrics
=======

Define various metrics for evaluating signal quality and comparing data:

-   :func:`colour.utilities.metric_mse`
-   :func:`colour.utilities.metric_psnr`

References
----------
-   :cite:`Wikipedia2003c` : Wikipedia. (2003). Mean squared error. Retrieved
    March 5, 2018, from https://en.wikipedia.org/wiki/Mean_squared_error
-   :cite:`Wikipedia2004` : Wikipedia. (2004). Peak signal-to-noise ratio.
    Retrieved March 5, 2018, from
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
"""

from __future__ import annotations

import typing

from colour.algebra import sdiv, sdiv_mode

if typing.TYPE_CHECKING:
    from colour.hints import (
        ArrayLike,
        NDArrayFloat,
        Real,
        Tuple,
    )

from colour.utilities import array_namespace, as_float, as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "metric_mse",
    "metric_psnr",
]


def metric_mse(
    a: ArrayLike,
    b: ArrayLike,
    axis: int | Tuple[int] | None = None,
) -> NDArrayFloat:
    """
    Compute the mean squared error (MSE) between the specified arrays
    :math:`a` and :math:`b`.

    Parameters
    ----------
    a
        Variable :math:`a`.
    b
        Variable :math:`b`.
    axis
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array. If this is a tuple of
        integers, a mean is performed over multiple axes instead of a
        single axis or all the axes as before.

    Returns
    -------
    :class:`numpy.ndarray`
        Mean squared error (MSE).

    References
    ----------
    :cite:`Wikipedia2003c`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0.48222001, 0.31654775, 0.22070353])
    >>> b = a * 0.9
    >>> metric_mse(a, b)  # doctest: +ELLIPSIS
    np.float64(0.0012714...)
    """

    diff = as_float_array(a) - as_float_array(b)

    xp = array_namespace(diff)

    return as_float(xp.mean(diff**2, axis=axis))


def metric_psnr(
    a: ArrayLike,
    b: ArrayLike,
    max_a: Real = 1,
    axis: int | Tuple[int] | None = None,
) -> NDArrayFloat:
    """
    Compute the peak signal-to-noise ratio (PSNR) between the specified
    arrays :math:`a` and :math:`b`.

    Parameters
    ----------
    a
        Array :math:`a`.
    b
        Array :math:`b`.
    max_a
        Maximum possible value of the array :math:`a`.
    axis
        Axis or axes along which the means are computed. The default is
        to compute the mean of the flattened array. If this is a tuple
        of integers, a mean is performed over multiple axes, instead of
        a single axis or all the axes as before.

    Returns
    -------
    :class:`numpy.ndarray`
        Peak signal-to-noise ratio (PSNR).

    References
    ----------
    :cite:`Wikipedia2004`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0.48222001, 0.31654775, 0.22070353])
    >>> b = a * 0.9
    >>> metric_psnr(a, b)  # doctest: +ELLIPSIS
    np.float64(28.9568515...)
    """

    mse = as_float_array(metric_mse(a, b, axis))

    xp = array_namespace(mse)

    with sdiv_mode():
        psnr = xp.where(mse != 0, 10 * xp.log10(sdiv(max_a**2, mse)), 0)

    return as_float(psnr)
