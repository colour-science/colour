"""
Metrics
=======

Defines various metrics:

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

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.hints import (
    ArrayLike,
    NDArrayFloat,
    Real,
    Tuple,
)
from colour.utilities import as_float, as_float_array, zeros

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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
    Compute the mean squared error (MSE) or mean squared deviation (MSD)
    between given variables :math:`a` and :math:`b`.

    Parameters
    ----------
    a
        Variable :math:`a`.
    b
        Variable :math:`b`.
    axis
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.

    Returns
    -------
    :class:`numpy.ndarray`
        Mean squared error (MSE).

    References
    ----------
    :cite:`Wikipedia2003c`

    Examples
    --------
    >>> a = np.array([0.48222001, 0.31654775, 0.22070353])
    >>> b = a * 0.9
    >>> metric_mse(a, b)  # doctest: +ELLIPSIS
    0.0012714...
    """

    return as_float(
        np.mean((as_float_array(a) - as_float_array(b)) ** 2, axis=axis)
    )


def metric_psnr(
    a: ArrayLike,
    b: ArrayLike,
    max_a: Real = 1,
    axis: int | Tuple[int] | None = None,
) -> NDArrayFloat:
    """
    Compute the peak signal-to-noise ratio (PSNR) between given variables
    :math:`a` and :math:`b`.

    Parameters
    ----------
    a
        Variable :math:`a`.
    b
        Variable :math:`b`.
    max_a
        Maximum possible pixel value of the :math:`a` variable.
    axis
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.

    Returns
    -------
    :class:`numpy.ndarray`
        Peak signal-to-noise ratio (PSNR).

    References
    ----------
    :cite:`Wikipedia2004`

    Examples
    --------
    >>> a = np.array([0.48222001, 0.31654775, 0.22070353])
    >>> b = a * 0.9
    >>> metric_psnr(a, b)  # doctest: +ELLIPSIS
    28.9568515...
    """

    mse = as_float_array(metric_mse(a, b, axis))
    psnr = zeros(mse.shape)

    with sdiv_mode():
        psnr[mse != 0] = 10 * np.log10(sdiv(max_a**2, mse[mse != 0]))

    return as_float(psnr)
