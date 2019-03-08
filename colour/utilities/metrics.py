# -*- coding: utf-8 -*-
"""
Metrics
=======

Defines various metrics:

-   :func:`colour.utilities.metric_mse`
-   :func:`colour.utilities.metric_psnr`

References
----------
-   :cite:`Wikipedia2004` : Wikipedia. (2004). Peak signal-to-noise ratio.
    Retrieved March 5, 2018, from https://en.wikipedia.org/wiki/\
Peak_signal-to-noise_ratio
-   :cite:`Wikipedia2003c` : Wikipedia. (2003). Mean squared error. Retrieved
    March 5, 2018, from https://en.wikipedia.org/wiki/Mean_squared_error
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['metric_mse', 'metric_psnr']


def metric_mse(a, b):
    """
    Computes the mean squared error (MSE) or mean squared deviation (MSD)
    between given *array_like* :math:`a` and :math:`b` variables.

    Parameters
    ----------
    a : array_like
        :math:`a` variable.
    b : array_like
        :math:`b` variable.

    Returns
    -------
    float
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

    return np.mean((as_float_array(a) - as_float_array(b)) ** 2)


def metric_psnr(a, b, max_a=1):
    """
    Computes the peak signal-to-noise ratio (PSNR) between given *array_like*
    :math:`a` and :math:`b` variables.

    Parameters
    ----------
    a : array_like
        :math:`a` variable.
    b : array_like
        :math:`b` variable.
    max_a : numeric, optional
        Maximum possible pixel value of the :math:`a` variable.

    Returns
    -------
    float
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

    return 10 * np.log10(max_a ** 2 / metric_mse(a, b))
