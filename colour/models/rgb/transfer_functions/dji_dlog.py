# -*- coding: utf-8 -*-
"""
DJI D-Log Log Encoding
======================

Defines the *DJI D-Log* log encoding:

-   :func:`colour.models.log_encoding_DJIDLog`
-   :func:`colour.models.log_decoding_DJIDLog`

References
----------
-   :cite:`DJI2017` : Dji. (2017). White Paper on D-Log and D-Gamut of DJI
    Cinema Color System (pp. 1-5).
    https://dl.djicdn.com/downloads/zenmuse+x7/20171010/\
D-Log_D-Gamut_Whitepaper.pdf
"""

from __future__ import annotations

import numpy as np

from colour.hints import FloatingOrArrayLike, FloatingOrNDArray
from colour.utilities import as_float, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'log_encoding_DJIDLog',
    'log_decoding_DJIDLog',
]


def log_encoding_DJIDLog(x: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *DJI D-Log* log encoding curve.

    Parameters
    ----------
    x
        Linear reflection data :math`x`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *DJI D-Log* encoded data :math:`y`.

    References
    ----------
    :cite:`DJI2017`

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_DJIDLog(0.18)  # doctest: +ELLIPSIS
    0.3987645...
    """

    x = to_domain_1(x)

    y = np.where(x <= 0.0078, 6.025 * x + 0.0929,
                 (np.log10(x * 0.9892 + 0.0108)) * 0.256663 + 0.584555)

    return as_float(from_range_1(y))


def log_decoding_DJIDLog(y: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *DJI D-Log* log decoding curve.

    Parameters
    ----------
    y
        *DJI D-Log* encoded data :math:`y`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear reflection data :math`x`.

    References
    ----------
    :cite:`DJI2017`

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_decoding_DJIDLog(0.3987645561893306)  # doctest: +ELLIPSIS
    0.1799998...
    """

    y = to_domain_1(y)

    x = np.where(y <= 0.14, (y - 0.0929) / 6.025,
                 (10 ** (3.89616 * y - 2.27752) - 0.0108) / 0.9892)

    return as_float(from_range_1(x))
