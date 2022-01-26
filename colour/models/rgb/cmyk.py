# -*- coding: utf-8 -*-
"""
CMYK Colour Transformations
===========================

Defines various Cyan-Magenta-Yellow (Black) (CMY(K)) colour transformations:

-   :func:`colour.RGB_to_CMY`
-   :func:`colour.CMY_to_RGB`
-   :func:`colour.CMY_to_CMYK`
-   :func:`colour.CMYK_to_CMY`

References
----------
-   :cite:`EasyRGBh` : EasyRGB. (n.d.). RGB --> CMY. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=11#text11
-   :cite:`EasyRGBi` : EasyRGB. (n.d.). CMY --> RGB. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=12#text12
-   :cite:`EasyRGBm` : EasyRGB. (n.d.). CMYK --> CMY. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=14#text14
-   :cite:`EasyRGBo` : EasyRGB. (n.d.). CMY --> CMYK. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=13#text13
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, NDArray
from colour.utilities import (
    as_float_array,
    from_range_1,
    to_domain_1,
    tsplit,
    tstack,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'RGB_to_CMY',
    'CMY_to_RGB',
    'CMY_to_CMYK',
    'CMYK_to_CMY',
]


def RGB_to_CMY(RGB: ArrayLike) -> NDArray:
    """
    Converts from *RGB* colourspace to *CMY* colourspace.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CMY* array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``CMY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`EasyRGBh`

    Examples
    --------
    >>> RGB = np.array([0.45620519, 0.03081071, 0.04091952])
    >>> RGB_to_CMY(RGB)  # doctest: +ELLIPSIS
    array([ 0.5437948...,  0.9691892...,  0.9590804...])
    """

    CMY = 1 - to_domain_1(RGB)

    return from_range_1(CMY)


def CMY_to_RGB(CMY: ArrayLike) -> NDArray:
    """
    Converts from *CMY* colourspace to *CMY* colourspace.

    Parameters
    ----------
    CMY
        *CMY* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``CMY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`EasyRGBi`

    Examples
    --------
    >>> CMY = np.array([0.54379481, 0.96918929, 0.95908048])
    >>> CMY_to_RGB(CMY)  # doctest: +ELLIPSIS
    array([ 0.4562051...,  0.0308107...,  0.0409195...])
    """

    RGB = 1 - to_domain_1(CMY)

    return from_range_1(RGB)


def CMY_to_CMYK(CMY: ArrayLike) -> NDArray:
    """
    Converts from *CMY* colourspace to *CMYK* colourspace.

    Parameters
    ----------
    CMY
        *CMY* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CMYK* array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``CMY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``CMYK``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`EasyRGBo`

    Examples
    --------
    >>> CMY = np.array([0.54379481, 0.96918929, 0.95908048])
    >>> CMY_to_CMYK(CMY)  # doctest: +ELLIPSIS
    array([ 0.        ,  0.9324630...,  0.9103045...,  0.5437948...])
    """

    C, M, Y = tsplit(to_domain_1(CMY))

    K = np.where(C < 1, C, 1)
    K = np.where(M < K, M, K)
    K = np.where(Y < K, Y, K)

    C = as_float_array((C - K) / (1 - K))
    M = as_float_array((M - K) / (1 - K))
    Y = as_float_array((Y - K) / (1 - K))

    C[np.asarray(K == 1)] = 0
    M[np.asarray(K == 1)] = 0
    Y[np.asarray(K == 1)] = 0

    CMYK = tstack([C, M, Y, K])

    return from_range_1(CMYK)


def CMYK_to_CMY(CMYK: ArrayLike) -> NDArray:
    """
    Converts from *CMYK* colourspace to *CMY* colourspace.

    Parameters
    ----------
    CMYK
        *CMYK* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CMY* array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``CMYK``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``CMY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`EasyRGBm`

    Examples
    --------
    >>> CMYK = np.array([0.50000000, 0.00000000, 0.74400000, 0.01960784])
    >>> CMYK_to_CMY(CMYK)  # doctest: +ELLIPSIS
    array([ 0.5098039...,  0.0196078...,  0.7490196...])
    """

    C, M, Y, K = tsplit(to_domain_1(CMYK))

    CMY = tstack([C * (1 - K) + K, M * (1 - K) + K, Y * (1 - K) + K])

    return from_range_1(CMY)
