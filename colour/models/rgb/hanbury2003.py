# -*- coding: utf-8 -*-
"""
IHLS Colour Encoding
====================

Defines the :math:`IHLS` (Improved HLS) colourspace related transformations:

-   :func:`colour.RGB_to_IHLS`
-   :func:`colour.IHLS_to_RGB`

References
----------
-   :cite:`Hanbury2003` : Hanbury, A. (2003). A 3D-Polar Coordinate Colour
    Representation Well Adapted to Image Analysis. In J. Bigun & T. Gustavsson
    (Eds.), Image Analysis (pp. 804-811). Springer Berlin Heidelberg.
    ISBN:978-3-540-45103-7
"""

from __future__ import annotations

import numpy as np

from colour.algebra import vector_dot
from colour.hints import ArrayLike, NDArray
from colour.utilities import (
    from_range_1,
    to_domain_1,
    tstack,
    tsplit,
    zeros,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'RGB_to_IHLS',
    'IHLS_to_RGB',
]

MATRIX_RGB_TO_YC_1_C_2: NDArray = np.array([
    [0.2126, 0.7152, 0.0722],
    [1, -0.5, -0.5],
    [0, -np.sqrt(3) / 2, np.sqrt(3) / 2],
])
"""
*RGB* colourspace to *YC_1C_2* colourspace matrix.
"""

MATRIX_YC_1_C_2_TO_RGB: NDArray = np.linalg.inv(MATRIX_RGB_TO_YC_1_C_2)
"""
*YC_1C_2* colourspace to *RGB* colourspace matrix.
"""


def RGB_to_IHLS(RGB: ArrayLike) -> NDArray:
    """
    Converts from *RGB* colourspace to *IHLS* (Improved HLS) colourspace.

    Parameters
    ----------
    RGB
       *RGB* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *HYS* colourspace array.

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
    | ``HYS``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Hanbury2003`

    Examples
    --------
    >>> RGB = np.array([0.45595571, 0.03039702, 0.04087245])
    >>> RGB_to_IHLS(RGB)  # doctest: +ELLIPSIS
    array([ 6.2616051...,  0.1216271...,  0.4255586...])
    """

    RGB = to_domain_1(RGB)
    R, G, B = tsplit(RGB)

    Y, C_1, C_2 = tsplit(vector_dot(MATRIX_RGB_TO_YC_1_C_2, RGB))

    C = np.sqrt(C_1 ** 2 + C_2 ** 2)

    acos_C_1_C_2 = zeros(C.shape)
    acos_C_1_C_2[C != 0] = np.arccos(C_1[C != 0] / C[C != 0])
    H = np.where(C_2 <= 0, acos_C_1_C_2, (np.pi * 2) - acos_C_1_C_2)

    S = np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)

    HYS = tstack([H, Y, S])

    return from_range_1(HYS)


def IHLS_to_RGB(HYS: ArrayLike) -> NDArray:
    """
    Converts from *IHLS* (Improved HLS) colourspace to *RGB* colourspace.

    Parameters
    ----------
    HYS
        *IHLS* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``HYS``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Hanbury2003`

    Examples
    --------
    >>> HYS = np.array([6.26160518, 0.12162712, 0.42555869])
    >>> IHLS_to_RGB(HYS)  # doctest: +ELLIPSIS
    array([ 0.4559557...,  0.0303970...,  0.0408724...])
    """

    H, Y, S = tsplit(to_domain_1(HYS))

    pi_3 = np.pi / 3

    k = np.floor(H / (pi_3))
    H_s = H - k * (pi_3)
    C = (np.sqrt(3) * S) / (2 * np.sin((2 * pi_3) - H_s))

    C_1 = C * np.cos(H)
    C_2 = -C * np.sin(H)

    RGB = vector_dot(MATRIX_YC_1_C_2_TO_RGB, tstack([Y, C_1, C_2]))

    return from_range_1(RGB)
