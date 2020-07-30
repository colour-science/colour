# -*- coding: utf-8 -*-
"""
IHLS Colour Encoding
====================

Defines the :math:`IHLS` colour encoding related transformations:

-   :func:`colour.RGB_to_IHLS`
-   :func:`colour.IHLS_to_RGB`

References
----------
-   :cite:`Hanbury2003` : Hanbury, A. (2003). A 3D-Polar Coordinate Colour
    Representation Well Adapted to Image Analysis. In J. Bigun & T. Gustavsson
    (Eds.), Image Analysis (pp. 804–811). Springer Berlin Heidelberg.
    ISBN:978-3-540-45103-7
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import (dot_vector, from_range_1, to_domain_1, tstack,
                              tsplit)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['RGB_to_IHLS', 'IHLS_to_RGB']


def RGB_to_IHLS(RGB):
    """
    Converts from *RGB* colourspace to *IHLS* colourspace.

    Parameters
    ----------
    RGB : (..., 3) array-like
       *RGB* colourspace array.

    Returns
    -------
     HLS : array_like (..., 3) ndarray
        *HLS* colourspace array.

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
    | ``HLS``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Hanbury2003`

    Examples
    --------
    >>> RGB = np.array([0.45595571, 0.03039702, 0.04087245])
    >>> RGB_to_IHLS(RGB)  # doctest: +ELLIPSIS
    array([  3.5997842...e+02,   1.2162712...e-01,  -1.5791520...e-01])
    """

    RGB = to_domain_1(RGB)

    Y, C_1, C_2 = tsplit(
        dot_vector([
            [0.2125, 0.7154, 0.0721],
            [1, -0.5, -0.5],
            [0, -np.sqrt(3) / 2, -np.sqrt(3) / 2],
        ], RGB))

    C = np.sqrt(C_1 ** 2 + C_2 ** 2)

    acos_C_1_C_2 = np.arccos(C_1 / C)
    H = np.where(C_2 <= 0, acos_C_1_C_2, (np.pi * 2) - acos_C_1_C_2)

    k = np.arange(6).reshape([1] * H.ndim + [6])
    H_s = H[..., np.newaxis] - k * (np.pi / 3)
    H_s = H_s[np.logical_and(H_s >= 0, H_s <= np.pi / 3)].reshape(H.shape)
    S = (2 * C * np.sin(2 / 3 * np.pi - H_s)) / np.sqrt(3)

    IHLS = tstack([H, Y, S])

    return from_range_1(IHLS)


def RGB_to_IHLS2(RGB):

    R, G, B = tsplit(to_domain_1(RGB))

    Y = 0.2125 * R + 0.7154 * G + 0.0721 * B
    S = np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)
    H_p = np.arccos((R - 0.5 * G - 0.5 * B) /
                    (R ** 2 + G ** 2 + B ** 2 - R * G - R * B - B * G) ** 0.5)
    H = np.where(B > G, 2 * np.pi - H_p, H_p)

    IHLS = tstack([H, Y, S])

    return from_range_1(IHLS)


def IHLS_to_RGB(IHLS):
    """
    Converts from *RGB* colourspace to *IHLS* colourspace.

    Parameters
    ----------
    IHLS : (..., 3) array-like
        *HLS* colourspace array.

    Returns
    -------
     RGB : (..., 3) ndarray
        *RGB* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``HLS``    | [0, 1]                | [0, 1]        |
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
    """

    H, Y, S = tsplit(to_domain_1(IHLS))

    k = np.arange(6).reshape([1] * H.ndim + [6])
    H_s = H[..., np.newaxis] - k * (np.pi / 3)
    H_s = H_s[np.logical_and(H_s >= 0, H_s <= np.pi / 3)].reshape(H.shape)

    C = (np.sqrt(3) * S) / (2 * np.sin(2 / 3 * np.pi - H_s))
    C_1 = C * np.cos(H)
    C_2 = -C * np.sin(H)

    RGB = dot_vector(
        np.array([
            [1, 0.7875, 0.3714],
            [1, -0.2125, -0.2059],
            [1, -0.2125, 0.9488],
        ]), tstack([Y, C_1, C_2]))

    return from_range_1(RGB)


RGB = np.array([0.45620519, 0.03081071, 0.04091952])
print(RGB_to_IHLS(RGB))
print(RGB_to_IHLS2(RGB))
# np.testing.assert_almost_equal(IHLS_to_RGB(RGB_to_IHLS(RGB)), RGB)
np.testing.assert_almost_equal(IHLS_to_RGB(RGB_to_IHLS2(RGB)), RGB)
