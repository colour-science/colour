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
    R, G, B = tsplit(to_domain_1(RGB))

    # TODO: Try matrix form here.
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    C1 = 1 * R + (-(1 / 2) * G) + (-(1 / 2) * B)
    C2 = 0 * R + (-(np.sqrt(3) / 2) * G) + ((np.sqrt(3) / 2) * B)

    C = np.sqrt((C1 * C1) + (C2 * C2))

    W = (np.arccos(C1 / C))
    Q = (360 - np.arccos(C1 / C))

    H = (np.where(C2 > 0, Q, W))

    K = np.arange(6)
    H1 = H - K * 60
    H2 = np.where(H1 < 0, 0, H1)
    H3 = np.where(H2 < 60, H2, 0)
    H4 = np.max(H3)

    S = ((2 * C * (np.sin(120 - H4))) / np.sqrt(3))

    IHLS = tstack([H, L, S])

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

    H, L, S = tsplit(to_domain_1(IHLS))

    K = np.arange(6)
    H1 = H - K * 60  #
    H2 = np.where(H1 < 0, 0.00, H1)
    H3 = np.where(H2 < 60, H2, 0.00)
    # #extraction of the correct range of H1 from array of zero
    H4 = np.max(H3)

    C = (np.sqrt(3) * S) / (2 * (np.sin(120 - H4)))
    C1 = C * np.cos(H)
    C2 = (-(C * (np.sin(H))))

    # R = 1.0000 * L + 0.7875 * C1 + 0.3714 * C2
    # G = 1.0000 * L -0.2125 * C1 -0.2059 * C2
    # B = 1.0000 * L -0.2125 * C1 + 0.9488 * C2

    M = np.array([
        [1, 0.7875, 0.3714],
        [1, -0.2125, -0.2059],
        [1, -0.2125, 0.9488],
    ])

    RGB = dot_vector(M, tstack([L, C1, C2]))
    # RGB = tstack([R, G, B])

    return from_range_1(RGB)
