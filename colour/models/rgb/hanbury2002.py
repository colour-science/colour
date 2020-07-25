# -*- coding: utf-8 -*-
"""
:math:`IHLS` Colour Encoding
===============================

Defines the :math:`IHLS` colour encoding related transformations:

-   :func:`colour.RGB_to_IHLS`
-   :func:`colour.IHLS_to_RGB`

References
----------
-   :cite:
    https://www.researchgate.net/publication/243602454_A_3D-Polar_Coordinate_Colour_Representation_Suitable_for_Image_Analysis
"""
from __future__ import division, unicode_literals
import numpy as np

from colour.utilities import (from_range_1, to_domain_1, tstack, tsplit)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'
__all__ = ['RGB_to_IHLS', 'IHLS_to_RGB']
"""
:math:`IHLS` Colour Encoding
===============================

Defines the :math:`IHLS` colour encoding related transformations:

-   :func:`colour.RGB_to_IHLS`
-   :func:`colour.IHLS_to_RGB`

References
----------
:cite:
    https://www.researchgate.net/publication/243602454_A_3D-Polar_Coordinate_Colour_Representation_Suitable_for_Image_Analysis
"""


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
    :cite: A_3D-Polar_Coordinate_Colour_Representation_
    Suitable_for_Image_Analysis
    https://www.researchgate.net/publication/243602454_
    A_3D-Polar_Coordinate_Colour_Representation_Suitable_for_Image_Analysis


    # *** [ No need for all that follows: ]
    RGB = np.asarray(RGB)

    # # check length of the last dimension, should be _some_ sort of rgb

    RGB = np.array(
        RGB, copy=False,
        dtype=np.promote_types(RGB.dtype, np.float32),  # Don't work on ints.
        ndmin=2, )

    R = RGB[..., 0]
    G = RGB[..., 1]
    B = RGB[..., 2]

    # *** [ Replace with: ]
    """
    R, G, B = tsplit(to_domain_1(RGB))

    L = 0.2126 * R + 0.7152 * G + 0.0722 * B  # Luminance

    C1 = 1 * R + (-(1 / 2) * G) + (-(1 / 2) * B)

    C2 = 0 * R + (-(np.sqrt(3) / 2) * G) + ((np.sqrt(3) / 2) * B)

    C = np.sqrt((C1 * C1) + (C2 * C2))

    W = (np.arccos(C1 / C))

    Q = (360 - np.arccos(C1 / C))

    H = (np.where(C2 > 0, Q, W))  # Condition for the HUE

    K = np.arange(6)  # array for K
    H1 = H - K * 60  # determining H1
    H2 = np.where(H1 < 0, 0.00,
                  H1)  # condition of H1 if its any values smaller than 0
    H3 = np.where(H2 < 60, H2, 0)
    H4 = np.max(H3)  # extraction of the correct range of H1

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
    :cite: A_3D-Polar_Coordinate_Colour_Representation_
    Suitable_for_Image_Analysis

    https://www.researchgate.net/publication/243602454_A_3D-Polar_Coordinate_Colour_Representation_Suitable_for_Image_Analysis


    """

    H, L, S = tsplit(to_domain_1(IHLS))

    K = np.arange(6)
    H1 = H - K * 60  #
    H2 = np.where(H1 < 0, 0.00, H1)
    H3 = np.where(H2 < 60, H2, 0.00)
    H4 = np.max(
        H3)  # #extraction of the correct range of H1 from array of zero

    C = (np.sqrt(3) * S) / (2 * (np.sin(120 - H4)))
    C1 = C * np.cos(H)
    C2 = (-(C * (np.sin(H))))
    R = 1.0000 * L + 0.7875 * C1 + 0.3714 * C2
    G = 1.0000 * L + (-(0.2125 * C1)) + (-(0.2059 * C2))
    B = 1.0000 * L + (-(0.2125 * C1)) + 0.9488 * C2

    RGB = tstack([R, G, B])

    return from_range_1(RGB)
