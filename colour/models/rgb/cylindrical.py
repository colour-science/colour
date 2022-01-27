# -*- coding: utf-8 -*-
"""
Cylindrical & Spherical Colour Models
=====================================

Defines various cylindrical and spherical colour models:

-   :func:`colour.RGB_to_HSV`
-   :func:`colour.HSV_to_RGB`
-   :func:`colour.RGB_to_HSL`
-   :func:`colour.HSL_to_RGB`
-   :func:`colour.RGB_to_HCL`
-   :func:`colour.HCL_to_RGB`

These colour models trade off perceptual relevance for computation speed.
They should not be used in the colour science domain although they are useful
for image analysis and provide end user software colour selection tools.

They are provided for convenience and completeness.

References
----------
-   :cite:`EasyRGBj` : EasyRGB. (n.d.). RGB --> HSV. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=20#text20
-   :cite:`EasyRGBk` : EasyRGB. (n.d.). HSL --> RGB. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=19#text19
-   :cite:`EasyRGBl` : EasyRGB. (n.d.). RGB --> HSL. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=18#text18
-   :cite:`EasyRGBn` : EasyRGB. (n.d.). HSV --> RGB. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=21#text21
-   :cite:`Smith1978b` : Smith, A. R. (1978). Color gamut transform pairs.
    Proceedings of the 5th Annual Conference on Computer Graphics and
    Interactive Techniques - SIGGRAPH "78, 12-19. doi:10.1145/800248.807361
-   :cite:`Wikipedia2003` : Wikipedia. (2003). HSL and HSV. Retrieved
    September 10, 2014, from http://en.wikipedia.org/wiki/HSL_and_HSV
-   :cite:`Sarifuddin2005` : Sarifuddin, M., & Missaoui, R. (2005). A New
    Perceptually Uniform Color Space with Associated Color Similarity Measure
    for ContentBased Image and Video Retrieval.
-   :cite:`Wikipedia2015` : Wikipedia. (2015). HCL color space. Retrieved
    April 4, 2021, from https://en.wikipedia.org/wiki/HCL_color_space
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, Floating, NDArray
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
    'RGB_to_HSV',
    'HSV_to_RGB',
    'RGB_to_HSL',
    'HSL_to_RGB',
    'RGB_to_HCL',
    'HCL_to_RGB',
]


def RGB_to_HSV(RGB: ArrayLike) -> NDArray:
    """
    Converts from *RGB* colourspace to *HSV* colourspace.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *HSV* array.

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
    | ``HSV``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`EasyRGBj`, :cite:`Smith1978b`, :cite:`Wikipedia2003`

    Examples
    --------
    >>> RGB = np.array([0.45620519, 0.03081071, 0.04091952])
    >>> RGB_to_HSV(RGB)  # doctest: +ELLIPSIS
    array([ 0.9960394...,  0.9324630...,  0.4562051...])
    """

    RGB = to_domain_1(RGB)

    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    V = maximum

    R, G, B = tsplit(RGB)

    S = as_float_array(delta / maximum)
    S[np.asarray(delta == 0)] = 0

    delta_R = (((maximum - R) / 6) + (delta / 2)) / delta
    delta_G = (((maximum - G) / 6) + (delta / 2)) / delta
    delta_B = (((maximum - B) / 6) + (delta / 2)) / delta

    H = delta_B - delta_G
    H = np.where(G == maximum, (1 / 3) + delta_R - delta_B, H)
    H = np.where(B == maximum, (2 / 3) + delta_G - delta_R, H)
    H[np.asarray(H < 0)] += 1
    H[np.asarray(H > 1)] -= 1
    H[np.asarray(delta == 0)] = 0

    HSV = tstack([H, S, V])

    return from_range_1(HSV)


def HSV_to_RGB(HSV: ArrayLike) -> NDArray:
    """
    Converts from *HSV* colourspace to *RGB* colourspace.

    Parameters
    ----------
    HSV
        *HSV* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``HSV``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`EasyRGBn`, :cite:`Smith1978b`, :cite:`Wikipedia2003`

    Examples
    --------
    >>> HSV = np.array([0.99603944, 0.93246304, 0.45620519])
    >>> HSV_to_RGB(HSV)  # doctest: +ELLIPSIS
    array([ 0.4562051...,  0.0308107...,  0.0409195...])
    """

    H, S, V = tsplit(to_domain_1(HSV))

    h = as_float_array(H * 6)
    h[np.asarray(h == 6)] = 0

    i = np.floor(h)
    j = V * (1 - S)
    k = V * (1 - S * (h - i))
    l = V * (1 - S * (1 - (h - i)))  # noqa

    i = tstack([i, i, i]).astype(np.uint8)

    RGB = np.choose(
        i,
        [
            tstack([V, l, j]),
            tstack([k, V, j]),
            tstack([j, V, l]),
            tstack([j, k, V]),
            tstack([l, j, V]),
            tstack([V, j, k]),
        ],
        mode='clip',
    )

    return from_range_1(RGB)


def RGB_to_HSL(RGB: ArrayLike) -> NDArray:
    """
    Converts from *RGB* colourspace to *HSL* colourspace.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *HSL* array.

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
    | ``HSL``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`EasyRGBl`, :cite:`Smith1978b`, :cite:`Wikipedia2003`

    Examples
    --------
    >>> RGB = np.array([0.45620519, 0.03081071, 0.04091952])
    >>> RGB_to_HSL(RGB)  # doctest: +ELLIPSIS
    array([ 0.9960394...,  0.8734714...,  0.2435079...])
    """

    RGB = to_domain_1(RGB)

    minimum = np.amin(RGB, -1)
    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    R, G, B = tsplit(RGB)

    L = (maximum + minimum) / 2

    S = np.where(
        L < 0.5,
        delta / (maximum + minimum),
        delta / (2 - maximum - minimum),
    )
    S[np.asarray(delta == 0)] = 0

    delta_R = (((maximum - R) / 6) + (delta / 2)) / delta
    delta_G = (((maximum - G) / 6) + (delta / 2)) / delta
    delta_B = (((maximum - B) / 6) + (delta / 2)) / delta

    H = delta_B - delta_G
    H = np.where(G == maximum, (1 / 3) + delta_R - delta_B, H)
    H = np.where(B == maximum, (2 / 3) + delta_G - delta_R, H)
    H[np.asarray(H < 0)] += 1
    H[np.asarray(H > 1)] -= 1
    H[np.asarray(delta == 0)] = 0

    HSL = tstack([H, S, L])

    return from_range_1(HSL)


def HSL_to_RGB(HSL: ArrayLike) -> NDArray:
    """
    Converts from *HSL* colourspace to *RGB* colourspace.

    Parameters
    ----------
    HSL
        *HSL* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``HSL``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`EasyRGBk`, :cite:`Smith1978b`, :cite:`Wikipedia2003`

    Examples
    --------
    >>> HSL = np.array([0.99603944, 0.87347144, 0.24350795])
    >>> HSL_to_RGB(HSL)  # doctest: +ELLIPSIS
    array([ 0.4562051...,  0.0308107...,  0.0409195...])
    """

    H, S, L = tsplit(to_domain_1(HSL))

    def H_to_RGB(vi, vj, vH):
        """
        Converts *hue* value to *RGB* colourspace.
        """

        vH = as_float_array(vH)

        vH[np.asarray(vH < 0)] += 1
        vH[np.asarray(vH > 1)] -= 1

        v = np.where(
            6 * vH < 1,
            vi + (vj - vi) * 6 * vH,
            np.nan,
        )
        v = np.where(np.logical_and(2 * vH < 1, np.isnan(v)), vj, v)
        v = np.where(
            np.logical_and(3 * vH < 2, np.isnan(v)),
            vi + (vj - vi) * ((2 / 3) - vH) * 6,
            v,
        )
        v = np.where(np.isnan(v), vi, v)

        return v

    j = np.where(L < 0.5, L * (1 + S), (L + S) - (S * L))
    i = 2 * L - j

    R = H_to_RGB(i, j, H + (1 / 3))
    G = H_to_RGB(i, j, H)
    B = H_to_RGB(i, j, H - (1 / 3))

    R = np.where(S == 0, L, R)
    G = np.where(S == 0, L, G)
    B = np.where(S == 0, L, B)

    RGB = tstack([R, G, B])

    return from_range_1(RGB)


def RGB_to_HCL(RGB: ArrayLike, gamma: Floating = 3,
               Y_0: Floating = 100) -> NDArray:
    """
    Converts from *RGB* colourspace to *HCL* colourspace according to
    *Sarifuddin and Missaoui (2005)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    gamma
        Non-linear lightness exponent matching *Lightness* :math:`L^*`.
    Y_0
        White reference luminance :math:`Y_0`.

    Returns
    -------
    :class:`numpy.ndarray`
        *HCL* array.

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
    | ``HCL``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Sarifuddin2005`, :cite:`Wikipedia2015`

    Examples
    --------
    >>> RGB = np.array([0.45620519, 0.03081071, 0.04091952])
    >>> RGB_to_HCL(RGB)  # doctest: +ELLIPSIS
    array([-0.0316785...,  0.2841715...,  0.2285964...])
    """

    R, G, B = tsplit(to_domain_1(RGB))

    Min = np.minimum(np.minimum(R, G), B)
    Max = np.maximum(np.maximum(R, G), B)

    alpha = (Min / Max) / Y_0

    Q = np.exp(alpha * gamma)

    L = (Q * Max + (Q - 1) * Min) / 2

    R_G = R - G
    G_B = G - B
    B_R = B - R

    C = Q * (np.abs(R_G) + np.abs(G_B) + np.abs(B_R)) / 3

    H = np.arctan(G_B / R_G)

    _2_3_H = 2 / 3 * H
    _4_3_H = 4 / 3 * H

    H = np.select(
        [
            np.logical_and(R_G >= 0, G_B >= 0),
            np.logical_and(R_G >= 0, G_B < 0),
            np.logical_and(R_G < 0, G_B >= 0),
            np.logical_and(R_G < 0, G_B < 0),
        ],
        [
            _2_3_H,
            _4_3_H,
            np.pi + _4_3_H,
            _2_3_H - np.pi,
        ],
    )

    HCL = tstack([H, C, L])

    return from_range_1(HCL)


def HCL_to_RGB(HCL: ArrayLike, gamma: Floating = 3,
               Y_0: Floating = 100) -> NDArray:
    """
    Converts from *HCL* colourspace to *RGB* colourspace according to
    *Sarifuddin and Missaoui (2005)* method.

    Parameters
    ----------
    HCL
        *HCL* colourspace array.
    gamma
        Non-linear lightness exponent matching *Lightness* :math:`L^*`.
    Y_0
        White reference luminance :math:`Y_0`.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``HCL``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Sarifuddin2005`, :cite:`Wikipedia2015`

    Examples
    --------
    >>> HCL = np.array([-0.03167854, 0.28417150, 0.22859647])
    >>> HCL_to_RGB(HCL)  # doctest: +ELLIPSIS
    array([ 0.4562033...,  0.0308104...,  0.0409192...])
    """

    H, C, L = tsplit(to_domain_1(HCL))

    Q = np.exp((1 - (3 * C) / (4 * L)) * (gamma / Y_0))

    Min = (4 * L - 3 * C) / (4 * Q - 2)
    Max = Min + (3 * C) / (2 * Q)

    def _1_2_3(a):
        """
        Tail-stack given :math:`a` array as a *bool* dtype.
        """

        return tstack([a, a, a], dtype=np.bool_)

    tan_3_2_H = np.tan(3 / 2 * H)
    tan_3_4_H_MP = np.tan(3 / 4 * (H - np.pi))
    tan_3_4_H = np.tan(3 / 4 * H)
    tan_3_2_H_PP = np.tan(3 / 2 * (H + np.pi))

    RGB = np.select(
        [
            _1_2_3(np.logical_and(0 <= H, H <= np.radians(60))),
            _1_2_3(np.logical_and(np.radians(60) < H, H <= np.radians(120))),
            _1_2_3(np.logical_and(np.radians(120) < H, H <= np.pi)),
            _1_2_3(np.logical_and(np.radians(-60) <= H, H < 0)),
            _1_2_3(np.logical_and(np.radians(-120) <= H, H < np.radians(-60))),
            _1_2_3(np.logical_and(-np.pi < H, H < np.radians(-120))),
        ],
        [
            tstack([
                Max,
                (Max * tan_3_2_H + Min) / (1 + tan_3_2_H),
                Min,
            ]),
            tstack([
                (Max * (1 + tan_3_4_H_MP) - Min) / tan_3_4_H_MP,
                Max,
                Min,
            ]),
            tstack([
                Min,
                Max,
                Max * (1 + tan_3_4_H_MP) - Min * tan_3_4_H_MP,
            ]),
            tstack([
                Max,
                Min,
                Min * (1 + tan_3_4_H) - Max * tan_3_4_H,
            ]),
            tstack([
                (Min * (1 + tan_3_4_H) - Max) / tan_3_4_H,
                Min,
                Max,
            ]),
            tstack([
                Min,
                (Min * tan_3_2_H_PP + Max) / (1 + tan_3_2_H_PP),
                Max,
            ]),
        ],
    )

    return from_range_1(RGB)
