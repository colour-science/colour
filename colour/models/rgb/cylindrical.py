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
They should not be used in the colour science domain, although they are useful
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
-   :cite:`Sarifuddin2005a` : Sarifuddin, M., & Missaoui, R. (2005). HCL: a new
    Color Space for a more Effective Content-based Image Retrieval.
    http://w3.uqo.ca/missaoui/Publications/TRColorSpace.zip
-   :cite:`Sarifuddin2021` : Sarifuddin, M. (2021). RGB to HCL and HCL to RGB
    color conversion (1.0.0). https://www.mathworks.com/matlabcentral/\
fileexchange/100878-rgb-to-hcl-and-hcl-to-rgb-color-conversion
-   :cite:`Wikipedia2015` : Wikipedia. (2015). HCL color space. Retrieved
    April 4, 2021, from https://en.wikipedia.org/wiki/HCL_color_space
"""

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.hints import ArrayLike, NDArrayFloat, cast
from colour.utilities import (
    as_float_array,
    from_range_1,
    to_domain_1,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "RGB_to_HSV",
    "HSV_to_RGB",
    "RGB_to_HSL",
    "HSL_to_RGB",
    "RGB_to_HCL",
    "HCL_to_RGB",
]


def RGB_to_HSV(RGB: ArrayLike) -> NDArrayFloat:
    """
    Convert from *RGB* colourspace to *HSV* colourspace.

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

    with sdiv_mode():
        S = sdiv(delta, maximum)

        delta_R = sdiv(((maximum - R) / 6) + (delta / 2), delta)
        delta_G = sdiv(((maximum - G) / 6) + (delta / 2), delta)
        delta_B = sdiv(((maximum - B) / 6) + (delta / 2), delta)

    H = delta_B - delta_G
    H = np.where(maximum == G, (1 / 3) + delta_R - delta_B, H)
    H = np.where(maximum == B, (2 / 3) + delta_G - delta_R, H)
    H[np.asarray(H < 0)] += 1
    H[np.asarray(H > 1)] -= 1
    H[np.asarray(delta == 0)] = 0

    HSV = tstack([H, S, V])

    return from_range_1(HSV)


def HSV_to_RGB(HSV: ArrayLike) -> NDArrayFloat:
    """
    Convert from *HSV* colourspace to *RGB* colourspace.

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
    l = V * (1 - S * (1 - (h - i)))  # noqa: E741

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
        mode="clip",
    )

    return from_range_1(RGB)


def RGB_to_HSL(RGB: ArrayLike) -> NDArrayFloat:
    """
    Convert from *RGB* colourspace to *HSL* colourspace.

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

    with sdiv_mode():
        S = np.where(
            L < 0.5,
            sdiv(delta, maximum + minimum),
            sdiv(delta, 2 - maximum - minimum),
        )

        delta_R = sdiv(((maximum - R) / 6) + (delta / 2), delta)
        delta_G = sdiv(((maximum - G) / 6) + (delta / 2), delta)
        delta_B = sdiv(((maximum - B) / 6) + (delta / 2), delta)

    H = delta_B - delta_G
    H = np.where(maximum == G, (1 / 3) + delta_R - delta_B, H)
    H = np.where(maximum == B, (2 / 3) + delta_G - delta_R, H)
    H[np.asarray(H < 0)] += 1
    H[np.asarray(H > 1)] -= 1
    H[np.asarray(delta == 0)] = 0

    HSL = tstack([H, S, L])

    return from_range_1(HSL)


def HSL_to_RGB(HSL: ArrayLike) -> NDArrayFloat:
    """
    Convert from *HSL* colourspace to *RGB* colourspace.

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

    def H_to_RGB(
        vi: NDArrayFloat, vj: NDArrayFloat, vH: NDArrayFloat
    ) -> NDArrayFloat:
        """Convert *hue* value to *RGB* colourspace."""

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


def RGB_to_HCL(
    RGB: ArrayLike, gamma: float = 3, Y_0: float = 100
) -> NDArrayFloat:
    """
    Convert from *RGB* colourspace to *HCL* colourspace according to
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

    -   This implementation used the equations given in
        :cite:`Sarifuddin2005a` and the corrections from
        :cite:`Sarifuddin2021`.

    References
    ----------
    :cite:`Sarifuddin2005`, :cite:`Sarifuddin2005a`, :cite:`Wikipedia2015`

    Examples
    --------
    >>> RGB = np.array([0.45620519, 0.03081071, 0.04091952])
    >>> RGB_to_HCL(RGB)  # doctest: +ELLIPSIS
    array([-0.0316785...,  0.2841715...,  0.2285964...])
    """

    R, G, B = tsplit(to_domain_1(RGB))

    Min = np.minimum(np.minimum(R, G), B)
    Max = np.maximum(np.maximum(R, G), B)

    with sdiv_mode():
        Q = np.exp(sdiv(Min * gamma, Max * Y_0))

    L = (Q * Max + (Q - 1) * Min) / 2

    R_G = R - G
    G_B = G - B
    B_R = B - R

    C = Q * (np.abs(R_G) + np.abs(G_B) + np.abs(B_R)) / 3

    with sdiv_mode("Ignore"):
        H = np.arctan(sdiv(G_B, R_G))

    _2_H_3 = 2 * H / 3
    _4_H_3 = 4 * H / 3

    H = np.select(
        [
            C == 0,
            np.logical_and(R_G >= 0, G_B >= 0),
            np.logical_and(R_G >= 0, G_B < 0),
            np.logical_and(R_G < 0, G_B >= 0),
            np.logical_and(R_G < 0, G_B < 0),
        ],
        [
            0,
            _2_H_3,
            _4_H_3,
            np.pi + _4_H_3,
            _2_H_3 - np.pi,
        ],
    )

    HCL = tstack([H, C, L])

    return from_range_1(HCL)


def HCL_to_RGB(
    HCL: ArrayLike, gamma: float = 3, Y_0: float = 100
) -> NDArrayFloat:
    """
    Convert from *HCL* colourspace to *RGB* colourspace according to
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

    -   This implementation used the equations given in
        :cite:`Sarifuddin2005a` and the corrections from
        :cite:`Sarifuddin2021`.

    References
    ----------
    :cite:`Sarifuddin2005`, :cite:`Sarifuddin2005a`, :cite:`Wikipedia2015`

    Examples
    --------
    >>> HCL = np.array([-0.03167854, 0.28417150, 0.22859647])
    >>> HCL_to_RGB(HCL)  # doctest: +ELLIPSIS
    array([ 0.4562033...,  0.0308104...,  0.0409192...])
    """

    H, C, L = tsplit(to_domain_1(HCL))

    with sdiv_mode():
        Q = np.exp((1 - sdiv(3 * C, 4 * L)) * gamma / Y_0)

        Min = sdiv(4 * L - 3 * C, 4 * Q - 2)
        Max = Min + sdiv(3 * C, 2 * Q)

    tan_3_2_H = np.tan(3 / 2 * H)
    tan_3_4_H_MP = np.tan(3 / 4 * (H - np.pi))
    tan_3_4_H = np.tan(3 / 4 * H)
    tan_3_2_H_PP = np.tan(3 / 2 * (H + np.pi))

    r_p60 = np.radians(60)
    r_p120 = np.radians(120)
    r_n60 = np.radians(-60)
    r_n120 = np.radians(-120)

    def _1_2_3(a: ArrayLike) -> NDArrayFloat:
        """Tail-stack given :math:`a` array as a *bool* dtype."""

        return tstack(cast(ArrayLike, [a, a, a]), dtype=np.bool_)

    with sdiv_mode():
        RGB = np.select(
            [
                _1_2_3(np.logical_and(0 <= H, r_p60 >= H)),
                _1_2_3(np.logical_and(r_p60 < H, r_p120 >= H)),
                _1_2_3(np.logical_and(r_p120 < H, np.pi >= H)),
                _1_2_3(np.logical_and(r_n60 <= H, H < 0)),
                _1_2_3(np.logical_and(r_n120 <= H, r_n60 > H)),
                _1_2_3(np.logical_and(-np.pi < H, r_n120 > H)),
            ],
            [
                tstack(
                    [
                        Max,
                        (Max * tan_3_2_H + Min) / (1 + tan_3_2_H),
                        Min,
                    ]
                ),
                tstack(
                    [
                        sdiv(Max * (1 + tan_3_4_H_MP) - Min, tan_3_4_H_MP),
                        Max,
                        Min,
                    ]
                ),
                tstack(
                    [
                        Min,
                        Max,
                        Max * (1 + tan_3_4_H_MP) - Min * tan_3_4_H_MP,
                    ]
                ),
                tstack(
                    [
                        Max,
                        Min,
                        Min * (1 + tan_3_4_H) - Max * tan_3_4_H,
                    ]
                ),
                tstack(
                    [
                        sdiv(Min * (1 + tan_3_4_H) - Max, tan_3_4_H),
                        Min,
                        Max,
                    ]
                ),
                tstack(
                    [
                        Min,
                        (Min * tan_3_2_H_PP + Max) / (1 + tan_3_2_H_PP),
                        Max,
                    ]
                ),
            ],
        )

    return from_range_1(RGB)
