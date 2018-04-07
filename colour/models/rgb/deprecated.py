# -*- coding: utf-8 -*-
"""
Deprecated Colour Models Transformations
========================================

Defines various deprecated colour models transformations:

-   :func:`colour.RGB_to_HSV`
-   :func:`colour.HSV_to_RGB`
-   :func:`colour.RGB_to_HSL`
-   :func:`colour.HSL_to_RGB`
-   :func:`colour.RGB_to_CMY`
-   :func:`colour.CMY_to_RGB`
-   :func:`colour.CMY_to_CMYK`
-   :func:`colour.CMYK_to_CMY`

These colour models are stated as deprecated because they trade off perceptual
relevance for computation speed. They should not be used in the colour science
domain although they are useful for image analysis and provide end user
software colour selection tools.

They are provided for convenience and completeness.

Warning
-------
Don't use that! Seriously...

References
----------
-   :cite:`EasyRGBh` : EasyRGB. (n.d.). RGB > CMY. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=11#text11
-   :cite:`EasyRGBi` : EasyRGB. (n.d.). CMY > RGB. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=12#text12
-   :cite:`EasyRGBj` : EasyRGB. (n.d.). RGB > HSV. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=20#text20
-   :cite:`EasyRGBk` : EasyRGB. (n.d.). HSL > RGB. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=19#text19
-   :cite:`EasyRGBl` : EasyRGB. (n.d.). RGB > HSL. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=18#text18
-   :cite:`EasyRGBm` : EasyRGB. (n.d.). CMYK > CMY. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=14#text14
-   :cite:`EasyRGBn` : EasyRGB. (n.d.). HSV > RGB. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=21#text21
-   :cite:`EasyRGBo` : EasyRGB. (n.d.). CMY > CMYK. Retrieved May 18, 2014,
    from http://www.easyrgb.com/index.php?X=MATH&H=13#text13
-   :cite:`Smith1978b` : Smith, A. R. (1978). Color gamut transform pairs. In
    Proceedings of the 5th annual conference on Computer graphics and
    interactive techniques - SIGGRAPH '78 (pp. 12-19). New York, New York,
    USA: ACM Press. doi:10.1145/800248.807361
-   :cite:`Wikipediacg` : Wikipedia. (n.d.). HSL and HSV. Retrieved September
    10, 2014, from http://en.wikipedia.org/wiki/HSL_and_HSV
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import from_range_1, to_domain_1, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RGB_to_HSV', 'HSV_to_RGB', 'RGB_to_HSL', 'HSL_to_RGB', 'RGB_to_CMY',
    'CMY_to_RGB', 'CMY_to_CMYK', 'CMYK_to_CMY'
]


def RGB_to_HSV(RGB):
    """
    Converts from *RGB* colourspace to *HSV* colourspace.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.

    Returns
    -------
    ndarray
        *HSV* array.

    Notes
    -----
    -   Input *RGB* colourspace array is normalised to domain [0, 1].
    -   Output *HSV* colourspace array is normalised to range [0, 1].

    References
    ----------
    -   :cite:`EasyRGBj`
    -   :cite:`Smith1978b`
    -   :cite:`Wikipediacg`

    Examples
    --------
    >>> RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    >>> RGB_to_HSV(RGB)  # doctest: +ELLIPSIS
    array([ 0.2786738...,  0.744     ,  0.98039216])
    """

    RGB = to_domain_1(RGB)

    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    V = maximum

    R, G, B = tsplit(RGB)

    S = np.asarray(delta / maximum)
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

    HSV = tstack((H, S, V))

    return from_range_1(HSV)


def HSV_to_RGB(HSV):
    """
    Converts from *HSV* colourspace to *RGB* colourspace.

    Parameters
    ----------
    HSV : array_like
        *HSV* colourspace array.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   Input *HSV* colourspace array is normalised to domain [0, 1].
    -   Output *RGB* colourspace array is normalised to range [0, 1].

    References
    ----------
    -   :cite:`EasyRGBn`
    -   :cite:`Smith1978b`
    -   :cite:`Wikipediacg`

    Examples
    --------
    >>> HSV = np.array([0.27867384, 0.74400000, 0.98039216])
    >>> HSV_to_RGB(HSV)  # doctest: +ELLIPSIS
    array([ 0.4901960...,  0.9803921...,  0.2509803...])
    """

    H, S, V = tsplit(to_domain_1(HSV))

    h = np.asarray(H * 6)
    h[np.asarray(h == 6)] = 0

    i = np.floor(h)
    j = V * (1 - S)
    k = V * (1 - S * (h - i))
    l = V * (1 - S * (1 - (h - i)))  # noqa

    i = tstack((i, i, i)).astype(np.uint8)

    RGB = np.choose(
        i, [
            tstack((V, l, j)),
            tstack((k, V, j)),
            tstack((j, V, l)),
            tstack((j, k, V)),
            tstack((l, j, V)),
            tstack((V, j, k)),
        ],
        mode='clip')

    return from_range_1(RGB)


def RGB_to_HSL(RGB):
    """
    Converts from *RGB* colourspace to *HSL* colourspace.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.

    Returns
    -------
    ndarray
        *HSL* array.

    Notes
    -----
    -   Input *RGB* colourspace array is normalised to domain [0, 1].
    -   Output *HSL* colourspace array is normalised to range [0, 1].

    References
    ----------
    -   :cite:`EasyRGBl`
    -   :cite:`Smith1978b`
    -   :cite:`Wikipediacg`

    Examples
    --------
    >>> RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    >>> RGB_to_HSL(RGB)  # doctest: +ELLIPSIS
    array([ 0.2786738...,  0.9489796...,  0.6156862...])
    """

    RGB = to_domain_1(RGB)

    minimum = np.amin(RGB, -1)
    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    R, G, B = tsplit(RGB)

    L = (maximum + minimum) / 2

    S = np.where(L < 0.5, delta / (maximum + minimum),
                 delta / (2 - maximum - minimum))
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

    HSL = tstack((H, S, L))

    return from_range_1(HSL)


def HSL_to_RGB(HSL):
    """
    Converts from *HSL* colourspace to *RGB* colourspace.

    Parameters
    ----------
    HSL : array_like
        *HSL* colourspace array.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   Input *HSL* colourspace array is normalised to domain [0, 1].
    -   Output *RGB* colourspace array is normalised to range [0, 1].

    References
    ----------
    -   :cite:`EasyRGBk`
    -   :cite:`Smith1978b`
    -   :cite:`Wikipediacg`

    Examples
    --------
    >>> HSL = np.array([0.27867384, 0.94897959, 0.61568627])
    >>> HSL_to_RGB(HSL)  # doctest: +ELLIPSIS
    array([ 0.4901960...,  0.9803921...,  0.2509803...])
    """

    H, S, L = tsplit(to_domain_1(HSL))

    def H_to_RGB(vi, vj, vH):
        """
        Converts *hue* value to *RGB* colourspace.
        """

        vH = np.asarray(vH)

        vH[np.asarray(vH < 0)] += 1
        vH[np.asarray(vH > 1)] -= 1

        v = np.full(vi.shape, np.nan)

        v = np.where(
            np.logical_and(6 * vH < 1, np.isnan(v)),
            vi + (vj - vi) * 6 * vH, v)  # yapf: disable
        v = np.where(np.logical_and(2 * vH < 1, np.isnan(v)), vj, v)
        v = np.where(
            np.logical_and(3 * vH < 2, np.isnan(v)),
            vi + (vj - vi) * ((2 / 3) - vH) * 6, v)
        v = np.where(np.isnan(v), vi, v)

        return v

    j = np.where(L < 0.5, L * (1 + S), (L + S) - (S * L))
    i = 2 * L - j

    R = H_to_RGB(i, j, H + (1 / 3))
    G = H_to_RGB(i, j, H)
    B = H_to_RGB(i, j, H - (1 / 3))

    R = np.where(S == 1, L, R)
    G = np.where(S == 1, L, G)
    B = np.where(S == 1, L, B)

    RGB = tstack((R, G, B))

    return from_range_1(RGB)


def RGB_to_CMY(RGB):
    """
    Converts from *RGB* colourspace to *CMY* colourspace.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.

    Returns
    -------
    ndarray
        *CMY* array.

    Notes
    -----
    -   Input *RGB* colourspace array is normalised to domain [0, 1].
    -   Output *CMY* colourspace array is normalised to range [0, 1].

    References
    ----------
    -   :cite:`EasyRGBh`

    Examples
    --------
    >>> RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    >>> RGB_to_CMY(RGB)  # doctest: +ELLIPSIS
    array([ 0.5098039...,  0.0196078...,  0.7490196...])
    """

    CMY = 1 - to_domain_1(RGB)

    return from_range_1(CMY)


def CMY_to_RGB(CMY):
    """
    Converts from *CMY* colourspace to *CMY* colourspace.

    Parameters
    ----------
    CMY : array_like
        *CMY* colourspace array.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   Input *CMY* colourspace array is normalised to domain [0, 1].
    -   Output *RGB* colourspace array is normalised to range [0, 1].

    References
    ----------
    -   :cite:`EasyRGBi`

    Examples
    --------
    >>> CMY = np.array([0.50980392, 0.01960784, 0.74901961])
    >>> CMY_to_RGB(CMY)  # doctest: +ELLIPSIS
    array([ 0.4901960...,  0.9803921...,  0.2509803...])
    """

    RGB = 1 - to_domain_1(CMY)

    return from_range_1(RGB)


def CMY_to_CMYK(CMY):
    """
    Converts from *CMY* colourspace to *CMYK* colourspace.

    Parameters
    ----------
    CMY : array_like
        *CMY* colourspace array.

    Returns
    -------
    ndarray
        *CMYK* array.

    Notes
    -----
    -   Input *CMY* colourspace array is normalised to domain [0, 1].
    -   Output*CMYK* colourspace array is normalised to range [0, 1].

    References
    ----------
    -   :cite:`EasyRGBo`

    Examples
    --------
    >>> CMY = np.array([0.50980392, 0.01960784, 0.74901961])
    >>> CMY_to_CMYK(CMY)  # doctest: +ELLIPSIS
    array([ 0.5       ,  0.        ,  0.744     ,  0.0196078...])
    """

    C, M, Y = tsplit(to_domain_1(CMY))

    K = np.ones(C.shape)
    K = np.where(C < K, C, K)
    K = np.where(M < K, M, K)
    K = np.where(Y < K, Y, K)

    C = np.asarray((C - K) / (1 - K))
    M = np.asarray((M - K) / (1 - K))
    Y = np.asarray((Y - K) / (1 - K))

    C[np.asarray(K == 1)] = 0
    M[np.asarray(K == 1)] = 0
    Y[np.asarray(K == 1)] = 0

    CMYK = tstack((C, M, Y, K))

    return from_range_1(CMYK)


def CMYK_to_CMY(CMYK):
    """
    Converts from *CMYK* colourspace to *CMY* colourspace.

    Parameters
    ----------
    CMYK : array_like
        *CMYK* colourspace array.

    Returns
    -------
    ndarray
        *CMY* array.

    Notes
    -----
    -   Input *CMYK* colourspace array is normalised to domain [0, 1].
    -   Output *CMY* colourspace array is normalised to range [0, 1].

    References
    ----------
    -   :cite:`EasyRGBm`

    Examples
    --------
    >>> CMYK = np.array([0.50000000, 0.00000000, 0.74400000, 0.01960784])
    >>> CMYK_to_CMY(CMYK)  # doctest: +ELLIPSIS
    array([ 0.5098039...,  0.0196078...,  0.7490196...])
    """

    C, M, Y, K = tsplit(to_domain_1(CMYK))

    CMY = tstack((C * (1 - K) + K, M * (1 - K) + K, Y * (1 - K) + K))

    return from_range_1(CMY)
