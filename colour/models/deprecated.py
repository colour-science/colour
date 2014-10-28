#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deprecated Colour Models Transformations
========================================

Defines various deprecated colour models transformations:

-   :func:`RGB_to_HSV`
-   :func:`HSV_to_RGB`
-   :func:`RGB_to_HSL`
-   :func:`HSL_to_RGB`
-   :func:`RGB_to_CMY`
-   :func:`CMY_to_RGB`
-   :func:`CMY_to_CMYK`
-   :func:`CMYK_to_CMY`

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
.. [1]  Wikipedia. (n.d.). HSL and HSV. Retrieved September 10, 2014, from
        http://en.wikipedia.org/wiki/HSL_and_HSV
.. [2]  Smith, A. R. (1978). Color Gamut Transform Pairs. In Proceedings of
        the 5th Annual Conference on Computer Graphics and Interactive
        Techniques (pp. 12–19). New York, NY, USA: ACM.
        doi:10.1145/800248.807361
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_to_HSV',
           'HSV_to_RGB',
           'RGB_to_HSL',
           'HSL_to_RGB',
           'RGB_to_CMY',
           'CMY_to_RGB',
           'CMY_to_CMYK',
           'CMYK_to_CMY']


def RGB_to_HSV(RGB):
    """
    Converts from *RGB* colourspace to *HSV* colourspace.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *HSV* matrix.

    Notes
    -----
    -   Input *RGB* colourspace matrix is in domain [0, 1].
    -   Output *HSV* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [3]  EasyRGB. (n.d.). RGB —> HSV. Retrieved May 18, 2014, from
            http://www.easyrgb.com/index.php?X=MATH&H=20#text20

    Examples
    --------
    >>> RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    >>> RGB_to_HSV(RGB)  # doctest: +ELLIPSIS
    array([ 0.2786738...,  0.744     ,  0.98039216])
    """

    R, G, B = np.ravel(RGB)

    minimum = min(R, G, B)
    maximum = max(R, G, B)
    delta = maximum - minimum

    V = maximum

    if delta == 0:
        H = 0
        S = 0
    else:

        S = delta / maximum

        delta_R = (((maximum - R) / 6) + (delta / 2)) / delta
        delta_G = (((maximum - G) / 6) + (delta / 2)) / delta
        delta_B = (((maximum - B) / 6) + (delta / 2)) / delta

        if R == maximum:
            H = delta_B - delta_G
        elif G == maximum:
            H = (1 / 3) + delta_R - delta_B
        elif B == maximum:
            H = (2 / 3) + delta_G - delta_R

        if H < 0:
            H += 1
        if H > 1:
            H -= 1

    return np.array([H, S, V])


def HSV_to_RGB(HSV):
    """
    Converts from *HSV* colourspace to *RGB* colourspace.

    Parameters
    ----------
    HSV : array_like, (3,)
        *HSV* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *RGB* colourspace matrix.

    Notes
    -----
    -   Input *HSV* colourspace matrix is in domain [0, 1].
    -   Output *RGB* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [4]  EasyRGB. (n.d.). HSV —> RGB. Retrieved May 18, 2014, from
            http://www.easyrgb.com/index.php?X=MATH&H=21#text21

    Examples
    --------
    >>> HSV = np.array([0.27867384, 0.744, 0.98039216])
    >>> HSV_to_RGB(HSV)  # doctest: +ELLIPSIS
    array([ 0.4901960...,  0.9803921...,  0.2509803...])
    """

    H, S, V = np.ravel(HSV)

    if S == 0:
        R = V
        G = V
        B = V
    else:
        h = H * 6
        if h == 6:
            h = 0

        i = np.floor(h)
        j = V * (1 - S)
        k = V * (1 - S * (h - i))
        l = V * (1 - S * (1 - (h - i)))
        if i == 0:
            R = V
            G = l
            B = j
        elif i == 1:
            R = k
            G = V
            B = j
        elif i == 2:
            R = j
            G = V
            B = l
        elif i == 3:
            R = j
            G = k
            B = V
        elif i == 4:
            R = l
            G = j
            B = V
        elif i == 5:
            R = V
            G = j
            B = k

    return np.array([R, G, B])


def RGB_to_HSL(RGB):
    """
    Converts from *RGB* colourspace to *HSL* colourspace.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *HSL* matrix.

    Notes
    -----
    -   Input *RGB* colourspace matrix is in domain [0, 1].
    -   Output *HSL* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [5]  EasyRGB. (n.d.). RGB —> HSL. Retrieved May 18, 2014, from
            http://www.easyrgb.com/index.php?X=MATH&H=18#text18

    Examples
    --------
    >>> RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    >>> RGB_to_HSL(RGB)  # doctest: +ELLIPSIS
    array([ 0.2786738...,  0.9489796...,  0.6156862...])
    """

    R, G, B = np.ravel(RGB)

    minimum = min(R, G, B)
    maximum = max(R, G, B)
    delta = maximum - minimum

    L = (maximum + minimum) / 2

    if delta == 0:
        H = 0
        S = 0
    else:

        S = delta / (maximum + minimum) if L < 0.5 else delta / (
            2 - maximum - minimum)

        delta_R = (((maximum - R) / 6) + (delta / 2)) / delta
        delta_G = (((maximum - G) / 6) + (delta / 2)) / delta
        delta_B = (((maximum - B) / 6) + (delta / 2)) / delta

        if R == maximum:
            H = delta_B - delta_G
        elif G == maximum:
            H = (1 / 3) + delta_R - delta_B
        elif B == maximum:
            H = (2 / 3) + delta_G - delta_R

        if H < 0:
            H += 1
        if H > 1:
            H -= 1

    return np.array([H, S, L])


def HSL_to_RGB(HSL):
    """
    Converts from *HSL* colourspace to *RGB* colourspace.

    Parameters
    ----------
    HSL : array_like, (3,)
        *HSL* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *RGB* colourspace matrix.

    Notes
    -----
    -   Input *HSL* colourspace matrix is in domain [0, 1].
    -   Output *RGB* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [6]  EasyRGB. (n.d.). HSL —> RGB. Retrieved May 18, 2014, from
            http://www.easyrgb.com/index.php?X=MATH&H=19#text19

    Examples
    --------
    >>> HSL = np.array([0.27867384, 0.94897959, 0.61568627])
    >>> HSL_to_RGB(HSL)  # doctest: +ELLIPSIS
    array([ 0.4901960...,  0.9803921...,  0.2509803...])
    """

    H, S, L = np.ravel(HSL)

    if S == 1:
        R = L
        G = L
        B = L
    else:
        def H_to_RGB(vi, vj, vH):
            """
            Converts *hue* value to *RGB* colourspace.
            """

            if vH < 0:
                vH += 1
            if vH > 1:
                vH -= 1
            if 6 * vH < 1:
                return vi + (vj - vi) * 6 * vH
            if 2 * vH < 1:
                return vj
            if 3 * vH < 2:
                return vi + (vj - vi) * ((2 / 3) - vH) * 6
            return vi

        j = L * (1 + S) if L < 0.5 else (L + S) - (S * L)
        i = 2 * L - j

        R = H_to_RGB(i, j, H + (1 / 3))
        G = H_to_RGB(i, j, H)
        B = H_to_RGB(i, j, H - (1 / 3))

    return np.array([R, G, B])


def RGB_to_CMY(RGB):
    """
    Converts from *RGB* colourspace to *CMY* colourspace.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CMY* matrix.

    Notes
    -----
    -   Input *RGB* colourspace matrix is in domain [0, 1].
    -   Output *CMY* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [7]  EasyRGB. (n.d.). RGB —> CMY. Retrieved May 18, 2014, from
            http://www.easyrgb.com/index.php?X=MATH&H=11#text11

    Examples
    --------
    >>> RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    >>> RGB_to_CMY(RGB)  # doctest: +ELLIPSIS
    array([ 0.5098039...,  0.0196078...,  0.7490196...])
    """

    R, G, B = np.ravel(RGB)
    return np.array([1 - R, 1 - G, 1 - B])


def CMY_to_RGB(CMY):
    """
    Converts from *CMY* colourspace to *CMY* colourspace.

    Parameters
    ----------
    CMY : array_like, (3,)
        *CMY* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *RGB* colourspace matrix.

    Notes
    -----
    -   Input *CMY* colourspace matrix is in domain [0, 1].
    -   Output *RGB* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [8]  EasyRGB. (n.d.). CMY —> RGB. Retrieved May 18, 2014, from
            http://www.easyrgb.com/index.php?X=MATH&H=12#text12

    Examples
    --------
    >>> CMY = np.array([0.50980392, 0.01960784, 0.74901961])
    >>> CMY_to_RGB(CMY)  # doctest: +ELLIPSIS
    array([ 0.4901960...,  0.9803921...,  0.2509803...])
    """

    C, M, Y = np.ravel(CMY)
    return np.array([1 - C, 1 - M, 1 - Y])


def CMY_to_CMYK(CMY):
    """
    Converts from *CMY* colourspace to *CMYK* colourspace.

    Parameters
    ----------
    CMY : array_like, (3,)
        *CMY* colourspace matrix.

    Returns
    -------
    ndarray, (4,)
        *CMYK* matrix.

    Notes
    -----
    -   Input *CMY* colourspace matrix is in domain [0, 1].
    -   Output*CMYK* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [9]  EasyRGB. (n.d.). CMY —> CMYK. Retrieved May 18, 2014, from
            http://www.easyrgb.com/index.php?X=MATH&H=13#text13

    Examples
    --------
    >>> CMY = np.array([0.50980392, 0.01960784, 0.74901961])
    >>> CMY_to_CMYK(CMY)  # doctest: +ELLIPSIS
    array([ 0.5       ,  0.        ,  0.744     ,  0.0196078...])
    """

    C, M, Y = np.ravel(CMY)

    K = 1

    if C < K:
        K = C
    if M < K:
        K = M
    if Y < K:
        K = Y
    if K == 1:
        C = 0
        M = 0
        Y = 0
    else:
        C = (C - K) / (1 - K)
        M = (M - K) / (1 - K)
        Y = (Y - K) / (1 - K)

    return np.array([C, M, Y, K])


def CMYK_to_CMY(CMYK):
    """
    Converts from *CMYK* colourspace to *CMY* colourspace.

    Parameters
    ----------
    CMYK : array_like, (4,)
        *CMYK* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CMY* matrix.

    Notes
    -----
    -   Input *CMYK* colourspace matrix is in domain [0, 1].
    -   Output *CMY* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [10] EasyRGB. (n.d.). CMYK —> CMY. Retrieved May 18, 2014, from
            http://www.easyrgb.com/index.php?X=MATH&H=14#text14

    Examples
    --------
    >>> CMYK = np.array([0.5, 0, 0.744, 0.01960784])
    >>> CMYK_to_CMY(CMYK)  # doctest: +ELLIPSIS
    array([ 0.5098039...,  0.0196078...,  0.7490196...])
    """

    C, M, Y, K = np.ravel(CMYK)

    return np.array(
        [C * (1 - K) + K, M * (1 - K) + K, Y * (1 - K) + K])
