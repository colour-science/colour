#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE 1994 Chromatic Adaptation Model
===================================

Defines *CIE 1994* chromatic adaptation model objects:


See Also
--------
`CIE 1994 Chromatic Adaptation Model IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/adaptation/cie1994.ipynb>`_  # noqa

References
----------
"""

from __future__ import division, unicode_literals

import math
import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = []

CIE1994_XYZ_TO_RGB_MATRIX = np.array(
    [[0.40024, 0.70760, -0.08081],
     [-0.22630, 1.16532, 0.04570],
     [0.00000, 0.00000, 0.91822]])
"""
*CIE 1994* colour appearance model *CIE XYZ* colourspace to cone
responses matrix.

CIE1994_XYZ_TO_RGB_MATRIX : array_like, (3, 3)
"""

CIE1994_RGB_TO_XYZ_MATRIX = np.linalg.inv(CIE1994_XYZ_TO_RGB_MATRIX)


def cie1994(XYZ_1,
            xy_o1,
            xy_o2,
            E_o1,
            E_o2,
            Y_o,  # should be between 18 and 100---> issue warnings
            n=1):
    XYZ_1 = np.ravel(XYZ_1)

    RGB_1 = XYZ_to_RGB_cie1994(XYZ_1)

    xez_1 = intermediate_values(xy_o1)
    xez_2 = intermediate_values(xy_o2)

    RGB_o1 = effective_adapting_responses(Y_o, E_o1, xez_1)
    RGB_o2 = effective_adapting_responses(Y_o, E_o2, xez_2)

    bRGB_o1 = exponential_factors(RGB_o1)
    bRGB_o2 = exponential_factors(RGB_o2)

    K = coefficient_K(Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2, n)

    RGB_2 = corresponding_colour(
        RGB_1, Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2, K, n)
    XYZ_2 = RGB_to_XYZ_cie1994(RGB_2)

    return XYZ_2


def XYZ_to_RGB_cie1994(XYZ):
    """
    Converts from *CIE XYZ* colourspace to cone responses.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        Cone responses.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20, 21.78])
    >>> XYZ_to_RGB_cie1994(XYZ)  # doctest: +ELLIPSIS
    array([ 20.000520...,  19.999783...,  19.998831...])
    """

    return CIE1994_XYZ_TO_RGB_MATRIX.dot(XYZ)


def RGB_to_XYZ_cie1994(RGB):
    return CIE1994_RGB_TO_XYZ_MATRIX.dot(RGB)


def intermediate_values(xy_o):
    """
    Returns the intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.

    Parameters
    ----------
    xy_o : array_like
        Chromaticity coordinates *xy* of whitepoint.

    Returns
    -------
    ndarray, (3,)
        Intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.

    Examples
    --------
    """

    # Illuminant chromaticity coordinates.
    x_o, y_o = xy_o

    # Computing :math:`\xi`, :math:`\eta`, :math:`\zeta` values.
    xi = (0.48105 * x_o + 0.78841 * y_o - 0.08081) / y_o
    eta = (-0.27200 * x_o + 1.11962 * y_o + 0.04570) / y_o
    zeta = (0.91822 * (1 - x_o - y_o)) / y_o

    return np.array([xi, eta, zeta])


def effective_adapting_responses(Y_o, E_o, xez):
    """
    """

    RGB_o = ((Y_o * E_o) / (100 * math.pi)) * xez

    return RGB_o


def beta_1(x):
    """
    Computes the exponent :math:`\\beta_1` for the middle and long-wavelength
    sensitive cones.

    Parameters
    ----------
    x: numeric
        Middle and long-wavelength sensitive cone response.

    Returns
    -------
    numeric
        Exponent :math:`\\beta_1`.

    Examples
    --------
    >>> beta_1(318.323316315)  # doctest: +ELLIPSIS
    4.6106222...
    """

    return (6.469 + 6.362 * (x ** 0.4495)) / (6.469 + (x ** 0.4495))


def beta_2(x):
    """
    Computes the exponent :math:`\\beta_2` for the short-wavelength sensitive
    cones.

    Parameters
    ----------
    x: numeric
        Short-wavelength sensitive cone response.

    Returns
    -------
    numeric
        Exponent :math:`\\beta_2`.

    Examples
    --------
    >>> beta_2(318.323316315)  # doctest: +ELLIPSIS
    4.6522416...
    """

    return 0.7844 * (8.414 + 8.091 * (x ** 0.5128)) / (8.414 + (x ** 0.5128))


def exponential_factors(RGB_o):
    """
    Returns the chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
    `math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)` of given cone responses.

    Parameters
    ----------
    RGB_o: ndarray, (3,)
         Cone responses.

    Returns
    -------
    ndarray, (3,)
        Chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
        `math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)`.

    Examples
    --------
    >>> RGB_o = np.array([318.32331631, 318.30352317, 318.23283482])
    >>> exponential_factors(RGB_o)  # doctest: +ELLIPSIS
    array([ 4.6106222...,  4.6105892...,  4.6520698...])
    """

    R_o, G_o, B_o = np.ravel(RGB_o)

    bR_o = beta_1(R_o)
    bG_o = beta_1(G_o)
    bB_o = beta_2(B_o)

    return np.array([bR_o, bG_o, bB_o])


def coefficient_K(Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2, n):
    xi_1, eta_1, zeta_1 = xez_1
    xi_2, eta_2, zeta_2 = xez_2

    bR_o1, bG_o1, bB_o1 = bRGB_o1
    bR_o2, bG_o2, bB_o2 = bRGB_o2

    K = (((Y_o * xi_1 + n) / (20 * xi_1 + n)) ** ((2 / 3) * bR_o1) /
         ((Y_o * xi_2 + n) / (20 * xi_2 + n)) ** ((2 / 3) * bR_o2))

    K *= (((Y_o * eta_1 + n) / (20 * eta_1 + n)) ** ((1 / 3) * bG_o1) /
          ((Y_o * eta_2 + n) / (20 * eta_2 + n)) ** ((1 / 3) * bG_o2))
    return K


def corresponding_colour(RGB_1, Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2, K, n):
    R_1, G_1, B_1 = RGB_1
    xi_1, eta_1, zeta_1 = xez_1
    xi_2, eta_2, zeta_2 = xez_2

    bR_o1, bG_o1, bB_o1 = bRGB_o1
    bR_o2, bG_o2, bB_o2 = bRGB_o2

    RGBc = lambda x1, x2, y1, y2, z: (
        (Y_o * x2 + n) * K ** (1 / y2) *
        ((z + n) / (Y_o * x1 + n)) ** (y1 / y2) - n)

    R_2 = RGBc(xi_1, xi_2, bR_o1, bR_o2, R_1)
    G_2 = RGBc(eta_1, eta_2, bG_o1, bG_o2, G_1)
    B_2 = RGBc(zeta_1, zeta_2, bB_o1, bB_o2, B_1)

    return np.array([R_2, G_2, B_2])


print(cie1994(XYZ_1=np.array([28.0, 21.26, 5.27]),
              xy_o1=(0.4476, 0.4074),
              xy_o2=(0.3127, 0.3290),
              E_o1=1000,
              E_o2=1000,
              Y_o=20))

# [ 24.03379521  21.15621214  17.64301199]