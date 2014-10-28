#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE 1994 Chromatic Adaptation Model
===================================

Defines CIE 1994 chromatic adaptation model objects:

-   :func:`chromatic_adaptation_CIE1994`

See Also
--------
`CIE 1994 Chromatic Adaptation Model IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/adaptation/cie1994.ipynb>`_  # noqa

References
----------
.. [1]  CIE TC 1-32. (1994). CIE 109-1994 A Method of Predicting Corresponding
        Colours under Different Chromatic and Illuminance Adaptations
        (pp. 1–18). ISBN:978-3-900734-51-0
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.adaptation import VON_KRIES_CAT
from colour.utilities import warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CIE1994_XYZ_TO_RGB_MATRIX',
           'CIE1994_RGB_TO_XYZ_MATRIX',
           'chromatic_adaptation_CIE1994',
           'XYZ_to_RGB_cie1994',
           'RGB_to_XYZ_cie1994',
           'intermediate_values',
           'effective_adapting_responses',
           'beta_1',
           'beta_2',
           'exponential_factors',
           'K_coefficient',
           'corresponding_colour']

CIE1994_XYZ_TO_RGB_MATRIX = VON_KRIES_CAT
"""
CIE 1994 colour appearance model *CIE XYZ* colourspace to cone
responses matrix.

CIE1994_XYZ_TO_RGB_MATRIX : array_like, (3, 3)
"""

CIE1994_RGB_TO_XYZ_MATRIX = np.linalg.inv(CIE1994_XYZ_TO_RGB_MATRIX)
"""
CIE 1994 colour appearance model cone responses to *CIE XYZ* colourspace
matrix.

CIE1994_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""


def chromatic_adaptation_CIE1994(XYZ_1,
                                 xy_o1,
                                 xy_o2,
                                 Y_o,
                                 E_o1,
                                 E_o2,
                                 n=1):
    """
    Adapts given *CIE XYZ_1* colourspace stimulus from test viewing conditions
    to reference viewing conditions using CIE 1994 chromatic adaptation
    model.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    xy_o1 : array_like, (2,)
        Chromaticity coordinates :math:`x_{o1}` and :math:`y_{o1}` of test
        illuminant and background.
    xy_o2 : array_like, (2,)
        Chromaticity coordinates :math:`x_{o2}` and :math:`y_{o2}` of reference
        illuminant and background.
    Y_o : numeric
        Luminance factor :math:`Y_o` of achromatic background as percentage in
        domain [18, 100].
    E_o1 : numeric
        Test illuminance :math:`E_{o1}` in :math:`cd/m^2`.
    E_o2 : numeric
        Reference illuminance :math:`E_{o2}` in :math:`cd/m^2`.
    n : numeric, optional
        Noise component in fundamental primary system.

    Returns
    -------
    ndarray, (3,)
        Adapted *CIE XYZ_2* colourspace test stimulus.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ_1* colourspace matrix is in domain [0, 100].
    -   Output *CIE XYZ_2* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> XYZ_1 = np.array([28.0, 21.26, 5.27])
    >>> xy_o1 = (0.4476, 0.4074)
    >>> xy_o2 = (0.3127, 0.3290)
    >>> Y_o = 20
    >>> E_o1 = 1000
    >>> E_o2 = 1000
    >>> chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)  # noqa  # doctest: +ELLIPSIS
    array([ 24.0337952...,  21.1562121...,  17.6430119...])
    """

    if not 18 <= Y_o <= 100:
        warning(('"Y_o" luminance factor must be in [18, 100] domain, '
                 'unpredictable results may occur!'))

    XYZ_1 = np.ravel(XYZ_1)

    RGB_1 = XYZ_to_RGB_cie1994(XYZ_1)

    xez_1 = intermediate_values(xy_o1)
    xez_2 = intermediate_values(xy_o2)

    RGB_o1 = effective_adapting_responses(Y_o, E_o1, xez_1)
    RGB_o2 = effective_adapting_responses(Y_o, E_o2, xez_2)

    bRGB_o1 = exponential_factors(RGB_o1)
    bRGB_o2 = exponential_factors(RGB_o2)

    K = K_coefficient(Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2, n)

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
    >>> XYZ = np.array([28.0, 21.26, 5.27])
    >>> XYZ_to_RGB_cie1994(XYZ)  # doctest: +ELLIPSIS
    array([ 25.8244273...,  18.6791422...,   4.8390194...])
    """

    return np.dot(CIE1994_XYZ_TO_RGB_MATRIX, XYZ)


def RGB_to_XYZ_cie1994(RGB):
    """
    Converts from cone responses to *CIE XYZ* colourspace.

    Parameters
    ----------
    RGB : array_like, (3,)
        Cone responses.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* colourspace matrix.

    Examples
    --------
    >>> RGB = np.array([25.8244273, 18.6791422, 4.8390194])
    >>> RGB_to_XYZ_cie1994(RGB)  # doctest: +ELLIPSIS
    array([ 28.  ,  21.26,   5.27])
    """

    return np.dot(CIE1994_RGB_TO_XYZ_MATRIX, RGB)


def intermediate_values(xy_o):
    """
    Returns the intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.

    Parameters
    ----------
    xy_o : array_like, (2,)
        Chromaticity coordinates :math:`x_o` and :math:`y_o` of whitepoint.

    Returns
    -------
    ndarray, (3,)
        Intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.

    Examples
    --------
    >>> xy_o = (0.4476, 0.4074)
    >>> intermediate_values(xy_o)  # doctest: +ELLIPSIS
    array([ 1.1185719...,  0.9329553...,  0.3268087...])
    """

    x_o, y_o = xy_o

    # Computing :math:`\xi`, :math:`\eta`, :math:`\zeta` values.
    xi = (0.48105 * x_o + 0.78841 * y_o - 0.08081) / y_o
    eta = (-0.27200 * x_o + 1.11962 * y_o + 0.04570) / y_o
    zeta = (0.91822 * (1 - x_o - y_o)) / y_o

    return np.array([xi, eta, zeta])


def effective_adapting_responses(Y_o, E_o, xez):
    """
    Derives the effective adapting responses in the fundamental primary system
    of the test or reference field.

    Parameters
    ----------
    Y_o : numeric
        Luminance factor :math:`Y_o` of achromatic background as percentage in
        domain [18, 100].
    E_o : numeric
        Test or reference illuminance :math:`E_{o}` in lux.
    xez: ndarray, (3,)
        Intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.

    Returns
    -------
    ndarray, (3,)
        Effective adapting responses.

    Examples
    --------
    >>> Y_o = 20
    >>> E_o = 1000
    >>> xez = np.array([1.11857195, 0.9329553, 0.32680879])
    >>> effective_adapting_responses(Y_o, E_o, xez)  # doctest: +ELLIPSIS
    array([ 71.2105020...,  59.3937790...,  20.8052937...])
    """

    RGB_o = ((Y_o * E_o) / (100 * np.pi)) * xez

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
    :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)` of given cone responses.

    Parameters
    ----------
    RGB_o: ndarray, (3,)
         Cone responses.

    Returns
    -------
    ndarray, (3,)
        Chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
        :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)`.

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


def K_coefficient(Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2, n=1):
    """
    Computes the coefficient :math:`K` for correcting the difference between
    the test and references illuminances.

    Parameters
    ----------
    Y_o : numeric
        Luminance factor :math:`Y_o` of achromatic background as percentage in
        domain [18, 100].
    xez_1: ndarray, (3,)
        Intermediate values :math:`\\xi_1`, :math:`\eta_1`, :math:`\zeta_1` for
        the test illuminant and background.
    xez_2: ndarray, (3,)
        Intermediate values :math:`\\xi_2`, :math:`\eta_2`, :math:`\zeta_2` for
        the reference illuminant and background.
    bRGB_o1: ndarray, (3,)
        Chromatic adaptation exponential factors :math:`\\beta_1(R_{o1})`,
        :math:`\\beta_1(G_{o1})` and :math:`\\beta_2(B_{o1})` of test sample.
    bRGB_o2: ndarray, (3,)
        Chromatic adaptation exponential factors :math:`\\beta_1(R_{o2})`,
        :math:`\\beta_1(G_{o2})` and :math:`\\beta_2(B_{o2})` of reference
        sample.
    n : numeric, optional
        Noise component in fundamental primary system.

    Returns
    -------
    numeric
        Coefficient :math:`K`.

    Examples
    --------
    >>> Y_o = 20
    >>> xez_1 = np.array([1.11857195, 0.9329553, 0.32680879])
    >>> xez_2 = np.array([1.00000372, 1.00000176, 0.99999461])
    >>> bRGB_o1 = np.array([3.74852518, 3.63920879, 2.78924811])
    >>> bRGB_o2 = np.array([3.68102374, 3.68102256, 3.56557351])
    >>> K_coefficient(Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2)
    1.0
    """

    xi_1, eta_1, zeta_1 = xez_1
    xi_2, eta_2, zeta_2 = xez_2

    bR_o1, bG_o1, bB_o1 = bRGB_o1
    bR_o2, bG_o2, bB_o2 = bRGB_o2

    K = (((Y_o * xi_1 + n) / (20 * xi_1 + n)) ** ((2 / 3) * bR_o1) /
         ((Y_o * xi_2 + n) / (20 * xi_2 + n)) ** ((2 / 3) * bR_o2))

    K *= (((Y_o * eta_1 + n) / (20 * eta_1 + n)) ** ((1 / 3) * bG_o1) /
          ((Y_o * eta_2 + n) / (20 * eta_2 + n)) ** ((1 / 3) * bG_o2))
    return K


def corresponding_colour(RGB_1, Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2, K, n=1):
    """
    Computes the corresponding colour cone responses of given test sample cone
    responses :math:`RGB_1`.

    Parameters
    ----------
    RGB_1: ndarray, (3,)
        Test sample cone responses :math:`RGB_1`.
    Y_o : numeric
        Luminance factor :math:`Y_o` of achromatic background as percentage in
        domain [18, 100].
    xez_1: ndarray, (3,)
        Intermediate values :math:`\\xi_1`, :math:`\eta_1`, :math:`\zeta_1` for
        the test illuminant and background.
    xez_2: ndarray, (3,)
        Intermediate values :math:`\\xi_2`, :math:`\eta_2`, :math:`\zeta_2` for
        the reference illuminant and background.
    bRGB_o1: ndarray, (3,)
        Chromatic adaptation exponential factors :math:`\\beta_1(R_{o1})`,
        :math:`\\beta_1(G_{o1})` and :math:`\\beta_2(B_{o1})` of test sample.
    bRGB_o2: ndarray, (3,)
        Chromatic adaptation exponential factors :math:`\\beta_1(R_{o2})`,
        :math:`\\beta_1(G_{o2})` and :math:`\\beta_2(B_{o2})` of reference
        sample.
    K : numeric
        Coefficient :math:`K`.
    n : numeric, optional
        Noise component in fundamental primary system.

    Returns
    -------
    ndarray, (3,)
        Corresponding colour cone responses of given test sample cone
        responses.

    Examples
    --------
    >>> RGB_1 = np.array([25.8244273, 18.6791422, 4.8390194])
    >>> Y_o = 20
    >>> xez_1 = np.array([1.11857195, 0.9329553, 0.32680879])
    >>> xez_2 = np.array([1.00000372, 1.00000176, 0.99999461])
    >>> bRGB_o1 = np.array([3.74852518, 3.63920879, 2.78924811])
    >>> bRGB_o2 = np.array([3.68102374, 3.68102256, 3.56557351])
    >>> K = 1.0
    >>> corresponding_colour(RGB_1, Y_o, xez_1, xez_2, bRGB_o1, bRGB_o2, K)  # noqa  # doctest: +ELLIPSIS
    array([ 23.1636901...,  20.0211948...,  16.2001664...])
    """

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
