#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fairchild (1990) Chromatic Adaptation Model
===========================================

Defines Fairchild (1990) chromatic adaptation model objects:

-   :func:`chromatic_adaptation_Fairchild1990`

See Also
--------
`Fairchild (1990) Chromatic Adaptation Model IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/adaptation/fairchild1990.ipynb>`_  # noqa

References
----------
.. [1]  Fairchild, M. D. (1991). Formulation and testing of an
        incomplete-chromatic-adaptation model. Color Research & Application,
        16(4), 243–250. doi:10.1002/col.5080160406
.. [2]  Fairchild, M. D. (2013). FAIRCHILD’S 1990 MODEL. In Color Appearance
        Models (3rd ed., pp. 4418–4495). Wiley. ASIN:B00DAYO8E2
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.adaptation import VON_KRIES_CAT

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FAIRCHILD1990_XYZ_TO_RGB_MATRIX',
           'FAIRCHILD1990_RGB_TO_XYZ_MATRIX',
           'chromatic_adaptation_Fairchild1990',
           'XYZ_to_RGB_fairchild1990',
           'RGB_to_XYZ_fairchild1990',
           'degrees_of_adaptation']

FAIRCHILD1990_XYZ_TO_RGB_MATRIX = VON_KRIES_CAT
"""
Fairchild (1990) colour appearance model *CIE XYZ* colourspace to cone
responses matrix.

FAIRCHILD1990_XYZ_TO_RGB_MATRIX : array_like, (3, 3)
"""

FAIRCHILD1990_RGB_TO_XYZ_MATRIX = np.linalg.inv(VON_KRIES_CAT)
"""
Fairchild (1990) colour appearance model cone responses to *CIE XYZ*
colourspace to  matrix.

FAIRCHILD1990_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""


def chromatic_adaptation_Fairchild1990(XYZ_1,
                                       XYZ_n,
                                       XYZ_r,
                                       Y_n,
                                       discount_illuminant=False):
    """
    Adapts given *CIE XYZ_1* colourspace stimulus from test viewing conditions
    to reference viewing conditions using Fairchild (1990) chromatic
    adaptation model.

    Parameters
    ----------
    XYZ_1 : array_like, (3,)
        *CIE XYZ_1* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    XYZ_n : array_like, (3,)
        Test viewing condition *CIE XYZ_n* colourspace whitepoint matrix.
    XYZ_r : array_like, (3,)
        Reference viewing condition *CIE XYZ_r* colourspace whitepoint matrix.
    Y_n : numeric
        Luminance :math:`Y_n` of test adapting stimulus in :math:`cd/m^2`.
    discount_illuminant : bool, optional
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    ndarray, (3,)
        Adapted *CIE XYZ_2* colourspace test stimulus.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ_1*, *CIE XYZ_n* and *CIE XYZ_r* colourspace matrices are
        in domain [0, 100].
    -   Output *CIE XYZ_2* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> XYZ_1 = np.array([19.53, 23.07, 24.97])
    >>> XYZ_n = np.array([111.15, 100.00, 35.20])
    >>> XYZ_r = np.array([94.81, 100.00, 107.30])
    >>> Y_n = 200
    >>> chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n)  # noqa  # doctest: +ELLIPSIS
    array([ 23.3252634...,  23.3245581...,  76.1159375...])
    """

    XYZ_1, XYZ_n, XYZ_r = np.ravel(XYZ_1), np.ravel(XYZ_n), np.ravel(XYZ_r)

    LMS_1 = np.dot(FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ_1)
    LMS_n = np.dot(FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ_n)
    LMS_r = np.dot(FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ_r)

    p_LMS = degrees_of_adaptation(LMS_1,
                                  Y_n,
                                  discount_illuminant=discount_illuminant)

    a_LMS_1 = p_LMS / LMS_n
    a_LMS_2 = p_LMS / LMS_r

    diagonal = lambda x: np.diagflat(x).reshape((3, 3))
    A_1 = diagonal(a_LMS_1)
    A_2 = diagonal(a_LMS_2)

    LMSp_1 = np.dot(A_1, LMS_1)

    c = 0.219 - 0.0784 * np.log10(Y_n)
    C = np.array([[c, 0, 0], [0, c, 0], [0, 0, c]])

    LMS_a = np.dot(C, LMSp_1)
    LMSp_2 = np.dot(np.linalg.inv(C), LMS_a)

    LMS_c = np.dot(np.linalg.inv(A_2), LMSp_2)
    XYZ_c = np.dot(FAIRCHILD1990_RGB_TO_XYZ_MATRIX, LMS_c)

    return XYZ_c


def XYZ_to_RGB_fairchild1990(XYZ):
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
    >>> XYZ = np.array([19.53, 23.07, 24.97])
    >>> XYZ_to_RGB_fairchild1990(XYZ)  # doctest: +ELLIPSIS
    array([ 22.1231935...,  23.6054224...,  22.9279534...])
    """

    return np.dot(FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ)


def RGB_to_XYZ_fairchild1990(RGB):
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
    >>> RGB = np.array([22.1231935, 23.6054224, 22.9279534])
    >>> RGB_to_XYZ_fairchild1990(RGB)  # doctest: +ELLIPSIS
    array([ 19.53,  23.07,  24.97])
    """

    return np.dot(FAIRCHILD1990_RGB_TO_XYZ_MATRIX, RGB)


def degrees_of_adaptation(LMS, Y_n, v=1 / 3, discount_illuminant=False):
    """
    Computes the degrees of adaptation :math:`p_L`, :math:`p_M` and
    :math:`p_S`.

    Parameters
    ----------
    LMS : array_like, (3,)
        Cone responses.
    Y_n : numeric
        Luminance :math:`Y_n` of test adapting stimulus in :math:`cd/m^2`.
    v : numeric, optional
        Exponent :math:`v`.
    discount_illuminant : bool, optional
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    ndarray, (3,)
        Degrees of adaptation :math:`p_L`, :math:`p_M` and :math:`p_S`.

    Examples
    --------
    >>> LMS = np.array([ 20.0005206,  19.999783 ,  19.9988316])
    >>> Y_n = 31.83
    >>> degrees_of_adaptation(LMS, Y_n)  # doctest: +ELLIPSIS
    array([ 0.9799324...,  0.9960035...,  1.0233041...])
    >>> degrees_of_adaptation(LMS, Y_n, 1 / 3, True)
    array([1, 1, 1])
    """

    if discount_illuminant:
        return np.array([1, 1, 1])

    L, M, S = np.ravel(LMS)

    LMS_E = np.dot(VON_KRIES_CAT, np.array([1, 1, 1]))  # E illuminant.
    L_E, M_E, S_E = np.ravel(LMS_E)

    Ye_n = Y_n ** v

    f_E = lambda x, y: (3 * (x / y)) / (L / L_E + M / M_E + S / S_E)
    f_P = lambda x: (1 + Ye_n + x) / (1 + Ye_n + 1 / x)

    p_L = f_P(f_E(L, L_E))
    p_M = f_P(f_E(M, M_E))
    p_S = f_P(f_E(S, S_E))

    return np.array([p_L, p_M, p_S])
