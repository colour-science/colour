#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fairchild (1990) Chromatic Adaptation Model
===========================================

Defines *Fairchild (1990)* chromatic adaptation model objects:

-   :func:`chromatic_adaptation_Fairchild1990`

See Also
--------
`Fairchild (1990) Chromatic Adaptation Model Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/adaptation/fairchild1990.ipynb>`_

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
from colour.utilities import dot_vector, row_as_diagonal, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FAIRCHILD1990_XYZ_TO_RGB_MATRIX',
           'FAIRCHILD1990_RGB_TO_XYZ_MATRIX',
           'chromatic_adaptation_Fairchild1990',
           'XYZ_to_RGB_Fairchild1990',
           'RGB_to_XYZ_Fairchild1990',
           'degrees_of_adaptation']

FAIRCHILD1990_XYZ_TO_RGB_MATRIX = VON_KRIES_CAT
"""
*Fairchild (1990)* colour appearance model *CIE XYZ* tristimulus values to cone
responses matrix.

FAIRCHILD1990_XYZ_TO_RGB_MATRIX : array_like, (3, 3)
"""

FAIRCHILD1990_RGB_TO_XYZ_MATRIX = np.linalg.inv(VON_KRIES_CAT)
"""
*Fairchild (1990)* colour appearance model cone responses to *CIE XYZ*
tristimulus values matrix.

FAIRCHILD1990_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""


def chromatic_adaptation_Fairchild1990(XYZ_1,
                                       XYZ_n,
                                       XYZ_r,
                                       Y_n,
                                       discount_illuminant=False):
    """
    Adapts given stimulus *CIE XYZ_1* tristimulus values from test viewing
    conditions to reference viewing conditions using *Fairchild (1990)*
    chromatic adaptation model.

    Parameters
    ----------
    XYZ_1 : array_like
        *CIE XYZ_1* tristimulus values of test sample / stimulus in domain
        [0, 100].
    XYZ_n : array_like
        Test viewing condition *CIE XYZ_n* tristimulus values of whitepoint.
    XYZ_r : array_like
        Reference viewing condition *CIE XYZ_r* tristimulus values of
        whitepoint.
    Y_n : numeric or array_like
        Luminance :math:`Y_n` of test adapting stimulus in :math:`cd/m^2`.
    discount_illuminant : bool, optional
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    ndarray
        Adapted *CIE XYZ_2* tristimulus values of stimulus.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ_1*, *CIE XYZ_n* and *CIE XYZ_r* tristimulus values are
        in domain [0, 100].
    -   Output *CIE XYZ_2* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> XYZ_1 = np.array([19.53, 23.07, 24.97])
    >>> XYZ_n = np.array([111.15, 100.00, 35.20])
    >>> XYZ_r = np.array([94.81, 100.00, 107.30])
    >>> Y_n = 200
    >>> chromatic_adaptation_Fairchild1990(  # doctest: +ELLIPSIS
    ...     XYZ_1, XYZ_n, XYZ_r, Y_n)
    array([ 23.3252634...,  23.3245581...,  76.1159375...])
    """

    XYZ_1 = np.asarray(XYZ_1)
    XYZ_n = np.asarray(XYZ_n)
    XYZ_r = np.asarray(XYZ_r)
    Y_n = np.asarray(Y_n)

    LMS_1 = dot_vector(FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ_1)
    LMS_n = dot_vector(FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ_n)
    LMS_r = dot_vector(FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ_r)

    p_LMS = degrees_of_adaptation(LMS_1,
                                  Y_n,
                                  discount_illuminant=discount_illuminant)

    a_LMS_1 = p_LMS / LMS_n
    a_LMS_2 = p_LMS / LMS_r

    A_1 = row_as_diagonal(a_LMS_1)
    A_2 = row_as_diagonal(a_LMS_2)

    LMSp_1 = dot_vector(A_1, LMS_1)

    c = 0.219 - 0.0784 * np.log10(Y_n)
    C = row_as_diagonal(tstack((c, c, c)))

    LMS_a = dot_vector(C, LMSp_1)
    LMSp_2 = dot_vector(np.linalg.inv(C), LMS_a)

    LMS_c = dot_vector(np.linalg.inv(A_2), LMSp_2)
    XYZ_c = dot_vector(FAIRCHILD1990_RGB_TO_XYZ_MATRIX, LMS_c)

    return XYZ_c


def XYZ_to_RGB_Fairchild1990(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to cone responses.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        Cone responses.

    Examples
    --------
    >>> XYZ = np.array([19.53, 23.07, 24.97])
    >>> XYZ_to_RGB_Fairchild1990(XYZ)  # doctest: +ELLIPSIS
    array([ 22.1231935...,  23.6054224...,  22.9279534...])
    """

    return dot_vector(FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ)


def RGB_to_XYZ_Fairchild1990(RGB):
    """
    Converts from cone responses to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB : array_like
        Cone responses.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> RGB = np.array([22.12319350, 23.60542240, 22.92795340])
    >>> RGB_to_XYZ_Fairchild1990(RGB)  # doctest: +ELLIPSIS
    array([ 19.53,  23.07,  24.97])
    """

    return dot_vector(FAIRCHILD1990_RGB_TO_XYZ_MATRIX, RGB)


def degrees_of_adaptation(LMS, Y_n, v=1 / 3, discount_illuminant=False):
    """
    Computes the degrees of adaptation :math:`p_L`, :math:`p_M` and
    :math:`p_S`.

    Parameters
    ----------
    LMS : array_like
        Cone responses.
    Y_n : numeric or array_like
        Luminance :math:`Y_n` of test adapting stimulus in :math:`cd/m^2`.
    v : numeric or array_like, optional
        Exponent :math:`v`.
    discount_illuminant : bool, optional
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    ndarray
        Degrees of adaptation :math:`p_L`, :math:`p_M` and :math:`p_S`.

    Examples
    --------
    >>> LMS = np.array([20.00052060, 19.99978300, 19.99883160])
    >>> Y_n = 31.83
    >>> degrees_of_adaptation(LMS, Y_n)  # doctest: +ELLIPSIS
    array([ 0.9799324...,  0.9960035...,  1.0233041...])
    >>> degrees_of_adaptation(LMS, Y_n, 1 / 3, True)
    array([ 1.,  1.,  1.])
    """

    LMS = np.asarray(LMS)
    if discount_illuminant:
        return np.ones(LMS.shape)

    Y_n = np.asarray(Y_n)
    v = np.asarray(v)

    L, M, S = tsplit(LMS)

    LMS_E = dot_vector(VON_KRIES_CAT, np.ones(LMS.shape))  # E illuminant.
    L_E, M_E, S_E = tsplit(LMS_E)

    Ye_n = Y_n ** v

    def m_E(x, y):
        """
        Computes the :math:`m_E` term.
        """

        return (3 * (x / y)) / (L / L_E + M / M_E + S / S_E)

    def P_c(x):
        """
        Computes the :math:`P_L`, :math:`P_M` or :math:`P_S` terms.
        """

        return (1 + Ye_n + x) / (1 + Ye_n + 1 / x)

    p_L = P_c(m_E(L, L_E))
    p_M = P_c(m_E(M, M_E))
    p_S = P_c(m_E(S, S_E))

    p_LMS = tstack((p_L, p_M, p_S))

    return p_LMS
