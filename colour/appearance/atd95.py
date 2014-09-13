#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ATD (1995) Colour Vision Model
==============================

Defines *ATD (1995)* colour vision model objects:

-   :class:`ATD95_Specification`
-   :func:`XYZ_to_ATD95`

See Also
--------
`ATD (1995) Colour Vision Model IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/appearance/atd95.ipynb>`_  # noqa

Notes
-----
-   According to *CIE TC1-34* definition of a colour appearance model, the
    *ATD95* model cannot be considered as a colour appearance model. It was
    developed with different aims and is described as a model of colour vision.

References
----------
.. [1]  **Mark D. Fairchild**, *Color Appearance Models, 3nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published June 2013, ASIN: B00DAYO8E2,
        Locations 5841-5991.
.. [2]  **S. Lee Guth**,
        *Further applications of the ATD model for color vision*,
        *IS&T/SPIE's Symposium on Electronic Imaging: Science & Technology*,
        *International Society for Optics and Photonics*,
        pages 12-26,
        DOI: http://dx.doi.org/10.1117/12.206546
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ATD95_ReferenceSpecification',
           'ATD95_Specification',
           'XYZ_to_ATD95',
           'luminance_to_retinal_illuminance',
           'XYZ_to_LMS_ATD95',
           'opponent_colour_dimensions',
           'final_response']


class ATD95_ReferenceSpecification(
    namedtuple('ATD95_ReferenceSpecification',
               ('H', 'C', 'Br', 'A_1', 'T_1', 'D_1', 'A_2', 'T_2', 'D_2'))):
    """
    Defines the *ATD (1995)* colour vision model reference specification.

    This specification has field names consistent with **Mark D. Fairchild**
    reference.

    Parameters
    ----------
    H : numeric
        *Hue* angle :math:`H` in degrees.
    C : numeric
        Correlate of *saturation* :math:`C`. *Guth (1995)* incorrectly uses the
        terms saturation and chroma interchangeably. However, :math:`C` is here
        a measure of saturation rather than chroma since it is measured
        relative to the achromatic response for the stimulus rather than that
        of a similarly illuminated white.
    Br : numeric
        Correlate of *brightness* :math:`Br`.
    A_1 : numeric
        First stage :math:`A_1` response.
    T_1 : numeric
        First stage :math:`T_1` response.
    D_1 : numeric
        First stage :math:`D_1` response.
    A_2 : numeric
        Second stage :math:`A_2` response.
    T_2 : numeric
        Second stage :math:`A_2` response.
    D_2 : numeric
        Second stage :math:`D_2` response.
    """


class ATD95_Specification(
    namedtuple('ATD95_Specification',
               ('h', 'C', 'Q', 'A_1', 'T_1', 'D_1', 'A_2', 'T_2', 'D_2'))):
    """
    Defines the *ATD (1995)* colour vision model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    **Mark D. Fairchild** reference.

    Notes
    -----
    -   This specification is the one used in the current model implementation.

    Parameters
    ----------
    h : numeric
        *Hue* angle :math:`H` in degrees.
    C : numeric
        Correlate of *saturation* :math:`C`. *Guth (1995)* incorrectly uses the
        terms saturation and chroma interchangeably. However, :math:`C` is here
        a measure of saturation rather than chroma since it is measured
        relative to the achromatic response for the stimulus rather than that
        of a similarly illuminated white.
    Q : numeric
        Correlate of *brightness* :math:`Br`.
    A_1 : numeric
        First stage :math:`A_1` response.
    T_1 : numeric
        First stage :math:`T_1` response.
    D_1 : numeric
        First stage :math:`D_1` response.
    A_2 : numeric
        Second stage :math:`A_2` response.
    T_2 : numeric
        Second stage :math:`A_2` response.
    D_2 : numeric
        Second stage :math:`D_2` response.
    """


def XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2, sigma=300):
    """
    Computes the *ATD (1995)* colour vision model correlates.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    XYZ_0 : array_like, (3,)
        *CIE XYZ* colourspace matrix of reference white in domain [0, 100].
    Y_0 : numeric
        Absolute adapting field luminance in :math:`cd/m^2`.
    k_1 : numeric
        Application specific weight :math:`k_1`.
    k_2 : numeric
        Application specific weight :math:`k_2`.
    sigma : numeric, optional
        Constant :math:`\sigma` varied to predict different types of data.

    Returns
    -------
    ATD95_Specification
        *ATD (1995)* colour vision model specification.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    -   Input *CIE XYZ_0* colourspace matrix is in domain [0, 100].
    -   For unrelated colors, there is only self-adaptation, and :math:`k_1` is
        set to 1.0 while :math:`k_2` is set to 0.0. For related colors such as
        typical colorimetric applications, :math:`k_1` is set to 0.0 and
        :math:`k_2` is set to a value between 15 and 50 *Guth (1995)*.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_0 = np.array([95.05, 100.00, 108.88])
    >>> Y_0 = 318.31
    >>> k_1 = 0.0
    >>> k_2 = 50.0
    >>> XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)  # doctest: +ELLIPSIS
    ATD95_Specification(h=1.9089869..., C=1.2064060..., Q=0.1814003..., A_1=0.1787931... T_1=0.0286942..., D_1=0.0107584..., A_2=0.0192182..., T_2=0.0205377..., D_2=0.0107584...)
    """

    XYZ = luminance_to_retinal_illuminance(XYZ, Y_0)
    XYZ_0 = luminance_to_retinal_illuminance(XYZ_0, Y_0)

    # Computing adaptation model.
    LMS = XYZ_to_LMS_ATD95(XYZ)
    XYZ_a = k_1 * XYZ + k_2 * XYZ_0
    LMS_a = XYZ_to_LMS_ATD95(XYZ_a)

    LMS_g = LMS * (sigma / (sigma + LMS_a))

    # Computing opponent colour dimensions.
    A_1, T_1, D_1, A_2, T_2, D_2 = opponent_colour_dimensions(LMS_g)

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`Br`.
    # -------------------------------------------------------------------------
    Br = (A_1 ** 2 + T_1 ** 2 + D_1 ** 2) ** 0.5

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`C`.
    # -------------------------------------------------------------------------
    C = (T_2 ** 2 + D_2 ** 2) ** 0.5 / A_2

    # -------------------------------------------------------------------------
    # Computing the *hue* :math:`H`.
    # -------------------------------------------------------------------------
    H = T_2 / D_2

    return ATD95_Specification(H, C, Br, A_1, T_1, D_1, A_2, T_2, D_2)


def luminance_to_retinal_illuminance(XYZ, Y_c):
    """
    Converts from luminance in :math:`cd/m^2` to retinal illuminance in
    trolands.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.

    Y_c : numeric
        Absolute adapting field luminance in :math:`cd/m^2`.

    Returns
    -------
    ndarray
        Converted *CIE XYZ* colourspace matrix in trolands.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20., 21.78])
    >>> Y_0 = 318.31
    >>> luminance_to_retinal_illuminance(XYZ, Y_0)  # doctest: +ELLIPSIS
    array([ 479.4445924...,  499.3174313...,  534.5631673...])
    """

    return 18. * (Y_c * XYZ / 100.) ** 0.8


def XYZ_to_LMS_ATD95(XYZ):
    """
    Converts from *CIE XYZ* colourspace to *LMS* cone responses.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *LMS* cone responses.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20., 21.78])
    >>> XYZ_to_LMS_ATD95(XYZ)  # doctest: +ELLIPSIS
    array([ 6.2283272...,  7.4780666...,  3.8859772...])
    """

    X, Y, Z = np.ravel(XYZ)

    L = ((0.66 * (0.2435 * X + 0.8524 * Y - 0.0516 * Z)) ** 0.7) + 0.024
    M = ((-0.3954 * X + 1.1642 * Y + 0.0837 * Z) ** 0.7) + 0.036
    S = ((0.43 * (0.04 * Y + 0.6225 * Z)) ** 0.7) + 0.31

    return np.array([L, M, S])


def opponent_colour_dimensions(LMS_g):
    """
    Returns opponent colour dimensions from given post adaptation cone signals
    matrix.

    Parameters
    ----------
    LMS_g : array_like, (3,)
        Post adaptation cone signals matrix.

    Returns
    -------
    tuple
        Opponent colour dimensions.

    Examples
    --------
    >>> from pprint import pprint
    >>> LMS_g = np.array([6.95457922, 7.08945043, 6.44069316])
    >>> pprint(opponent_colour_dimensions(LMS_g))  # doctest: +ELLIPSIS
    (0.1787931...,
     0.0286942...,
     0.0107584...,
     0.0192182...,
     0.0205377...,
     0.0107584...)
    """

    L_g, M_g, S_g = LMS_g

    A_1i = 3.57 * L_g + 2.64 * M_g
    T_1i = 7.18 * L_g - 6.21 * M_g
    D_1i = -0.7 * L_g + 0.085 * M_g + S_g
    A_2i = 0.09 * A_1i
    T_2i = 0.43 * T_1i + 0.76 * D_1i
    D_2i = D_1i

    A_1 = final_response(A_1i)
    T_1 = final_response(T_1i)
    D_1 = final_response(D_1i)
    A_2 = final_response(A_2i)
    T_2 = final_response(T_2i)
    D_2 = final_response(D_2i)

    return A_1, T_1, D_1, A_2, T_2, D_2


def final_response(value):
    """
    Returns the final response of given opponent colour dimension.

    Parameters
    ----------
    value : numeric
         Opponent colour dimension.

    Returns
    -------
    numeric
        Final response of opponent colour dimension.

    Examples
    --------
    >>> final_response(43.54399695501678)  # doctest: +ELLIPSIS
    0.1787931...
    """

    return value / (200 + abs(value))
