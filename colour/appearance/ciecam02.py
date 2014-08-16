#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIECAM02
========

Defines *CIECAM02* colour appearance model objects:

-   :func:`CIECAM02_Specification`
-   :func:`XYZ_to_CIECAM02`
-   :func:`CIECAM02_to_XYZ`

References
----------
.. [1]  http://en.wikipedia.org/wiki/CIECAM02
        (Last accessed 14 August 2014)
.. [2]  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published 19 November 2004, ISBN-13: 978-0470012161,
        pages 265-277.
.. [3]  **Stephen Westland, Caterina Ripamonti, Vien Cheung**,
        *Computational Colour Science Using MATLAB, 2nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published July 2012, ISBN-13: 978-0-470-66569-5, page  38.
.. [4]  `The CIECAM02 Color Appearance Model
        <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_
        (Last accessed 30 July 2014)
"""

from __future__ import division, unicode_literals

import bisect
import math
import numpy as np
from collections import namedtuple

from colour.adaptation.cat import CAT02_CAT, CAT02_INVERSE_CAT
from colour.appearance.hunt import (HPE_MATRIX,
                                    HPE_MATRIX_INVERSE,
                                    get_luminance_level_adaptation_factor)
from colour.utilities import memoize

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CIECAM02_InductionFactors',
           'CIECAM02_VIEWING_CONDITIONS',
           'HUE_DATA_FOR_HUE_QUADRATURE',
           'CIECAM02_Specification',
           'XYZ_to_CIECAM02',
           'CIECAM02_to_XYZ',
           'get_chromatic_induction_factors',
           'get_base_exponential_non_linearity',
           'get_viewing_condition_dependent_parameters',
           'get_degree_of_adaptation',
           'get_full_chromatic_adaptation_forward',
           'get_full_chromatic_adaptation_reverse',
           'RGB_to_rgb',
           'rgb_to_RGB',
           'post_adaptation_non_linear_response_compression_forward',
           'post_adaptation_non_linear_response_compression_reverse',
           'get_opponent_colour_dimensions_forward',
           'get_opponent_colour_dimensions_reverse',
           'get_hue_angle',
           'get_hue_quadrature',
           'get_eccentricity_factor',
           'get_achromatic_response_forward',
           'get_achromatic_response_reverse',
           'get_lightness_correlate',
           'get_brightness_correlate',
           'get_temporary_magnitude_quantity_forward',
           'get_temporary_magnitude_quantity_reverse',
           'get_chroma_correlate',
           'get_colourfulness_correlate',
           'get_saturation_correlate',
           'get_P',
           'get_post_adaptation_non_linear_response_compression_matrix']

CIECAM02_InductionFactors = namedtuple('CIECAM02_InductionFactors',
                                       ('F', 'c', 'N_c'))

CIECAM02_VIEWING_CONDITIONS = {
    'Average': CIECAM02_InductionFactors(1., 0.69, 1.),
    'Dim': CIECAM02_InductionFactors(0.9, 0.59, 0.95),
    'Dark': CIECAM02_InductionFactors(0.8, 0.525, 0.8)}
"""
Reference *CIECAM02* colour appearance model viewing conditions.

CIECAM02_VIEWING_CONDITIONS : dict
('Average', 'Dim', 'Dark')
"""

HUE_DATA_FOR_HUE_QUADRATURE = {
    'h_i': np.array([20.14, 90.00, 164.25, 237.53, 380.14]),
    'e_i': np.array([0.8, 0.7, 1.0, 1.2, 0.8]),
    'H_i': np.array([0.0, 100.0, 200.0, 300.0, 400.0])}

CIECAM02_Specification = namedtuple('CIECAM02_Specification',
                                    ('J', 'C', 'h', 'Q', 'M', 's', 'H'))
"""
Defines the *CIECAM02* colour appearance model specification.

Parameters
----------
J : float
    Correlate of *Lightness* :math:`J`.
C : float
    Correlate of *chroma* :math:`C`.
h : float
    *Hue* angle :math:`h` in degrees.
Q : float
    Correlate of *brightness* :math:`Q`.
M : float
    Correlate of *colourfulness* :math:`M`.
s : float
    Correlate of *saturation* :math:`s`.
H : float
    Hue :math:`h` quadrature :math:`H`.
"""

_CIECAM02_VIEWING_CONDITION_DEPENDENT_PARAMETERS_CACHE = {}


def XYZ_to_CIECAM02(XYZ,
                    XYZ_w,
                    L_A,
                    Y_b,
                    surround=CIECAM02_VIEWING_CONDITIONS.get('Average'),
                    discount_illuminant=False):
    """
    Computes the *CIECAM02* colour appearance model correlates from given
    *CIE XYZ* colourspace matrix.

    This is the *forward* implementation.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    XYZ_w : array_like, (3,)
        *CIE XYZ* colourspace matrix of reference white in domain [0, 100].
    L_A : float
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    Y_b : float
        Adapting field *Y* tristimulus value :math:`Y_b`.
    surround : CIECAM02_InductionFactors
        Surround viewing conditions induction factors.
    discount_illuminant : bool
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    CIECAM02_Specification
        *CIECAM02* colour appearance model specification.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    -   Input *CIE XYZ_w* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> colour.XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b)
    CIECAM02_Specification(J=41.731091132513917, C=0.1047077571711053, h=-140.9515673417281, Q=195.37132596607671, M=0.1088421756692261, s=2.3603053739204447, H=278.06073585662813)
    """

    XYZ = np.array(XYZ).reshape((3, 1))
    XYZ_w = np.array(XYZ_w).reshape((3, 1))
    X, Y, Z = np.ravel(XYZ)
    X_w, Y_w, Z_w = np.ravel(XYZ_w)

    n, F_L, N_bb, N_cb, z = get_viewing_condition_dependent_parameters(Y_b,
                                                                       Y_w,
                                                                       L_A)

    # Converting *CIE XYZ* colourspace matrices to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB = np.dot(CAT02_CAT, XYZ)
    RGB_w = np.dot(CAT02_CAT, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = get_degree_of_adaptation(surround.F,
                                 L_A) if not discount_illuminant else 1.

    # Computing full chromatic adaptation.
    RGB_c = get_full_chromatic_adaptation_forward(RGB, RGB_w, Y_w, D)
    RGB_wc = get_full_chromatic_adaptation_forward(RGB_w, RGB_w, Y_w, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_p = RGB_to_rgb(RGB_c)
    RGB_pw = RGB_to_rgb(RGB_wc)

    # Applying forward post-adaptation non linear response compression.
    RGB_a = post_adaptation_non_linear_response_compression_forward(
        RGB_p, F_L)
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_pw, F_L)

    # Converting to preliminary cartesian coordinates.
    a, b = get_opponent_colour_dimensions_forward(RGB_a)

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`h`.
    h = get_hue_angle(a, b)
    # -------------------------------------------------------------------------
    # Computing hue :math:`h` quadrature :math:`H`.
    H = get_hue_quadrature(h)

    # Computing eccentricity factor *e_t*.
    e_t = get_eccentricity_factor(h)

    # Computing achromatic responses for the stimulus and the whitepoint.
    A = get_achromatic_response_forward(RGB_a, N_bb)
    A_w = get_achromatic_response_forward(RGB_aw, N_bb)

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`J`.
    # -------------------------------------------------------------------------
    J = get_lightness_correlate(A, A_w, surround.c, z)

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`Q`.
    # -------------------------------------------------------------------------
    Q = get_brightness_correlate(surround.c, J, A_w, F_L)

    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`C`.
    # -------------------------------------------------------------------------
    C = get_chroma_correlate(J, n, surround.N_c, N_cb, e_t, a, b, RGB_a)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`M`.
    # -------------------------------------------------------------------------
    M = get_colourfulness_correlate(C, F_L)

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s`.
    # -------------------------------------------------------------------------
    s = get_saturation_correlate(M, Q)

    return CIECAM02_Specification(J, C, h, Q, M, s, H)


def CIECAM02_to_XYZ(CIECAM02_Specification,
                    XYZ_w,
                    L_A,
                    Y_b,
                    surround=CIECAM02_VIEWING_CONDITIONS.get(
                        'Average'),
                    discount_illuminant=False):
    """
    Converts *CIECAM02* specification to *CIE XYZ* colourspace matrix.

    This is the *reverse* implementation.

    Parameters
    ----------
    CIECAM02_Specification : CIECAM02_Specification
        *CIECAM02* specification.
    XYZ_w : array_like
        *CIE XYZ* colourspace matrix of reference white.
    L_A : float
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    Y_b : float
        Adapting field *Y* tristimulus value :math:`Y_b`.
    surround : CIECAM02_Surround
        Surround viewing conditions.
    discount_illuminant : bool
        Discount the illuminant.

    Returns
    -------
    XYZ : ndarray
        *CIE XYZ* colourspace matrix.

    Warning
    -------
    The output domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ_w* colourspace matrix is in domain [0, 100].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> specification = colour.CIECAM02_Specification(J=41.731091132513917,
                                                      C=0.1047077571711053,
                                                      h=-140.9515673417281,
                                                      Q=195.37132596607671,
                                                      M=0.1088421756692261,
                                                      s=2.3603053739204447,
                                                      H=278.06073585662813)
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> colour.CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b)
    array([ 19.01,  20.  ,  21.78])
    """

    XYZ_w = np.array(XYZ_w).reshape((3, 1))
    X_w, Y_w, Zw = np.ravel(XYZ_w)

    n, F_L, N_bb, N_cb, z = get_viewing_condition_dependent_parameters(Y_b,
                                                                       Y_w,
                                                                       L_A)

    J, C, h, Q, M, s, H = CIECAM02_Specification


    # Converting *CIE XYZ* colourspace matrices to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB_w = np.dot(CAT02_CAT, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = get_degree_of_adaptation(surround.F,
                                 L_A) if not discount_illuminant else 1.

    # Computation full chromatic adaptation.
    RGB_wc = get_full_chromatic_adaptation_forward(RGB_w, RGB_w, Y_w, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_pw = RGB_to_rgb(RGB_wc)

    # Applying post-adaptation non linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_pw, F_L)

    # Computing achromatic responses for the stimulus and the whitepoint.
    A_w = get_achromatic_response_forward(RGB_aw, N_bb)

    # Computing temporary magnitude quantity :math:`t`.
    t = get_temporary_magnitude_quantity_reverse(C, J, n)

    # Computing eccentricity factor *e_t*.
    e_t = get_eccentricity_factor(h)

    # Computing achromatic response :math:`A` for the stimulus.
    A = get_achromatic_response_reverse(A_w, J, surround.c, z)

    # Computing *P_1* to *P_3*.
    P_1, P_2, P_3 = get_P(surround.N_c, N_cb, e_t, t, A, N_bb)

    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    a, b = get_opponent_colour_dimensions_reverse((P_1, P_2, P_3), h)

    # Computing post-adaptation non linear response compression matrix.
    RGB_a = get_post_adaptation_non_linear_response_compression_matrix(P_2, a,
                                                                       b)

    # Applying reverse post-adaptation non linear response compression.
    RGB_p = post_adaptation_non_linear_response_compression_reverse(RGB_a,
                                                                    F_L)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_c = rgb_to_RGB(RGB_p)

    # Applying reverse full chromatic adaptation.
    RGB = get_full_chromatic_adaptation_reverse(RGB_c, RGB_w, Y_w, D)

    # Converting *CMCCAT2000* transform sharpened *RGB* values to *CIE XYZ*
    # colourspace matrices.
    XYZ = np.dot(CAT02_INVERSE_CAT, RGB)

    return XYZ


def get_chromatic_induction_factors(n):
    """
    Returns the chromatic induction factors :math:`N_{bb}` and :math:`N_{cb}`.

    Parameters
    ----------
    n : float
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    tuple
        Chromatic induction factors :math:`N_{bb}` and :math:`N_{cb}`.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_chromatic_induction_factors(0.2)
    (1.0003040045593807, 1.0003040045593807)
    """

    N_bb = N_cb = 0.725 * (1. / n) ** 0.2
    return N_bb, N_cb


def get_base_exponential_non_linearity(n):
    """
    Returns the base exponential non linearity :math:`n`.

    Parameters
    ----------
    n : float
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    float
        Base exponential non linearity :math:`z`.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_base_exponential_non_linearity(0.2)
    1.9272135954999579
    """

    z = 1.48 + math.sqrt(n)
    return z


@memoize(_CIECAM02_VIEWING_CONDITION_DEPENDENT_PARAMETERS_CACHE)
def get_viewing_condition_dependent_parameters(Y_b, Y_w, L_A):
    """
    Returns the viewing condition dependent parameters.

    Parameters
    ----------
    Y_b : float
        Adapting field *Y* tristimulus value :math:`Y_b`.
    Y_w : float
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    L_A : float
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    tuple
        Viewing condition dependent parameters.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_viewing_condition_dependent_parameters(20.0, 100.0, 318.31)
    (0.20000000000000001, 1.16754446414718, 1.0003040045593807, 1.0003040045593807, 1.9272135954999579)
    """

    n = Y_b / Y_w

    F_L = get_luminance_level_adaptation_factor(L_A)
    N_bb, N_cb = get_chromatic_induction_factors(n)
    z = get_base_exponential_non_linearity(n)

    return n, F_L, N_bb, N_cb, z


def get_degree_of_adaptation(F, L_A):
    """
    Returns the degree of adaptation :math:`D` from given surround maximum
    degree of adaptation :math:`F` and Adapting field *luminance* :math:`L_A`
    in :math:`cd/m^2`.

    Parameters
    ----------
    F : float
        Surround maximum degree of adaptation :math:`F`.
    L_A : float
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    float
        Degree of adaptation :math:`D`.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_degree_of_adaptation(1.0, 318.31)
    0.99446878008843742
    """

    D = F * (1. - (1. / 3.6) * np.exp((-L_A - 42.) / 92.))
    return D


def get_full_chromatic_adaptation_forward(RGB, RGB_w, Y_w, D):
    """
    Applies full chromatic adaptation to given *CMCCAT2000* transform sharpened
    *RGB* matrix using given *CMCCAT2000* transform sharpened whitepoint *RGB_w*
    matrix.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* matrix.
    RGB_w : array_like
        *CMCCAT2000* transform sharpened whitepoint *RGB_w* matrix.
    Y_w : float
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    D : float
        Degree of adaptation :math:`D`.

    Returns
    -------
    ndarray, (3,)
        Adapted *RGB* matrix.

    Examples
    --------
    >>> RGB = np.array([18.985456, 20.707422, 21.747482])
    >>> RGB_w = np.array([94.930528, 103.536988, 108.717742])
    >>> Y_w = 100.0
    >>> D = 0.994468780088
    >>> colour.appearance.ciecam02.get_full_chromatic_adaptation_forward(RGB, RGB_w, Y_w, D)
    array([ 19.99370783,  20.00393634,  20.01326387])
    """

    R, G, B = np.ravel(RGB)
    R_w, G_w, B_w = np.ravel(RGB_w)

    equation = lambda x, y: ((Y_w * D / y) + 1 - D) * x

    R_c = equation(R, R_w)
    G_c = equation(G, G_w)
    B_c = equation(B, B_w)

    return np.array([R_c, G_c, B_c])


def get_full_chromatic_adaptation_reverse(RGB, RGB_w, Y_w, D):
    """
    Reverts full chromatic adaptation of given *CMCCAT2000* transform sharpened
    *RGB* matrix using given *CMCCAT2000* transform sharpened whitepoint *RGB_w*
    matrix.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* matrix.
    RGB_w : array_like
        *CMCCAT2000* transform sharpened whitepoint *RGB_w* matrix.
    Y_w : float
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    D : float
        Degree of adaptation :math:`D`.

    Returns
    -------
    ndarray, (3,)
        Adapted *RGB* matrix.

    Examples
    --------
    >>> RGB = np.array([19.99370783, 20.00393634, 20.01326387])
    >>> RGB_w = np.array([94.930528, 103.536988, 108.717742])
    >>> Y_w = 100.0
    >>> D = 0.994468780088
    >>> colour.appearance.ciecam02.get_full_chromatic_adaptation_reverse(RGB, RGB_w, Y_w, D)
    array([ 18.985456,  20.707422,  21.747482])
    """

    R, G, B = np.ravel(RGB)
    R_w, G_w, B_w = np.ravel(RGB_w)

    equation = lambda x, y: x / (Y_w * (D / y) + 1. - D)

    R_c = equation(R, R_w)
    G_c = equation(G, G_w)
    B_c = equation(B, B_w)

    return np.array([R_c, G_c, B_c])


def RGB_to_rgb(RGB):
    """
    Converts given *RGB* matrix to *Hunt-Pointer-Estevez*
    :math:`\rho\gamma\beta` colourspace.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* matrix.

    Returns
    -------
    ndarray, (3,)
        *Hunt-Pointer-Estevez* :math:`\rho\gamma\beta` colourspace matrix.

    Examples
    --------
    >>> RGB = np.array([19.99370783, 20.00393634, 20.01326387])
    >>> colour.appearance.ciecam02.RGB_to_rgb(RGB)
    array([ 19.99693975,  20.00186123,  20.0135053 ])
    """

    rgb = np.dot(np.dot(HPE_MATRIX, CAT02_INVERSE_CAT), RGB)
    return rgb


def rgb_to_RGB(rgb):
    """
    Converts given *Hunt-Pointer-Estevez* :math:`\rho\gamma\beta` colourspace
    matrix to *RGB* matrix.

    Parameters
    ----------
    rgb : array_like, (3,)
        *Hunt-Pointer-Estevez* :math:`\rho\gamma\beta` colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *RGB* matrix.

    Examples
    --------
    >>> rgb = np.array([19.99693975, 20.00186123, 20.0135053])
    >>> colour.appearance.ciecam02.rgb_to_RGB(rgb)
    array([ 19.99370783,  20.00393634,  20.01326387])
    """

    RGB = np.dot(np.dot(CAT02_CAT, HPE_MATRIX_INVERSE), rgb)
    return RGB


def post_adaptation_non_linear_response_compression_forward(RGB, F_L):
    """
    Returns given *CMCCAT2000* transform sharpened *RGB* matrix with post
    adaptation non linear response compression.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* matrix.

    Returns
    -------
    ndarray, (3,)
        Compressed *CMCCAT2000* transform sharpened *RGB* matrix.

    Examples
    --------
    >>> RGB = np.array([19.99693975, 20.00186123, 20.0135053])
    >>> F_L = 1.16754446415
    >>> colour.appearance.ciecam02.post_adaptation_non_linear_response_compression_forward(RGB, F_L)
    array([ 7.9463202 ,  7.94711528,  7.94899595])
    """

    # TODO: Check for negative values and their handling.
    RGB_c = ((((400. * (F_L * RGB / 100) ** 0.42) /
               (27.13 + (F_L * RGB / 100) ** 0.42))) + 0.1)
    return RGB_c


def post_adaptation_non_linear_response_compression_reverse(RGB, F_L):
    """
    Returns given *CMCCAT2000* transform sharpened *RGB* matrix without post
    adaptation non linear response compression.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* matrix.

    Returns
    -------
    ndarray, (3,)
        Uncompressed *CMCCAT2000* transform sharpened *RGB* matrix.

    Examples
    --------
    >>> RGB = np.array([7.9463202, 7.94711528, 7.94899595])
    >>> F_L = 1.16754446415
    >>> colour.appearance.ciecam02.post_adaptation_non_linear_response_compression_reverse(RGB, F_L)
    array([ 19.99693978,  20.00186124,  20.01350528])
    """

    RGB_p = ((np.sign(RGB - 0.1) *
              (100. / F_L) * ((27.13 * np.abs(RGB - 0.1)) /
                              (400 - np.abs(RGB - 0.1))) ** (1 / 0.42)))
    return RGB_p


def get_opponent_colour_dimensions_forward(RGB):
    """
    Returns opponent colour dimensions from given compressed *CMCCAT2000*
    transform sharpened *RGB* matrix for forward *CIECAM02* implementation

    Parameters
    ----------
    RGB : array_like
        Compressed *CMCCAT2000* transform sharpened *RGB* matrix.

    Returns
    -------
    tuple
        Opponent colour dimensions.

    Examples
    --------
    >>> RGB = np.array([7.9463202, 7.94711528,7.94899595])
    >>> colour.appearance.ciecam02.get_opponent_colour_dimensions_forward(RGB)
    (-0.00062411000000173189, -0.00050626888888870443)
    """

    R, G, B = np.ravel(RGB)

    a = R - 12. * G / 11. + B / 11.
    b = (R + G - 2. * B) / 9.

    return a, b


def get_opponent_colour_dimensions_reverse(P, h):
    """
    Returns opponent colour dimensions from given points :math:`P` and hue :math:`h`
    in degrees for reverse *CIECAM02* implementation.

    Parameters
    ----------
    p : array_like
        Points :math:`P`.
    h : float
        Hue :math:`h` in degrees.

    Returns
    -------
    tuple
        Opponent colour dimensions.

    Examples
    --------
    >>> p = (30162.890815335879, 24.237205467134817, 1.05)
    >>> h = -140.9515673417281
    >>> colour.appearance.ciecam02.get_opponent_colour_dimensions_reverse(p, h)
    (-0.0006241120682426434, -0.0005062701067729668)
    """

    P_1, P_2, P_3 = P
    hr = math.radians(h)

    sin_hr, cos_hr = math.sin(hr), math.cos(hr)
    P_4 = P_1 / sin_hr
    P_5 = P_1 / cos_hr
    n = P_2 * (2. + P_3) * (460. / 1403.)

    if abs(sin_hr) >= abs(cos_hr):
        b = n / (P_4 + (2. + P_3) * (220. / 1403.) * (cos_hr / sin_hr) - (
            27. / 1403.) + P_3 * (6300. / 1403))
        a = b * (cos_hr / sin_hr)
    else:
        a = n / (P_5 + (2. + P_3) * (220. / 1403.) - (
            (27. / 1403.) - P_3 * (6300. / 1403.)) * (sin_hr / cos_hr))
        b = a * (sin_hr / cos_hr)

    return a, b


def get_hue_angle(a, b):
    """
    Returns the *hue* angle :math:`h` in degrees.

    Parameters
    ----------
    a : float
        Opponent colour dimension :math:`a`.
    b : float
        Opponent colour dimension :math:`b`.

    Returns
    -------
    float
        *Hue* angle :math:`h` in degrees.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_hue_correlate(-0.0006241120682426434, -0.0005062701067729668)
    219.0484326582719
    """

    h = math.degrees(np.arctan2(b, a)) % 360
    return h


def get_hue_quadrature(h):
    """
    Returns the hue quadrature from given hue :math:`h` angle in degrees.

    Parameters
    ----------
    h : float
        Hue :math:`h` angle in degrees.

    Returns
    -------
    float
        Hue quadrature.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_hue_quadrature(-140.951567342)
    278.06073585629122
    """

    h_i = HUE_DATA_FOR_HUE_QUADRATURE.get('h_i')
    e_i = HUE_DATA_FOR_HUE_QUADRATURE.get('e_i')
    H_i = HUE_DATA_FOR_HUE_QUADRATURE.get('H_i')

    h_p = h + 360 if h < h_i[0] else h
    index = bisect.bisect_left(h_i, h_p) - 1

    H = (H_i[index] + ((100 * (h_p - h_i[index]) / e_i[index]) /
                       ((h_p - h_i[index]) / e_i[index] +
                        (h_i[index + 1] - h_p) / e_i[index + 1])))

    return H


def get_eccentricity_factor(h):
    """
    Returns the eccentricity factor :math:`e_t` from given hue :math:`h` angle
    for forward *CIECAM02* implementation.

    Parameters
    ----------
    h : float
        Hue :math:`h` angle in degrees.

    Returns
    -------
    float
        Eccentricity factor :math:`e_t`.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_eccentricity_factor(-140.951567342)
    1.1740054728513878
    """

    e_t = 1. / 4. * (math.cos(2. + h * math.pi / 180.) + 3.8)
    return e_t


def get_achromatic_response_forward(RGB, N_bb):
    """
    Returns the achromatic response :math:`A` from given compressed
    *CMCCAT2000* transform sharpened *RGB* matrix and :math:`N_{bb}` chromatic
    induction factor for forward *CIECAM02* implementation.

    Parameters
    ----------
    RGB : array_like
        Compressed *CMCCAT2000* transform sharpened *RGB* matrix.
    N_bb : float
        Chromatic induction factor :math:`N_{bb}`.

    Returns
    -------
    float
        Achromatic response :math:`A`.

    Examples
    --------
    >>> RGB = np.array([7.9463202, 7.94711528,7.94899595])
    >>> N_bb = 1.0003040045593807
    >>> colour.appearance.ciecam02.get_achromatic_response_forward(RGB, N_bb)
    23.939480977081196
    """

    R, G, B = np.ravel(RGB)

    A = (2. * R + G + (1. / 20.) * B - 0.305) * N_bb
    return A


def get_achromatic_response_reverse(A_w, J, c, z):
    """
    Returns the achromatic response :math:`A` from given achromatic response
    :math:`A_w` for the whitepoint, *Lightness* correlate :math:`J`, surround
    exponential non linearity :math:`c` and base exponential non linearity
    :math:`z` for reverse *CIECAM02* implementation.

    Parameters
    ----------
    A_w : float
        Achromatic response :math:`A_w` for the whitepoint.
    J : float
        *Lightness* correlate :math:`J`.
    c : float
        Surround exponential non linearity :math:`c`.
    z : float
        Base exponential non linearity :math:`z`.

    Returns
    -------
    float
        Achromatic response :math:`A`.

    Examples
    --------
    >>> A_w = 46.1882087914
    >>> J = 41.73109113251392
    >>> c = 0.69
    >>> z = 1.9272135954999579
    >>> colour.appearance.ciecam02.get_achromatic_response_reverse(A_w, J, c, z)
    23.93948096673739
    """

    A = A_w * (J / 100.) ** (1. / (c * z))
    return A


def get_lightness_correlate(A, A_w, c, z):
    """
    Returns the *Lightness* correlate :math:`J`.

    Parameters
    ----------
    A : float
        Achromatic response :math:`A` for the stimulus.
    A_w : float
        Achromatic response :math:`A_w` for the whitepoint.
    c : float
        Surround exponential non linearity :math:`c`.
    z : float
        Base exponential non linearity :math:`z`.

    Returns
    -------
    float
        *Lightness* correlate :math:`J`.

    Examples
    --------
    >>> A = 23.9394809667
    >>> A_w = 46.1882087914
    >>> c = 0.69
    >>> z = 1.9272135955
    >>> colour.appearance.ciecam02.get_lightness_correlate(A, A_w, c, z)
    41.73109113242645
    """

    J = 100. * (A / A_w) ** (c * z)
    return J


def get_brightness_correlate(c, J, A_w, F_L):
    """
    Returns the *brightness* correlate :math:`Q`.

    Parameters
    ----------
    c : float
        Surround exponential non linearity :math:`c`.
    J : float
        *Lightness* correlate :math:`J`.
    A_w : float
        Achromatic response :math:`A_w` for the whitepoint.
    F_L : float
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    float
        *Brightness* correlate :math:`Q`.

    Examples
    --------
    >>> c = 0.69
    >>> J = 41.7310911325
    >>> A_w = 46.1882087914
    >>> F_L = 1.16754446415
    >>> colour.appearance.ciecam02.get_brightness_correlate(c, J, A_w, F_L)
    195.37132596634626
    """

    Q = (4. / c) * math.sqrt(J / 100.) * (A_w + 4) * F_L ** 0.25
    return Q


def get_temporary_magnitude_quantity_forward(N_c, N_cb, e_t, a, b, RGB_a):
    """
    Returns the temporary magnitude quantity :math:`t`. for forward *CIECAM02*
    implementation.

    Parameters
    ----------
    N_c : float
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb : float
        Chromatic induction factor :math:`N_{cb}`.
    e_t : float
        Eccentricity factor :math:`e_t`.
    a : float
        Opponent colour dimension :math:`a`.
    b : float
        Opponent colour dimension :math:`b`.
    RGB_a : array_like
        Compressed stimulus *CMCCAT2000* transform sharpened *RGB* matrix.

    Returns
    -------
    float
         Temporary magnitude quantity :math:`t`.

    Examples
    --------
    >>> N_c = 1.0
    >>> N_cb = 1.00030400456
    >>> e_t = 1.1740054728519145
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> RGB_a = np.array([7.9463202, 7.94711528, 7.94899595])
    >>> colour.appearance.ciecam02.get_temporary_magnitude_quantity_forward(N_c, N_cb, e_t, a, b, RGB_a)
    0.14974620289879878
    """

    Ra, Ga, Ba = np.ravel(RGB_a)
    t = ((50000. / 13.) * N_c * N_cb) * (e_t * (a ** 2 + b ** 2) ** 0.5) / (
        Ra + Ga + 21. * Ba / 20.)
    return t


def get_temporary_magnitude_quantity_reverse(C, J, n):
    """
    Returns the temporary magnitude quantity :math:`t`. for reverse *CIECAM02*
    implementation.

    Parameters
    ----------
    C : float
        *Chroma* correlate :math:`C`.
    J : float
        *Lightness* correlate :math:`J`.
    n : float
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    float
         Temporary magnitude quantity :math:`t`.

    Examples
    --------
    >>> C = 0.1047077571711053
    >>> J = 41.73109113251392
    >>> n = 0.2
    >>> colour.appearance.ciecam02.get_temporary_magnitude_quantity_reverse(C, J, n)
    0.14974620292124402
    """

    t = (C / (math.sqrt(J / 100.) * (1.64 - 0.29 ** n) ** 0.73)) ** (1. / 0.9)
    return t


def get_chroma_correlate(J, n, N_c, N_cb, e_t, a, b, RGB_a):
    """
    Returns the *chroma* correlate :math:`C`.

    Parameters
    ----------
    J : float
        *Lightness* correlate :math:`J`.
    n : float
        Function of the luminance factor of the background :math:`n`.
    N_c : float
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb : float
        Chromatic induction factor :math:`N_{cb}`.
    e_t : float
        Eccentricity factor :math:`e_t`.
    a : float
        Opponent colour dimension :math:`a`.
    b : float
        Opponent colour dimension :math:`b`.
    RGB_a : array_like
        Compressed stimulus *CMCCAT2000* transform sharpened *RGB* matrix.

    Returns
    -------
    float
        *Chroma* correlate :math:`C`.

    Examples
    --------
    >>> J = 41.7310911325
    >>> n = 0.2
    >>> N_c = 1.0
    >>> N_cb = 1.00030400456
    >>> e_t = 1.17400547285
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> RGB_a = np.array([7.9463202, 7.94711528,7.94899595])
    >>> colour.appearance.ciecam02.get_chroma_correlate(J, n, N_c, N_cb, e_t, a, b, RGB_a)
    0.10470775715680908
    """

    t = get_temporary_magnitude_quantity_forward(N_c, N_cb, e_t, a, b, RGB_a)
    C = t ** 0.9 * (J / 100.) ** 0.5 * (1.64 - 0.29 ** n) ** 0.73

    return C


def get_colourfulness_correlate(C, F_L):
    """
    Returns the *colourfulness* correlate :math:`M`.

    Parameters
    ----------
    C : float
        *Chroma* correlate :math:`C`.
    F_L : float
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    float
        *Colourfulness* correlate :math:`M`.

    Examples
    --------
    >>> C = 0.104707757171
    >>> F_L = 1.16754446415
    >>> colour.appearance.ciecam02.get_colourfulness_correlate(C, F_L)
    0.10884217566918239
    """

    M = C * F_L ** 0.25
    return M


def get_saturation_correlate(M, Q):
    """
    Returns the *saturation* correlate :math:`s`.

    Parameters
    ----------
    M : float
        *Colourfulness* correlate :math:`M`.
    Q : float
        *Brightness* correlate :math:`C`.

    Returns
    -------
    float
        *Saturation* correlate :math:`s`.

    Examples
    --------
    >>> M = 0.108842175669
    >>> Q = 195.371325966
    >>> colour.appearance.ciecam02.get_saturation_correlate()
    2.3603053739184565
    """

    s = 100. * (M / Q) ** 0.5
    return s


def get_P(N_c, N_cb, e_t, t, A, N_bb):
    """
    Returns the points :math:`P_1`, :math:`P_2` and :math:`P_3`.

    Parameters
    ----------
    N_c : float
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb : float
        Chromatic induction factor :math:`N_{cb}`.
    e_t : float
        Eccentricity factor :math:`e_t`.
    t : float
        Temporary magnitude quantity :math:`t`.
    A : float
        Achromatic response  :math:`A` for the stimulus.
    N_bb : float
        Chromatic induction factor :math:`N_{bb}`.

    Returns
    -------
    tuple
        Points :math:`P`.

    Examples
    --------
    >>> N_c = 1.0
    >>> N_cb = 1.00030400456
    >>> e_t = 1.1740054728519145
    >>> t = 0.149746202921
    >>> A = 23.9394809667
    >>> N_bb = 1.00030400456
    >>> colour.appearance.ciecam02.get_P(N_c, N_cb, e_t, t, A, N_bb)
    (30162.8908154037, 24.23720546710714, 1.05)
    """

    P_1 = ((50000. / 13.) * N_c * N_cb * e_t) / t
    P_2 = A / N_bb + 0.305
    P_3 = 21. / 20.

    return P_1, P_2, P_3


def get_post_adaptation_non_linear_response_compression_matrix(P_2, a, b):
    """
    Returns the post adaptation non linear response compression matrix.

    Parameters
    ----------
    P_2 : float
        Point :math:`P_2`.
    a : float
        Opponent colour dimension :math:`a`.
    b : float
        Opponent colour dimension :math:`b`.

    Returns
    -------
    ndarray, (3,)
        Points :math:`P`.

    Examples
    --------
    >>> P_2 = 24.2372054671
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> colour.appearance.ciecam02.get_post_adaptation_non_linear_response_compression_matrix(P_2, a, b)
    array([ 7.9463202 ,  7.94711528,  7.94899595])
    """

    R_a = (460. * P_2 + 451. * a + 288. * b) / 1403.
    G_a = (460. * P_2 - 891. * a - 261. * b) / 1403.
    B_a = (460. * P_2 - 220. * a - 6300. * b) / 1403.

    return np.array([R_a, G_a, B_a])