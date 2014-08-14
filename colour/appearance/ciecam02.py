#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIECAM02
========

Defines *CIECAM02* colour appearance model objects:

-   :func:`XYZ_to_CIECAM02`
-   :func:`CIECAM02_to_XYZ`

References
----------
.. [1]  http://en.wikipedia.org/wiki/CIECAM02
        (Last accessed 14 August 2014)
.. [2]  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published 19 November 2004, ISBN-13: 978-0470012161,
        Pages 265-277.
.. [3]  **Stephen Westland, Caterina Ripamonti, Vien Cheung**,
        *Computational Colour Science Using MATLAB, 2nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published July 2012, ISBN-13: 978-0-470-66569-5, Page 38.
.. [4]  `The CIECAM02 Color Appearance Model
        <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_
        (Last accessed 30 July 2014)
"""

from __future__ import unicode_literals

import bisect
import math
from collections import namedtuple

import numpy as np

import colour.adaptation.cat
import colour.utilities.decorators

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CIECAM02_SURROUND_FCNC',
           'CIECAM02_VIEWING_CONDITIONS',
           'HPE',
           'CAT02_CAT_INVERSE',
           'HUE_DATA_FOR_HUE_QUADRATURE',
           'CIECAM02_Specification',
           'XYZ_to_CIECAM02',
           'CIECAM02_to_XYZ',
           'get_luminance_level_adaptation_factor',
           'get_chromatic_induction_factors',
           'get_base_exponential_non_linearity',
           'get_viewing_condition_dependent_parameters',
           'get_degree_of_adaptation',
           'forward_full_chromatic_adaptation',
           'RGB_to_HPE',
           'post_adaptation_non_linear_response_compression_forward',
           'get_opponent_colour_dimensions_forward',
           'get_hue_quadrature',
           'get_eccentricity_factor',
           'get_achromatic_response_forward',
           'get_lightness_correlate',
           'get_brightness_correlate',
           'get_chroma_correlate',
           'get_colourfulness_correlate',
           'get_saturation_correlate']

CIECAM02_SURROUND_FCNC = namedtuple('CIECAM02_surround', ('F', 'c', 'Nc'))

CIECAM02_VIEWING_CONDITIONS = {
    'Average': CIECAM02_SURROUND_FCNC(1., 0.69, 1.),
    'Dim': CIECAM02_SURROUND_FCNC(0.9, 0.59, 0.95),
    'Dark': CIECAM02_SURROUND_FCNC(0.8, 0.525, 0.8)}
"""
*CIECAM02* viewing conditions.

CIECAM02_VIEWING_CONDITIONS : dict
('Average', 'Dim', 'Dark')
"""

HPE = np.array([[0.38971, 0.68898, -0.07868],
                [-0.22981, 1.18340, 0.04641],
                [0.00000, 0.00000, 1.00000]])

HPE_INVERSE = np.linalg.inv(HPE)

CAT02_CAT_INVERSE = np.linalg.inv(colour.adaptation.cat.CAT02_CAT)

HUE_DATA_FOR_HUE_QUADRATURE = {
    'hi': np.array([20.14, 90.00, 164.25, 237.53, 380.14]),
    'ei': np.array([0.8, 0.7, 1.0, 1.2, 0.8]),
    'Hi': np.array([0.0, 100.0, 200.0, 300.0, 400.0])}

CIECAM02_Specification = namedtuple('CIECAM02_Specification',
                                    ('J', 'C', 'h', 'Q', 'M', 's', 'H'))
"""
Defines a *CIECAM02* specification.

Parameters
----------
J : float
    Correlate of *Lightness* :math:`J`.
C : float
    Correlate of *chroma* :math:`C`.
h : float
    Hue :math:`h` in degrees.
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
                    XYZw,
                    LA,
                    Yb,
                    surround=CIECAM02_VIEWING_CONDITIONS.get(
                        'Average'),
                    discount_illuminant=False):
    """
    Computes the *CIECAM02* colour appearance model correlates from given
    *CIE XYZ* colourspace matrix.

    This is the *forward* implementation.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* colourspace stimulus matrix.
    XYZw : array_like
        *CIE XYZ* colourspace whitepoint matrix.
    LA : float
        Adapting field *luminance* :math:`L_A` in cd/m2.
    Yb : float
        Adapting field *Y* tristimulus value :math:`Y_b`.
    surround : CIECAM02_Surround
        Surround viewing conditions.
    discount_illuminant : bool
        Discount the illuminant.

    Returns
    -------
    CIECAM02_Specification
        *CIECAM02* specification.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    -   Input *CIE XYZw* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZw = np.array([95.05, 100.00, 108.88])
    >>> LA = 318.31
    >>> Yb = 20.0
    >>> colour.XYZ_to_CIECAM02(XYZ, XYZw, LA, Yb)
    CIECAM02_Specification(J=41.731091132513917, C=0.1047077571711053, h=-140.9515673417281, Q=195.37132596607671, M=0.1088421756692261, s=2.3603053739204447, H=278.06073585662813)
    """

    XYZ = np.array(XYZ).reshape((3, 1))
    XYZw = np.array(XYZw).reshape((3, 1))
    X, Y, Z = np.ravel(XYZ)
    Xw, Yw, Zw = np.ravel(XYZw)

    n, FL, Nbb, Ncb, z = get_viewing_condition_dependent_parameters(Yb, Yw, LA)

    # Converting *CIE XYZ* colourspace matrices to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB = np.dot(colour.adaptation.cat.CAT02_CAT, XYZ)
    RGBw = np.dot(colour.adaptation.cat.CAT02_CAT, XYZw)

    # Computing degree of adaptation :math:`D`.
    D = get_degree_of_adaptation(surround.F,
                                 LA) if not discount_illuminant else 1.

    # Computing full chromatic adaptation.
    RGBc = forward_full_chromatic_adaptation(RGB, RGBw, Yw, D)
    RGBwc = forward_full_chromatic_adaptation(RGBw, RGBw, Yw, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGBp = RGB_to_HPE(RGBc)
    RGBpw = RGB_to_HPE(RGBwc)

    # Applying forward post-adaptation non linear response compression.
    RGBa = post_adaptation_non_linear_response_compression_forward(
        RGBp, FL)
    RGBaw = post_adaptation_non_linear_response_compression_forward(
        RGBpw, FL)

    # Converting to preliminary cartesian coordinates.
    a, b = get_opponent_colour_dimensions_forward(RGBa)

    # Computing hue :math:`h`.
    h = math.degrees(math.atan2(b, a))

    # Computing hue :math:`h` quadrature :math:`H`.
    H = get_hue_quadrature(h)

    # Computing eccentricity factor *et*.
    et = get_eccentricity_factor(h)

    # Computing achromatic responses for the stimulus and the whitepoint.
    A = get_achromatic_response_forward(RGBa, Nbb)
    Aw = get_achromatic_response_forward(RGBaw, Nbb)

    # Computing the correlate of *Lightness* :math:`J`.
    J = get_lightness_correlate(A, Aw, surround.c, z)

    # Computing the correlate of *brightness* :math:`Q`.
    Q = get_brightness_correlate(surround.c, J, Aw, FL)

    # Computing the correlate of *chroma* :math:`C`.
    C = get_chroma_correlate(J, n, surround.Nc, Ncb, et, a, b, RGBa)

    # Computing the correlate of *colourfulness* :math:`M`.
    M = get_colourfulness_correlate(C, FL)

    # Computing the correlate of *saturation* :math:`s`.
    s = get_saturation_correlate(M, Q)

    return CIECAM02_Specification(J, C, h, Q, M, s, H)


def CIECAM02_to_XYZ(CIECAM02_Specification,
                    XYZw,
                    LA,
                    Yb,
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
    XYZw : array_like
        *CIE XYZ* colourspace whitepoint matrix.
    LA : float
        Adapting field *luminance* :math:`L_A` in cd/m2.
    Yb : float
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
    -   Input *CIE XYZw* colourspace matrix is in domain [0, 100].
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
    >>> XYZw = np.array([95.05, 100.00, 108.88])
    >>> LA = 318.31
    >>> Yb = 20.0
    >>> colour.CIECAM02_to_XYZ(specification, XYZw, LA, Yb)
    array([ 19.01,  20.  ,  21.78])
    """

    XYZw = np.array(XYZw).reshape((3, 1))
    Xw, Yw, Zw = np.ravel(XYZw)

    n, FL, Nbb, Ncb, z = get_viewing_condition_dependent_parameters(Yb, Yw, LA)

    J, C, h, Q, M, s, H = CIECAM02_Specification


    # Converting *CIE XYZ* colourspace matrices to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGBw = np.dot(colour.adaptation.cat.CAT02_CAT, XYZw)

    # Computing degree of adaptation :math:`D`.
    D = get_degree_of_adaptation(surround.F,
                                 LA) if not discount_illuminant else 1.

    # Computation full chromatic adaptation.
    RGBwc = forward_full_chromatic_adaptation(RGBw, RGBw, Yw, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGBpw = RGB_to_HPE(RGBwc)

    # Applying post-adaptation non linear response compression.
    RGBaw = post_adaptation_non_linear_response_compression_forward(
        RGBpw, FL)

    # Computing achromatic responses for the stimulus and the whitepoint.
    Aw = get_achromatic_response_forward(RGBaw, Nbb)

    # Computing temporary magnitude quantity :math:`t`.
    t = get_temporary_magnitude_quantity_reverse(C, J, n)

    # Computing eccentricity factor *et*.
    et = get_eccentricity_factor(h)

    # Computing achromatic response :math:`A` for the stimulus.
    A = get_achromatic_response_reverse(Aw, J, surround.c, z)

    # Computing *P1* to *P3*.
    P1, P2, P3 = get_P(surround.Nc, Ncb, et, t, A, Nbb)

    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    a, b = get_opponent_colour_dimensions_reverse((P1, P2, P3), h)

    # Computing post-adaptation non linear response compression matrix.
    RGBa = get_post_adaptation_non_linear_response_compression_matrix(P2, a, b)

    # Applying reverse post-adaptation non linear response compression.
    RGBp = post_adaptation_non_linear_response_compression_reverse(RGBa,
                                                                   FL)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGBc = HPE_to_RGB(RGBp)

    # Applying reverse full chromatic adaptation.
    RGB = reverse_full_chromatic_adaptation(RGBc, RGBw, Yw, D)

    # Converting *CMCCAT2000* transform sharpened *RGB* values to *CIE XYZ*
    # colourspace matrices.
    XYZ = np.dot(CAT02_CAT_INVERSE, RGB)

    return XYZ


def get_luminance_level_adaptation_factor(LA):
    """
    Returns the *luminance* level adaptation factor :math:`F_L`.

    Parameters
    ----------
    LA : float
        Adapting field *luminance* :math:`L_A` in cd/m2.

    Returns
    -------
    float
        *Luminance* level adaptation factor :math:`F_L`

    Examples
    --------
    >>> colour.appearance.ciecam02.get_luminance_level_adaptation_factor(318.31)
    1.16754446414718
    """

    k = 1. / (5. * LA + 1)
    k4 = k ** 4
    FL = 0.2 * k4 * (5. * LA) + 0.1 * (1. - k4) ** 2 * (5. * LA) ** (1. / 3.)
    return FL


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

    Nbb = Ncb = 0.725 * (1. / n) ** 0.2
    return Nbb, Ncb


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


@colour.utilities.decorators.memoize(
    _CIECAM02_VIEWING_CONDITION_DEPENDENT_PARAMETERS_CACHE)
def get_viewing_condition_dependent_parameters(Yb, Yw, LA):
    """
    Returns the viewing condition dependent parameters.

    Parameters
    ----------
    Yb : float
        Adapting field *Y* tristimulus value :math:`Y_b`.
    Yw : float
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    LA : float
        Adapting field *luminance* :math:`L_A` in cd/m2.

    Returns
    -------
    tuple
        Viewing condition dependent parameters.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_viewing_condition_dependent_parameters(20.0, 100.0, 318.31)
    (0.20000000000000001, 1.16754446414718, 1.0003040045593807, 1.0003040045593807, 1.9272135954999579)
    """

    n = Yb / Yw

    FL = get_luminance_level_adaptation_factor(LA)
    Nbb, Ncb = get_chromatic_induction_factors(n)
    z = get_base_exponential_non_linearity(n)

    return n, FL, Nbb, Ncb, z


def get_degree_of_adaptation(F, LA):
    """
    Returns the degree of adaptation :math:`D` from given surround maximum
    degree of adaptation :math:`F` and Adapting field *luminance* :math:`L_A`
    in cd/m2.

    Parameters
    ----------
    F : float
        Surround maximum degree of adaptation :math:`F`.
    LA : float
        Adapting field *luminance* :math:`L_A` in cd/m2.

    Returns
    -------
    float
        Degree of adaptation :math:`D`.

    Examples
    --------
    >>> colour.appearance.ciecam02.get_degree_of_adaptation(1.0, 318.31)
    0.99446878008843742
    """

    D = F * (1. - (1. / 3.6) * np.exp((-LA - 42.) / 92.))
    return D


def forward_full_chromatic_adaptation(RGB, RGBw, Yw, D):
    """
    Applies full chromatic adaptation to given *CMCCAT2000* transform sharpened
    *RGB* matrix using given *CMCCAT2000* transform sharpened whitepoint *RGBw*
    matrix.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* matrix.
    RGBw : array_like
        *CMCCAT2000* transform sharpened whitepoint *RGBw* matrix.
    Yw : float
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
    >>> RGBw = np.array([94.930528, 103.536988, 108.717742])
    >>> Yw = 100.0
    >>> D = 0.994468780088
    >>> colour.appearance.ciecam02.forward_full_chromatic_adaptation(RGB, RGBw, Yw, D)
    array([ 19.99370783,  20.00393634,  20.01326387])
    """

    R, G, B = np.ravel(RGB)
    Rw, Gw, Bw = np.ravel(RGBw)

    equation = lambda x, y: ((Yw * D / y) + 1 - D) * x

    Rc = equation(R, Rw)
    Gc = equation(G, Gw)
    Bc = equation(B, Bw)

    return np.array([Rc, Gc, Bc])


def reverse_full_chromatic_adaptation(RGB, RGBw, Yw, D):
    """
    Reverts full chromatic adaptation of given *CMCCAT2000* transform sharpened
    *RGB* matrix using given *CMCCAT2000* transform sharpened whitepoint *RGBw*
    matrix.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* matrix.
    RGBw : array_like
        *CMCCAT2000* transform sharpened whitepoint *RGBw* matrix.
    Yw : float
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
    >>> RGBw = np.array([94.930528, 103.536988, 108.717742])
    >>> Yw = 100.0
    >>> D = 0.994468780088
    >>> colour.appearance.ciecam02.reverse_full_chromatic_adaptation(RGB, RGBw, Yw, D)
    array([ 18.985456,  20.707422,  21.747482])
    """

    R, G, B = np.ravel(RGB)
    Rw, Gw, Bw = np.ravel(RGBw)

    equation = lambda x, y: x / (Yw * (D / y) + 1. - D)

    Rc = equation(R, Rw)
    Gc = equation(G, Gw)
    Bc = equation(B, Bw)

    return np.array([Rc, Gc, Bc])


def RGB_to_HPE(RGB):
    """
    Converts given *RGB* matrix to *Hunt-Pointer-Estevez* colourspace matrix.

    Parameters
    ----------
    RGB : array_like
        *RGB* matrix.

    Returns
    -------
    ndarray, (3,)
        *Hunt-Pointer-Estevez* colourspace matrix.

    Examples
    --------
    >>> RGB = np.array([19.99370783, 20.00393634, 20.01326387])
    >>> colour.appearance.ciecam02.RGB_to_HPE(RGB)
    array([ 19.99693975,  20.00186123,  20.0135053 ])
    """

    pyb = np.dot(np.dot(HPE, CAT02_CAT_INVERSE), RGB)
    return pyb


def HPE_to_RGB(pyb):
    """
    Converts given *Hunt-Pointer-Estevez* colourspace matrix to *RGB* matrix.

    Parameters
    ----------
    pyb : array_like
        *Hunt-Pointer-Estevez* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *RGB* matrix.

    Examples
    --------
    >>> HPE = np.array([19.99693975, 20.00186123, 20.0135053])
    >>> colour.appearance.ciecam02.HPE_to_RGB(HPE)
    array([ 19.99370783,  20.00393634,  20.01326387])
    """
    RGB = np.dot(np.dot(colour.adaptation.cat.CAT02_CAT, HPE_INVERSE), pyb)
    return RGB


def post_adaptation_non_linear_response_compression_forward(RGB, FL):
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
    >>> FL = 1.16754446415
    >>> colour.appearance.ciecam02.post_adaptation_non_linear_response_compression_forward(RGB, FL)
    array([ 7.9463202 ,  7.94711528,  7.94899595])
    """

    # TODO: Check for negative values and their handling.
    RGBc = ((((400. * (FL * RGB / 100) ** 0.42) /
              (27.13 + (FL * RGB / 100) ** 0.42))) + 0.1)
    return RGBc


def post_adaptation_non_linear_response_compression_reverse(RGB, FL):
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
    >>> FL = 1.16754446415
    >>> colour.appearance.ciecam02.post_adaptation_non_linear_response_compression_reverse(RGB, FL)
    array([ 19.99693978,  20.00186124,  20.01350528])
    """

    RGBp = ((np.sign(RGB - 0.1) *
             (100. / FL) * ((27.13 * np.abs(RGB - 0.1)) /
                            (400 - np.abs(RGB - 0.1))) ** (1 / 0.42)))
    return RGBp


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

    P1, P2, P3 = P
    hr = math.radians(h)

    sin_hr, cos_hr = math.sin(hr), math.cos(hr)
    P4 = P1 / sin_hr
    P5 = P1 / cos_hr
    n = P2 * (2. + P3) * (460. / 1403.)

    if abs(sin_hr) >= abs(cos_hr):
        b = n / (P4 + (2. + P3) * (220. / 1403.) * (cos_hr / sin_hr) - (
            27. / 1403.) + P3 * (6300. / 1403))
        a = b * (cos_hr / sin_hr)
    else:
        a = n / (P5 + (2. + P3) * (220. / 1403.) - (
            (27. / 1403.) - P3 * (6300. / 1403.)) * (sin_hr / cos_hr))
        b = a * (sin_hr / cos_hr)

    return a, b


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

    hi = HUE_DATA_FOR_HUE_QUADRATURE.get('hi')
    ei = HUE_DATA_FOR_HUE_QUADRATURE.get('ei')
    Hi = HUE_DATA_FOR_HUE_QUADRATURE.get('Hi')

    hp = h + 360 if h < hi[0] else h
    index = bisect.bisect_left(hi, hp) - 1

    H = (Hi[index] + ((100 * (hp - hi[index]) / ei[index]) /
                      ((hp - hi[index]) / ei[index] +
                       (hi[index + 1] - hp) / ei[index + 1])))

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

    et = 1. / 4. * (math.cos(2. + h * math.pi / 180.) + 3.8)
    return et


def get_achromatic_response_forward(RGB, Nbb):
    """
    Returns the achromatic response :math:`A` from given compressed
    *CMCCAT2000* transform sharpened *RGB* matrix and :math:`N_{bb}` chromatic
    induction factor for forward *CIECAM02* implementation.

    Parameters
    ----------
    RGB : array_like
        Compressed *CMCCAT2000* transform sharpened *RGB* matrix.
    Nbb : float
        Chromatic induction factor :math:`N_{bb}`.

    Returns
    -------
    float
        Achromatic response :math:`A`.

    Examples
    --------
    >>> RGB = np.array([7.9463202, 7.94711528,7.94899595])
    >>> Nbb = 1.0003040045593807
    >>> colour.appearance.ciecam02.get_achromatic_response_forward(RGB, Nbb)
    23.939480977081196
    """

    R, G, B = np.ravel(RGB)

    A = (2. * R + G + (1. / 20.) * B - 0.305) * Nbb
    return A


def get_achromatic_response_reverse(Aw, J, c, z):
    """
    Returns the achromatic response :math:`A` from given achromatic response
    :math:`A_w` for the whitepoint, *Lightness* correlate :math:`J`, surround
    exponential non linearity :math:`c` and base exponential non linearity
    :math:`z` for reverse *CIECAM02* implementation.

    Parameters
    ----------
    Aw : float
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
    >>> Aw = 46.1882087914
    >>> J = 41.73109113251392
    >>> c = 0.69
    >>> z = 1.9272135954999579
    >>> colour.appearance.ciecam02.get_achromatic_response_reverse(Aw, J, c, z)
    23.93948096673739
    """

    A = Aw * (J / 100.) ** (1. / (c * z))
    return A


def get_lightness_correlate(A, Aw, c, z):
    """
    Returns the *Lightness* correlate :math:`J`.

    Parameters
    ----------
    A : float
        Achromatic response :math:`A` for the stimulus.
    Aw : float
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
    >>> Aw = 46.1882087914
    >>> c = 0.69
    >>> z = 1.9272135955
    >>> colour.appearance.ciecam02.get_lightness_correlate(A, Aw, c, z)
    41.73109113242645
    """

    J = 100. * (A / Aw) ** (c * z)
    return J


def get_brightness_correlate(c, J, Aw, FL):
    """
    Returns the *brightness* correlate :math:`Q`.

    Parameters
    ----------
    c : float
        Surround exponential non linearity :math:`c`.
    J : float
        *Lightness* correlate :math:`J`.
    Aw : float
        Achromatic response :math:`A_w` for the whitepoint.
    FL : float
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    float
        *Brightness* correlate :math:`Q`.

    Examples
    --------
    >>> c = 0.69
    >>> J = 41.7310911325
    >>> Aw = 46.1882087914
    >>> FL = 1.16754446415
    >>> colour.appearance.ciecam02.get_brightness_correlate(c, J, Aw, FL)
    195.37132596634626
    """

    Q = (4. / c) * math.sqrt(J / 100.) * (Aw + 4) * FL ** 0.25
    return Q


def get_temporary_magnitude_quantity_forward(Nc, Ncb, et, a, b, RGBa):
    """
    Returns the temporary magnitude quantity :math:`t`. for forward *CIECAM02*
    implementation.

    Parameters
    ----------
    Nc : float
        Surround chromatic induction factor :math:`N_{c}`.
    Ncb : float
        Chromatic induction factor :math:`N_{cb}`.
    et : float
        Eccentricity factor :math:`e_t`.
    a : float
        Opponent colour dimension :math:`a`.
    b : float
        Opponent colour dimension :math:`b`.
    RGBa : array_like
        Compressed stimulus *CMCCAT2000* transform sharpened *RGB* matrix.

    Returns
    -------
    float
         Temporary magnitude quantity :math:`t`.

    Examples
    --------
    >>> Nc = 1.0
    >>> Ncb = 1.00030400456
    >>> et = 1.1740054728519145
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> RGBa = np.array([7.9463202, 7.94711528, 7.94899595])
    >>> colour.appearance.ciecam02.get_temporary_magnitude_quantity_forward(Nc, Ncb, et, a, b, RGBa)
    0.14974620289879878
    """

    Ra, Ga, Ba = np.ravel(RGBa)
    t = ((50000. / 13.) * Nc * Ncb) * (et * (a ** 2 + b ** 2) ** 0.5) / (
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


def get_chroma_correlate(J, n, Nc, Ncb, et, a, b, RGBa):
    """
    Returns the *chroma* correlate :math:`C`.

    Parameters
    ----------
    J : float
        *Lightness* correlate :math:`J`.
    n : float
        Function of the luminance factor of the background :math:`n`.
    Nc : float
        Surround chromatic induction factor :math:`N_{c}`.
    Ncb : float
        Chromatic induction factor :math:`N_{cb}`.
    et : float
        Eccentricity factor :math:`e_t`.
    a : float
        Opponent colour dimension :math:`a`.
    b : float
        Opponent colour dimension :math:`b`.
    RGBa : array_like
        Compressed stimulus *CMCCAT2000* transform sharpened *RGB* matrix.

    Returns
    -------
    float
        *Chroma* correlate :math:`C`.

    Examples
    --------
    >>> J = 41.7310911325
    >>> n = 0.2
    >>> Nc = 1.0
    >>> Ncb = 1.00030400456
    >>> et = 1.17400547285
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> RGBa = np.array([7.9463202, 7.94711528,7.94899595])
    >>> colour.appearance.ciecam02.get_chroma_correlate(J, n, Nc, Ncb, et, a, b, RGBa)
    0.10470775715680908
    """

    t = get_temporary_magnitude_quantity_forward(Nc, Ncb, et, a, b, RGBa)
    C = t ** 0.9 * (J / 100.) ** 0.5 * (1.64 - 0.29 ** n) ** 0.73

    return C


def get_colourfulness_correlate(C, FL):
    """
    Returns the *colourfulness* correlate :math:`M`.

    Parameters
    ----------
    C : float
        *Chroma* correlate :math:`C`.
    FL : float
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    float
        *Colourfulness* correlate :math:`M`.

    Examples
    --------
    >>> C = 0.104707757171
    >>> FL = 1.16754446415
    >>> colour.appearance.ciecam02.get_colourfulness_correlate(C, FL)
    0.10884217566918239
    """

    M = C * FL ** 0.25
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


def get_P(Nc, Ncb, et, t, A, Nbb):
    """
    Returns the points :math:`P_1`, :math:`P_2` and :math:`P_3`.

    Parameters
    ----------
    Nc : float
        Surround chromatic induction factor :math:`N_{c}`.
    Ncb : float
        Chromatic induction factor :math:`N_{cb}`.
    et : float
        Eccentricity factor :math:`e_t`.
    t : float
        Temporary magnitude quantity :math:`t`.
    A : float
        Achromatic response  :math:`A` for the stimulus.
    Nbb : float
        Chromatic induction factor :math:`N_{bb}`.

    Returns
    -------
    tuple
        Points :math:`P`.

    Examples
    --------
    >>> Nc = 1.0
    >>> Ncb = 1.00030400456
    >>> et = 1.1740054728519145
    >>> t = 0.149746202921
    >>> A = 23.9394809667
    >>> Nbb = 1.00030400456
    >>> colour.appearance.ciecam02.get_P(Nc, Ncb, et, t, A, Nbb)
    (30162.8908154037, 24.23720546710714, 1.05)
    """

    P1 = ((50000. / 13.) * Nc * Ncb * et) / t
    P2 = A / Nbb + 0.305
    P3 = 21. / 20.

    return P1, P2, P3


def get_post_adaptation_non_linear_response_compression_matrix(P2, a, b):
    """
    Returns the post adaptation non linear response compression matrix.

    Parameters
    ----------
    P2 : float
        Point :math:`P2`.
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
    >>> P2 = 24.2372054671
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> colour.appearance.ciecam02.get_post_adaptation_non_linear_response_compression_matrix(P2, a, b)
    array([ 7.9463202 ,  7.94711528,  7.94899595])
    """

    Ra = (460. * P2 + 451. * a + 288. * b) / 1403.
    Ga = (460. * P2 - 891. * a - 261. * b) / 1403.
    Ba = (460. * P2 - 220. * a - 6300. * b) / 1403.

    return np.array([Ra, Ga, Ba])