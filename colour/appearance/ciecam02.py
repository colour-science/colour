#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIECAM02 Colour Appearance Model
================================

Defines *CIECAM02* colour appearance model objects:

-   :class:`CIECAM02_InductionFactors`
-   :attr:`CIECAM02_VIEWING_CONDITIONS`
-   :class:`CIECAM02_Specification`
-   :func:`XYZ_to_CIECAM02`
-   :func:`CIECAM02_to_XYZ`

See Also
--------
`CIECAM02 Colour Appearance Model Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/appearance/ciecam02.ipynb>`_

References
----------
.. [1]  Wikipedia. (n.d.). CIECAM02. Retrieved August 14, 2014, from
        http://en.wikipedia.org/wiki/CIECAM02
.. [2]  Fairchild, M. D. (2004). CIECAM02. In Color Appearance Models
        (2nd ed., pp. 289–301). Wiley. ISBN:978-0470012161
.. [3]  Westland, S., Ripamonti, C., & Cheung, V. (2012). Extrapolation
        Methods. Computational Colour Science Using MATLAB (2nd ed., p. 38).
        ISBN:978-0-470-66569-5
.. [4]  Moroney, N., Fairchild, M. D., Hunt, R. W. G., Li, C., Luo, M. R., &
        Newman, T. (n.d.). The CIECAM02 Color Appearance Model. Color and
        Imaging Conference, 2002(1), 23–27. Retrieved from
        http://www.ingentaconnect.com/content/ist/cic\
/2002/00002002/00000001/art00006
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.adaptation import CAT02_CAT
from colour.appearance.hunt import (
    HPE_TO_XYZ_MATRIX,
    XYZ_TO_HPE_MATRIX,
    luminance_level_adaptation_factor)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_numeric,
    dot_matrix,
    dot_vector,
    tsplit,
    tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CAT02_INVERSE_CAT',
           'CIECAM02_InductionFactors',
           'CIECAM02_VIEWING_CONDITIONS',
           'HUE_DATA_FOR_HUE_QUADRATURE',
           'CIECAM02_Specification',
           'XYZ_to_CIECAM02',
           'CIECAM02_to_XYZ',
           'chromatic_induction_factors',
           'base_exponential_non_linearity',
           'viewing_condition_dependent_parameters',
           'degree_of_adaptation',
           'full_chromatic_adaptation_forward',
           'full_chromatic_adaptation_reverse',
           'RGB_to_rgb',
           'rgb_to_RGB',
           'post_adaptation_non_linear_response_compression_forward',
           'post_adaptation_non_linear_response_compression_reverse',
           'opponent_colour_dimensions_forward',
           'opponent_colour_dimensions_reverse',
           'hue_angle',
           'hue_quadrature',
           'eccentricity_factor',
           'achromatic_response_forward',
           'achromatic_response_reverse',
           'lightness_correlate',
           'brightness_correlate',
           'temporary_magnitude_quantity_forward',
           'temporary_magnitude_quantity_reverse',
           'chroma_correlate',
           'colourfulness_correlate',
           'saturation_correlate',
           'P',
           'post_adaptation_non_linear_response_compression_matrix']

CAT02_INVERSE_CAT = np.linalg.inv(CAT02_CAT)
"""
Inverse CAT02 chromatic adaptation transform.

CAT02_INVERSE_CAT : array_like, (3, 3)
"""


class CIECAM02_InductionFactors(
    namedtuple('CIECAM02_InductionFactors',
               ('F', 'c', 'N_c'))):
    """
    *CIECAM02* colour appearance model induction factors.

    Parameters
    ----------
    F : numeric or array_like
        Maximum degree of adaptation :math:`F`.
    c : numeric or array_like
        Exponential non linearity :math:`c`.
    N_c : numeric or array_like
        Chromatic induction factor :math:`N_c`.
    """


CIECAM02_VIEWING_CONDITIONS = CaseInsensitiveMapping(
    {'Average': CIECAM02_InductionFactors(1, 0.69, 1),
     'Dim': CIECAM02_InductionFactors(0.9, 0.59, 0.95),
     'Dark': CIECAM02_InductionFactors(0.8, 0.525, 0.8)})
"""
Reference *CIECAM02* colour appearance model viewing conditions.

CIECAM02_VIEWING_CONDITIONS : CaseInsensitiveMapping
    **{'Average', 'Dim', 'Dark'}**
"""

HUE_DATA_FOR_HUE_QUADRATURE = {
    'h_i': np.array([20.14, 90.00, 164.25, 237.53, 380.14]),
    'e_i': np.array([0.8, 0.7, 1.0, 1.2, 0.8]),
    'H_i': np.array([0.0, 100.0, 200.0, 300.0, 400.0])}


class CIECAM02_Specification(
    namedtuple('CIECAM02_Specification',
               ('J', 'C', 'h', 's', 'Q', 'M', 'H', 'HC'))):
    """
    Defines the *CIECAM02* colour appearance model specification.

    Parameters
    ----------
    J : numeric or array_like
        Correlate of *Lightness* :math:`J`.
    C : numeric or array_like
        Correlate of *chroma* :math:`C`.
    h : numeric or array_like
        *Hue* angle :math:`h` in degrees.
    s : numeric or array_like
        Correlate of *saturation* :math:`s`.
    Q : numeric or array_like
        Correlate of *brightness* :math:`Q`.
    M : numeric or array_like
        Correlate of *colourfulness* :math:`M`.
    H : numeric or array_like
        *Hue* :math:`h` quadrature :math:`H`.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H^C`.
    """


def XYZ_to_CIECAM02(XYZ,
                    XYZ_w,
                    L_A,
                    Y_b,
                    surround=CIECAM02_VIEWING_CONDITIONS['Average'],
                    discount_illuminant=False):
    """
    Computes the *CIECAM02* colour appearance model correlates from given
    *CIE XYZ* tristimulus values.

    This is the *forward* implementation.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of test sample / stimulus in domain
        [0, 100].
    XYZ_w : array_like
        *CIE XYZ* tristimulus values of reference white in domain [0, 100].
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    Y_b : numeric or array_like
        Relative luminance of background :math:`Y_b` in :math:`cd/m^2`.
    surround : CIECAM02_InductionFactors, optional
        Surround viewing conditions induction factors.
    discount_illuminant : bool, optional
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
    -   Input *CIE XYZ* tristimulus values are in domain [0, 100].
    -   Input *CIE XYZ_w* tristimulus values are in domain [0, 100].

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = CIECAM02_VIEWING_CONDITIONS['Average']
    >>> XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)  # doctest: +ELLIPSIS
    CIECAM02_Specification(J=41.7310911..., C=0.1047077..., h=219.0484326..., \
s=2.3603053..., Q=195.3713259..., M=0.1088421..., H=278.0607358..., HC=None)
    """

    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = np.asarray(L_A)
    Y_b = np.asarray(Y_b)

    n, F_L, N_bb, N_cb, z = tsplit(viewing_condition_dependent_parameters(
        Y_b, Y_w, L_A))

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB = dot_vector(CAT02_CAT, XYZ)
    RGB_w = dot_vector(CAT02_CAT, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = degree_of_adaptation(surround.F, L_A) if not discount_illuminant else 1

    # Computing full chromatic adaptation.
    RGB_c = full_chromatic_adaptation_forward(
        RGB, RGB_w, Y_w, D)
    RGB_wc = full_chromatic_adaptation_forward(
        RGB_w, RGB_w, Y_w, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_p = RGB_to_rgb(RGB_c)
    RGB_pw = RGB_to_rgb(RGB_wc)

    # Applying forward post-adaptation non linear response compression.
    RGB_a = post_adaptation_non_linear_response_compression_forward(
        RGB_p, F_L)
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_pw, F_L)

    # Converting to preliminary cartesian coordinates.
    a, b = tsplit(opponent_colour_dimensions_forward(RGB_a))

    # Computing the *hue* angle :math:`h`.
    h = hue_angle(a, b)

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h)
    # TODO: Compute hue composition.

    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing achromatic responses for the stimulus and the whitepoint.
    A = achromatic_response_forward(RGB_a, N_bb)
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    # Computing the correlate of *Lightness* :math:`J`.
    J = lightness_correlate(A, A_w, surround.c, z)

    # Computing the correlate of *brightness* :math:`Q`.
    Q = brightness_correlate(surround.c, J, A_w, F_L)

    # Computing the correlate of *chroma* :math:`C`.
    C = chroma_correlate(J, n, surround.N_c, N_cb, e_t, a, b, RGB_a)

    # Computing the correlate of *colourfulness* :math:`M`.
    M = colourfulness_correlate(C, F_L)

    # Computing the correlate of *saturation* :math:`s`.
    s = saturation_correlate(M, Q)

    return CIECAM02_Specification(J, C, h, s, Q, M, H, None)


def CIECAM02_to_XYZ(J,
                    C,
                    h,
                    XYZ_w,
                    L_A,
                    Y_b,
                    surround=CIECAM02_VIEWING_CONDITIONS['Average'],
                    discount_illuminant=False):
    """
    Converts *CIECAM02* specification to *CIE XYZ* tristimulus values.

    This is the *reverse* implementation.

    Parameters
    ----------
    J : numeric or array_like
        Correlate of *Lightness* :math:`J`.
    C : numeric or array_like
        Correlate of *chroma* :math:`C`.
    h : numeric or array_like
        *Hue* angle :math:`h` in degrees.
    XYZ_w : array_like
        *CIE XYZ* tristimulus values of reference white.
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    Y_b : numeric or array_like
        Relative luminance of background :math:`Y_b` in :math:`cd/m^2`.
    surround : CIECAM02_Surround, optional
        Surround viewing conditions.
    discount_illuminant : bool, optional
        Discount the illuminant.

    Returns
    -------
    XYZ : ndarray
        *CIE XYZ* tristimulus values.

    Warning
    -------
    The output range of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ_w* tristimulus values are in domain [0, 100].
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> J = 41.731091132513917
    >>> C = 0.104707757171105
    >>> h = 219.04843265827190
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> CIECAM02_to_XYZ(J, C, h, XYZ_w, L_A, Y_b)  # doctest: +ELLIPSIS
    array([ 19.01...,  20...  ,  21.78...])
    """

    _X_w, Y_w, _Zw = tsplit(XYZ_w)

    n, F_L, N_bb, N_cb, z = tsplit(viewing_condition_dependent_parameters(
        Y_b, Y_w, L_A))

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB_w = dot_vector(CAT02_CAT, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = degree_of_adaptation(surround.F, L_A) if not discount_illuminant else 1

    # Computing full chromatic adaptation.
    RGB_wc = full_chromatic_adaptation_forward(RGB_w, RGB_w, Y_w, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_pw = RGB_to_rgb(RGB_wc)

    # Applying post-adaptation non linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_pw, F_L)

    # Computing achromatic response for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    # Computing temporary magnitude quantity :math:`t`.
    t = temporary_magnitude_quantity_reverse(C, J, n)

    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing achromatic response :math:`A` for the stimulus.
    A = achromatic_response_reverse(A_w, J, surround.c, z)

    # Computing *P_1* to *P_3*.
    P_n = P(surround.N_c, N_cb, e_t, t, A, N_bb)
    _P_1, P_2, _P_3 = tsplit(P_n)

    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    a, b = tsplit(opponent_colour_dimensions_reverse(P_n, h))

    # Computing post-adaptation non linear response compression matrix.
    RGB_a = post_adaptation_non_linear_response_compression_matrix(
        P_2, a, b)

    # Applying reverse post-adaptation non linear response compression.
    RGB_p = post_adaptation_non_linear_response_compression_reverse(
        RGB_a, F_L)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_c = rgb_to_RGB(RGB_p)

    # Applying reverse full chromatic adaptation.
    RGB = full_chromatic_adaptation_reverse(RGB_c, RGB_w, Y_w, D)

    # Converting *CMCCAT2000* transform sharpened *RGB* values to *CIE XYZ*
    # tristimulus values.
    XYZ = dot_vector(CAT02_INVERSE_CAT, RGB)

    return XYZ


def chromatic_induction_factors(n):
    """
    Returns the chromatic induction factors :math:`N_{bb}` and :math:`N_{cb}`.

    Parameters
    ----------
    n : numeric or array_like
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    ndarray
        Chromatic induction factors :math:`N_{bb}` and :math:`N_{cb}`.

    Examples
    --------
    >>> chromatic_induction_factors(0.2)  # doctest: +ELLIPSIS
    array([ 1.000304...,  1.000304...])
    """

    n = np.asarray(n)

    N_bb = N_cb = 0.725 * (1 / n) ** 0.2
    N_bbcb = tstack((N_bb, N_cb))

    return N_bbcb


def base_exponential_non_linearity(n):
    """
    Returns the base exponential non linearity :math:`n`.

    Parameters
    ----------
    n : numeric or array_like
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    numeric or ndarray
        Base exponential non linearity :math:`z`.

    Examples
    --------
    >>> base_exponential_non_linearity(0.2)  # doctest: +ELLIPSIS
    1.9272135...
    """

    n = np.asarray(n)

    z = 1.48 + np.sqrt(n)

    return z


def viewing_condition_dependent_parameters(Y_b, Y_w, L_A):
    """
    Returns the viewing condition dependent parameters.

    Parameters
    ----------
    Y_b : numeric or array_like
        Adapting field *Y* tristimulus value :math:`Y_b`.
    Y_w : numeric or array_like
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    ndarray
        Viewing condition dependent parameters.

    Examples
    --------
    >>> viewing_condition_dependent_parameters(  # doctest: +ELLIPSIS
    ...     20.0, 100.0, 318.31)
    array([ 0.2...,  1.1675444...,  1.000304...,  1.000304...,  1.9272136...])
    """

    Y_b = np.asarray(Y_b)
    Y_w = np.asarray(Y_w)

    n = Y_b / Y_w

    F_L = luminance_level_adaptation_factor(L_A)
    N_bb, N_cb = tsplit(chromatic_induction_factors(n))
    z = base_exponential_non_linearity(n)

    return tstack((n, F_L, N_bb, N_cb, z))


def degree_of_adaptation(F, L_A):
    """
    Returns the degree of adaptation :math:`D` from given surround maximum
    degree of adaptation :math:`F` and Adapting field *luminance* :math:`L_A`
    in :math:`cd/m^2`.

    Parameters
    ----------
    F : numeric or array_like
        Surround maximum degree of adaptation :math:`F`.
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    numeric or ndarray
        Degree of adaptation :math:`D`.

    Examples
    --------
    >>> degree_of_adaptation(1.0, 318.31)  # doctest: +ELLIPSIS
    0.9944687...
    """

    F = np.asarray(F)
    L_A = np.asarray(L_A)

    D = F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92))

    return D


def full_chromatic_adaptation_forward(RGB, RGB_w, Y_w, D):
    """
    Applies full chromatic adaptation to given *CMCCAT2000* transform sharpened
    *RGB* array using given *CMCCAT2000* transform sharpened whitepoint
    *RGB_w* array.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* array.
    RGB_w : array_like
        *CMCCAT2000* transform sharpened whitepoint *RGB_w* array.
    Y_w : numeric or array_like
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    D : numeric or array_like
        Degree of adaptation :math:`D`.

    Returns
    -------
    ndarray
        Adapted *RGB* array.

    Examples
    --------
    >>> RGB = np.array([18.985456, 20.707422, 21.747482])
    >>> RGB_w = np.array([94.930528, 103.536988, 108.717742])
    >>> Y_w = 100.0
    >>> D = 0.994468780088
    >>> full_chromatic_adaptation_forward(  # doctest: +ELLIPSIS
    ...     RGB, RGB_w, Y_w, D)
    array([ 19.9937078...,  20.0039363...,  20.0132638...])
    """

    RGB = np.asarray(RGB)
    RGB_w = np.asarray(RGB_w)
    Y_w = np.asarray(Y_w)
    D = np.asarray(D)

    RGB_c = (((Y_w[..., np.newaxis] * D[..., np.newaxis] / RGB_w) +
              1 - D[..., np.newaxis]) * RGB)

    return RGB_c


def full_chromatic_adaptation_reverse(RGB, RGB_w, Y_w, D):
    """
    Reverts full chromatic adaptation of given *CMCCAT2000* transform sharpened
    *RGB* array using given *CMCCAT2000* transform sharpened whitepoint
    *RGB_w* array.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* array.
    RGB_w : array_like
        *CMCCAT2000* transform sharpened whitepoint *RGB_w* array.
    Y_w : numeric or array_like
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    D : numeric or array_like
        Degree of adaptation :math:`D`.

    Returns
    -------
    ndarray
        Adapted *RGB* array.

    Examples
    --------
    >>> RGB = np.array([19.99370783, 20.00393634, 20.01326387])
    >>> RGB_w = np.array([94.930528, 103.536988, 108.717742])
    >>> Y_w = 100.0
    >>> D = 0.994468780088
    >>> full_chromatic_adaptation_reverse(RGB, RGB_w, Y_w, D)
    array([ 18.985456,  20.707422,  21.747482])
    """

    RGB = np.asarray(RGB)
    RGB_w = np.asarray(RGB_w)
    Y_w = np.asarray(Y_w)
    D = np.asarray(D)

    RGB_c = (RGB / (Y_w[..., np.newaxis] * (D[..., np.newaxis] / RGB_w) +
                    1 - D[..., np.newaxis]))

    return RGB_c


def RGB_to_rgb(RGB):
    """
    Converts given *RGB* array to *Hunt-Pointer-Estevez*
    :math:`\\rho\gamma\\beta` colourspace.

    Parameters
    ----------
    RGB : array_like
        *RGB* array.

    Returns
    -------
    ndarray
        *Hunt-Pointer-Estevez* :math:`\\rho\gamma\\beta` colourspace array.

    Examples
    --------
    >>> RGB = np.array([19.99370783, 20.00393634, 20.01326387])
    >>> RGB_to_rgb(RGB)  # doctest: +ELLIPSIS
    array([ 19.9969397...,  20.0018612...,  20.0135053...])
    """

    rgb = dot_vector(dot_matrix(XYZ_TO_HPE_MATRIX, CAT02_INVERSE_CAT), RGB)

    return rgb


def rgb_to_RGB(rgb):
    """
    Converts given *Hunt-Pointer-Estevez* :math:`\\rho\gamma\\beta` colourspace
    array to *RGB* array.

    Parameters
    ----------
    rgb : array_like
        *Hunt-Pointer-Estevez* :math:`\\rho\gamma\\beta` colourspace array.

    Returns
    -------
    ndarray
        *RGB* array.

    Examples
    --------
    >>> rgb = np.array([19.99693975, 20.00186123, 20.01350530])
    >>> rgb_to_RGB(rgb)  # doctest: +ELLIPSIS
    array([ 19.9937078...,  20.0039363...,  20.0132638...])
    """

    RGB = dot_vector(dot_matrix(CAT02_CAT, HPE_TO_XYZ_MATRIX), rgb)

    return RGB


def post_adaptation_non_linear_response_compression_forward(RGB, F_L):
    """
    Returns given *CMCCAT2000* transform sharpened *RGB* array with post
    adaptation non linear response compression.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* array.
    F_L : array_like
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    ndarray
        Compressed *CMCCAT2000* transform sharpened *RGB* array.

    Examples
    --------
    >>> RGB = np.array([19.99693975, 20.00186123, 20.01350530])
    >>> F_L = 1.16754446415
    >>> post_adaptation_non_linear_response_compression_forward(
    ...     RGB, F_L)  # doctest: +ELLIPSIS
    array([ 7.9463202...,  7.9471152...,  7.9489959...])
    """

    RGB = np.asarray(RGB)
    F_L = np.asarray(F_L)

    RGB_c = ((((np.sign(RGB) * 400 * (F_L[..., np.newaxis] * abs(RGB) / 100) ** 0.42) /
               (27.13 + (F_L[..., np.newaxis] * abs(RGB) / 100) ** 0.42))) + 0.1)

    return RGB_c


def post_adaptation_non_linear_response_compression_reverse(RGB, F_L):
    """
    Returns given *CMCCAT2000* transform sharpened *RGB* array without post
    adaptation non linear response compression.

    Parameters
    ----------
    RGB : array_like
        *CMCCAT2000* transform sharpened *RGB* array.
    F_L : array_like
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    ndarray
        Uncompressed *CMCCAT2000* transform sharpened *RGB* array.

    Examples
    --------
    >>> RGB = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> F_L = 1.16754446415
    >>> post_adaptation_non_linear_response_compression_reverse(
    ...     RGB, F_L)  # doctest: +ELLIPSIS
    array([ 19.9969397...,  20.0018612...,  20.0135052...])
    """

    RGB = np.asarray(RGB)
    F_L = np.asarray(F_L)

    RGB_p = ((np.sign(RGB - 0.1) *
              (100 / F_L[..., np.newaxis]) * ((27.13 * np.abs(RGB - 0.1)) /
                                              (400 - np.abs(RGB - 0.1))) ** (
                  1 / 0.42)))

    return RGB_p


def opponent_colour_dimensions_forward(RGB):
    """
    Returns opponent colour dimensions from given compressed *CMCCAT2000*
    transform sharpened *RGB* array for forward *CIECAM02* implementation.

    Parameters
    ----------
    RGB : array_like
        Compressed *CMCCAT2000* transform sharpened *RGB* array.

    Returns
    -------
    ndarray
        Opponent colour dimensions.

    Examples
    --------
    >>> RGB = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> opponent_colour_dimensions_forward(RGB)  # doctest: +ELLIPSIS
    array([-0.0006241..., -0.0005062...])
    """

    R, G, B = tsplit(RGB)

    a = R - 12 * G / 11 + B / 11
    b = (R + G - 2 * B) / 9

    ab = tstack((a, b))

    return ab


def opponent_colour_dimensions_reverse(P_n, h):
    """
    Returns opponent colour dimensions from given points :math:`P_n` and hue
    :math:`h` in degrees for reverse *CIECAM02* implementation.

    Parameters
    ----------
    P_n : array_like
        Points :math:`P_n`.
    h : numeric or array_like
        Hue :math:`h` in degrees.

    Returns
    -------
    ndarray
        Opponent colour dimensions.

    Examples
    --------
    >>> P_n = np.array([30162.89081534, 24.23720547, 1.05000000])
    >>> h = -140.95156734
    >>> opponent_colour_dimensions_reverse(P_n, h)  # doctest: +ELLIPSIS
    array([-0.0006241..., -0.0005062...])
    """

    P_1, P_2, P_3 = tsplit(P_n)
    hr = np.radians(h)

    sin_hr = np.sin(hr)
    cos_hr = np.cos(hr)

    P_4 = P_1 / sin_hr
    P_5 = P_1 / cos_hr
    n = P_2 * (2 + P_3) * (460 / 1403)

    a = np.zeros(hr.shape)
    b = np.zeros(hr.shape)

    b = np.where(np.abs(sin_hr) >= np.abs(cos_hr),
                 (n / (P_4 + (2 + P_3) * (220 / 1403) * (cos_hr / sin_hr) -
                       (27 / 1403) + P_3 * (6300 / 1403))),
                 b)

    a = np.where(np.abs(sin_hr) >= np.abs(cos_hr), b * (cos_hr / sin_hr), a)

    a = np.where(np.abs(sin_hr) < np.abs(cos_hr),
                 (n / (P_5 + (2 + P_3) * (220 / 1403) -
                       ((27 / 1403) - P_3 * (6300 / 1403)) *
                       (sin_hr / cos_hr))),
                 a)

    b = np.where(np.abs(sin_hr) < np.abs(cos_hr), a * (sin_hr / cos_hr), b)

    ab = tstack((a, b))

    return ab


def hue_angle(a, b):
    """
    Returns the *hue* angle :math:`h` in degrees.

    Parameters
    ----------
    a : numeric or array_like
        Opponent colour dimension :math:`a`.
    b : numeric or array_like
        Opponent colour dimension :math:`b`.

    Returns
    -------
    numeric or ndarray
        *Hue* angle :math:`h` in degrees.

    Examples
    --------
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> hue_angle(a, b)  # doctest: +ELLIPSIS
    219.0484326...
    """

    a = np.asarray(a)
    b = np.asarray(b)

    h = np.degrees(np.arctan2(b, a)) % 360

    return h


def hue_quadrature(h):
    """
    Returns the hue quadrature from given hue :math:`h` angle in degrees.

    Parameters
    ----------
    h : numeric or array_like
        Hue :math:`h` angle in degrees.

    Returns
    -------
    numeric or ndarray
        Hue quadrature.

    Examples
    --------
    >>> hue_quadrature(219.0484326582719)  # doctest: +ELLIPSIS
    278.0607358...
    """

    h = np.asarray(h)

    h_i = HUE_DATA_FOR_HUE_QUADRATURE['h_i']
    e_i = HUE_DATA_FOR_HUE_QUADRATURE['e_i']
    H_i = HUE_DATA_FOR_HUE_QUADRATURE['H_i']

    # *np.searchsorted* returns an erroneous index if a *nan* is used as input.
    h[np.asarray(np.isnan(h))] = 0
    i = np.asarray(np.searchsorted(h_i, h, side='left') - 1)

    h_ii = h_i[i]
    e_ii = e_i[i]
    H_ii = H_i[i]
    h_ii1 = h_i[i + 1]
    e_ii1 = e_i[i + 1]

    H = H_ii + ((100 * (h - h_ii) / e_ii) /
                ((h - h_ii) / e_ii + (h_ii1 - h) / e_ii1))
    H = np.where(h < 20.14,
                 385.9 + (14.1 * h / 0.856) / (h / 0.856 + (20.14 - h) / 0.8),
                 H)
    H = np.where(h >= 237.53,
                 H_ii + ((85.9 * (h - h_ii) / e_ii) /
                         ((h - h_ii) / e_ii + (360 - h) / 0.856)),
                 H)
    return as_numeric(H)


def eccentricity_factor(h):
    """
    Returns the eccentricity factor :math:`e_t` from given hue :math:`h` angle
    in degrees for forward *CIECAM02* implementation.

    Parameters
    ----------
    h : numeric or array_like
        Hue :math:`h` angle in degrees.

    Returns
    -------
    numeric or ndarray
        Eccentricity factor :math:`e_t`.

    Examples
    --------
    >>> eccentricity_factor(-140.951567342)  # doctest: +ELLIPSIS
    1.1740054...
    """

    h = np.asarray(h)

    e_t = 1 / 4 * (np.cos(2 + h * np.pi / 180) + 3.8)

    return e_t


def achromatic_response_forward(RGB, N_bb):
    """
    Returns the achromatic response :math:`A` from given compressed
    *CMCCAT2000* transform sharpened *RGB* array and :math:`N_{bb}` chromatic
    induction factor for forward *CIECAM02* implementation.

    Parameters
    ----------
    RGB : array_like
        Compressed *CMCCAT2000* transform sharpened *RGB* array.
    N_bb : numeric or array_like
        Chromatic induction factor :math:`N_{bb}`.

    Returns
    -------
    numeric or ndarray
        Achromatic response :math:`A`.

    Examples
    --------
    >>> RGB = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> N_bb = 1.000304004559381
    >>> achromatic_response_forward(RGB, N_bb)  # doctest: +ELLIPSIS
    23.9394809...
    """

    R, G, B = tsplit(RGB)

    A = (2 * R + G + (1 / 20) * B - 0.305) * N_bb

    return A


def achromatic_response_reverse(A_w, J, c, z):
    """
    Returns the achromatic response :math:`A` from given achromatic response
    :math:`A_w` for the whitepoint, *Lightness* correlate :math:`J`, surround
    exponential non linearity :math:`c` and base exponential non linearity
    :math:`z` for reverse *CIECAM02* implementation.

    Parameters
    ----------
    A_w : numeric or array_like
        Achromatic response :math:`A_w` for the whitepoint.
    J : numeric or array_like
        *Lightness* correlate :math:`J`.
    c : numeric or array_like
        Surround exponential non linearity :math:`c`.
    z : numeric or array_like
        Base exponential non linearity :math:`z`.

    Returns
    -------
    numeric or ndarray
        Achromatic response :math:`A`.

    Examples
    --------
    >>> A_w = 46.1882087914
    >>> J = 41.73109113251392
    >>> c = 0.69
    >>> z = 1.927213595499958
    >>> achromatic_response_reverse(A_w, J, c, z)  # doctest: +ELLIPSIS
    23.9394809...
    """

    A_w = np.asarray(A_w)
    J = np.asarray(J)
    c = np.asarray(c)
    z = np.asarray(z)

    A = A_w * (J / 100) ** (1 / (c * z))

    return A


def lightness_correlate(A, A_w, c, z):
    """
    Returns the *Lightness* correlate :math:`J`.

    Parameters
    ----------
    A : numeric or array_like
        Achromatic response :math:`A` for the stimulus.
    A_w : numeric or array_like
        Achromatic response :math:`A_w` for the whitepoint.
    c : numeric or array_like
        Surround exponential non linearity :math:`c`.
    z : numeric or array_like
        Base exponential non linearity :math:`z`.

    Returns
    -------
    numeric or ndarray
        *Lightness* correlate :math:`J`.

    Examples
    --------
    >>> A = 23.9394809667
    >>> A_w = 46.1882087914
    >>> c = 0.69
    >>> z = 1.9272135955
    >>> lightness_correlate(A, A_w, c, z)  # doctest: +ELLIPSIS
    41.7310911...
    """

    A = np.asarray(A)
    A_w = np.asarray(A_w)
    c = np.asarray(c)
    z = np.asarray(z)

    J = 100 * (A / A_w) ** (c * z)

    return J


def brightness_correlate(c, J, A_w, F_L):
    """
    Returns the *brightness* correlate :math:`Q`.

    Parameters
    ----------
    c : numeric or array_like
        Surround exponential non linearity :math:`c`.
    J : numeric or array_like
        *Lightness* correlate :math:`J`.
    A_w : numeric or array_like
        Achromatic response :math:`A_w` for the whitepoint.
    F_L : numeric or array_like
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    numeric or ndarray
        *Brightness* correlate :math:`Q`.

    Examples
    --------
    >>> c = 0.69
    >>> J = 41.7310911325
    >>> A_w = 46.1882087914
    >>> F_L = 1.16754446415
    >>> brightness_correlate(c, J, A_w, F_L)  # doctest: +ELLIPSIS
    195.3713259...
    """

    c = np.asarray(c)
    J = np.asarray(J)
    A_w = np.asarray(A_w)
    F_L = np.asarray(F_L)

    Q = (4 / c) * np.sqrt(J / 100) * (A_w + 4) * F_L ** 0.25

    return Q


def temporary_magnitude_quantity_forward(N_c, N_cb, e_t, a, b, RGB_a):
    """
    Returns the temporary magnitude quantity :math:`t`. for forward *CIECAM02*
    implementation.

    Parameters
    ----------
    N_c : numeric or array_like
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb : numeric or array_like
        Chromatic induction factor :math:`N_{cb}`.
    e_t : numeric or array_like
        Eccentricity factor :math:`e_t`.
    a : numeric or array_like
        Opponent colour dimension :math:`a`.
    b : numeric or array_like
        Opponent colour dimension :math:`b`.
    RGB_a : array_like
        Compressed stimulus *CMCCAT2000* transform sharpened *RGB* array.

    Returns
    -------
    numeric or ndarray
         Temporary magnitude quantity :math:`t`.

    Examples
    --------
    >>> N_c = 1.0
    >>> N_cb = 1.00030400456
    >>> e_t = 1.174005472851914
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> RGB_a = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> temporary_magnitude_quantity_forward(  # doctest: +ELLIPSIS
    ...     N_c, N_cb, e_t, a, b, RGB_a)
    0.1497462...
    """

    N_c = np.asarray(N_c)
    N_cb = np.asarray(N_cb)
    e_t = np.asarray(e_t)
    a = np.asarray(a)
    b = np.asarray(b)
    Ra, Ga, Ba = tsplit(RGB_a)

    t = (((50000 / 13) * N_c * N_cb) * (e_t * (a ** 2 + b ** 2) ** 0.5) /
         (Ra + Ga + 21 * Ba / 20))

    return t


def temporary_magnitude_quantity_reverse(C, J, n):
    """
    Returns the temporary magnitude quantity :math:`t`. for reverse *CIECAM02*
    implementation.

    Parameters
    ----------
    C : numeric or array_like
        *Chroma* correlate :math:`C`.
    J : numeric or array_like
        *Lightness* correlate :math:`J`.
    n : numeric or array_like
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    numeric or ndarray
         Temporary magnitude quantity :math:`t`.

    Examples
    --------
    >>> C = 68.8364136888275
    >>> J = 41.749268505999
    >>> n = 0.2
    >>> temporary_magnitude_quantity_reverse(C, J, n)  # doctest: +ELLIPSIS
    202.3873619...
   """

    C = np.asarray(C)
    J = np.asarray(J)
    n = np.asarray(n)

    t = (C / (np.sqrt(J / 100) * (1.64 - 0.29 ** n) ** 0.73)) ** (1 / 0.9)

    return t


def chroma_correlate(J, n, N_c, N_cb, e_t, a, b, RGB_a):
    """
    Returns the *chroma* correlate :math:`C`.

    Parameters
    ----------
    J : numeric or array_like
        *Lightness* correlate :math:`J`.
    n : numeric or array_like
        Function of the luminance factor of the background :math:`n`.
    N_c : numeric or array_like
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb : numeric or array_like
        Chromatic induction factor :math:`N_{cb}`.
    e_t : numeric or array_like
        Eccentricity factor :math:`e_t`.
    a : numeric or array_like
        Opponent colour dimension :math:`a`.
    b : numeric or array_like
        Opponent colour dimension :math:`b`.
    RGB_a : array_like
        Compressed stimulus *CMCCAT2000* transform sharpened *RGB* array.

    Returns
    -------
    numeric or ndarray
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
    >>> RGB_a = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> chroma_correlate(  # doctest: +ELLIPSIS
    ...     J, n, N_c, N_cb, e_t, a, b, RGB_a)
    0.1047077...
    """

    J = np.asarray(J)
    n = np.asarray(n)

    t = temporary_magnitude_quantity_forward(N_c, N_cb, e_t, a, b, RGB_a)
    C = t ** 0.9 * (J / 100) ** 0.5 * (1.64 - 0.29 ** n) ** 0.73

    return C


def colourfulness_correlate(C, F_L):
    """
    Returns the *colourfulness* correlate :math:`M`.

    Parameters
    ----------
    C : numeric or array_like
        *Chroma* correlate :math:`C`.
    F_L : numeric or array_like
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    numeric or ndarray
        *Colourfulness* correlate :math:`M`.

    Examples
    --------
    >>> C = 0.104707757171
    >>> F_L = 1.16754446415
    >>> colourfulness_correlate(C, F_L)  # doctest: +ELLIPSIS
    0.1088421...
    """

    C = np.asarray(C)
    F_L = np.asarray(F_L)

    M = C * F_L ** 0.25

    return M


def saturation_correlate(M, Q):
    """
    Returns the *saturation* correlate :math:`s`.

    Parameters
    ----------
    M : numeric or array_like
        *Colourfulness* correlate :math:`M`.
    Q : numeric or array_like
        *Brightness* correlate :math:`C`.

    Returns
    -------
    numeric or ndarray
        *Saturation* correlate :math:`s`.

    Examples
    --------
    >>> M = 0.108842175669
    >>> Q = 195.371325966
    >>> saturation_correlate(M, Q)  # doctest: +ELLIPSIS
    2.3603053...
    """

    M = np.asarray(M)
    Q = np.asarray(Q)

    s = 100 * (M / Q) ** 0.5

    return s


def P(N_c, N_cb, e_t, t, A, N_bb):
    """
    Returns the points :math:`P_1`, :math:`P_2` and :math:`P_3`.

    Parameters
    ----------
    N_c : numeric or array_like
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb : numeric or array_like
        Chromatic induction factor :math:`N_{cb}`.
    e_t : numeric or array_like
        Eccentricity factor :math:`e_t`.
    t : numeric or array_like
        Temporary magnitude quantity :math:`t`.
    A : numeric or array_like
        Achromatic response  :math:`A` for the stimulus.
    N_bb : numeric or array_like
        Chromatic induction factor :math:`N_{bb}`.

    Returns
    -------
    ndarray
        Points :math:`P`.

    Examples
    --------
    >>> N_c = 1.0
    >>> N_cb = 1.00030400456
    >>> e_t = 1.174005472851914
    >>> t = 0.149746202921
    >>> A = 23.9394809667
    >>> N_bb = 1.00030400456
    >>> P(N_c, N_cb, e_t, t, A, N_bb)  # doctest: +ELLIPSIS
    array([  3.0162890...e+04,   2.4237205...e+01,   1.0500000...e+00])
    """

    N_c = np.asarray(N_c)
    N_cb = np.asarray(N_cb)
    e_t = np.asarray(e_t)
    t = np.asarray(t)
    A = np.asarray(A)
    N_bb = np.asarray(N_bb)

    P_1 = ((50000 / 13) * N_c * N_cb * e_t) / t
    P_2 = A / N_bb + 0.305
    P_3 = np.ones(P_1.shape) * (21 / 20)

    P_n = tstack((P_1, P_2, P_3))

    return P_n


def post_adaptation_non_linear_response_compression_matrix(P_2, a, b):
    """
    Returns the post adaptation non linear response compression matrix.

    Parameters
    ----------
    P_2 : numeric or array_like
        Point :math:`P_2`.
    a : numeric or array_like
        Opponent colour dimension :math:`a`.
    b : numeric or array_like
        Opponent colour dimension :math:`b`.

    Returns
    -------
    ndarray
        Points :math:`P`.

    Examples
    --------
    >>> P_2 = 24.2372054671
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> post_adaptation_non_linear_response_compression_matrix(
    ...     P_2, a, b)  # doctest: +ELLIPSIS
    array([ 7.9463202...,  7.9471152...,  7.9489959...])
    """

    P_2 = np.asarray(P_2)
    a = np.asarray(a)
    b = np.asarray(b)

    R_a = (460 * P_2 + 451 * a + 288 * b) / 1403
    G_a = (460 * P_2 - 891 * a - 261 * b) / 1403
    B_a = (460 * P_2 - 220 * a - 6300 * b) / 1403

    RGB_a = tstack((R_a, G_a, B_a))

    return RGB_a
