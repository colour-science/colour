# -*- coding: utf-8 -*-
"""
Hunt Colour Appearance Model
============================

Defines *Hunt* colour appearance model objects:

-   :class:`colour.appearance.Hunt_InductionFactors`
-   :attr:`colour.HUNT_VIEWING_CONDITIONS`
-   :class:`colour.Hunt_Specification`
-   :func:`colour.XYZ_to_Hunt`

See Also
--------
`Hunt Colour Appearance Model Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/
blob/master/notebooks/appearance/hunt.ipynb>`_

References
----------
-   :cite:`Fairchild2013u` : Fairchild, M. D. (2013). The Hunt Model. In Color
    Appearance Models (3rd ed., pp. 5094-5556). Wiley. ISBN:B00DAYO8E2
-   :cite:`Hunt2004b` : Hunt, R. W. G. (2004). The Reproduction of Colour
    (6th ed.). Chichester, UK: John Wiley & Sons, Ltd. doi:10.1002/0470024275
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.algebra import spow
from colour.utilities import (CaseInsensitiveMapping, as_float_array,
                              dot_vector, from_range_degrees, to_domain_100,
                              tsplit, tstack, usage_warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'Hunt_InductionFactors', 'HUNT_VIEWING_CONDITIONS',
    'HUE_DATA_FOR_HUE_QUADRATURE', 'XYZ_TO_HPE_MATRIX', 'HPE_TO_XYZ_MATRIX',
    'Hunt_ReferenceSpecification', 'Hunt_Specification', 'XYZ_to_Hunt',
    'luminance_level_adaptation_factor', 'illuminant_scotopic_luminance',
    'XYZ_to_rgb', 'f_n', 'chromatic_adaptation',
    'adjusted_reference_white_signals', 'achromatic_post_adaptation_signal',
    'colour_difference_signals', 'hue_angle', 'eccentricity_factor',
    'low_luminance_tritanopia_factor', 'yellowness_blueness_response',
    'redness_greenness_response', 'overall_chromatic_response',
    'saturation_correlate', 'achromatic_signal', 'brightness_correlate',
    'lightness_correlate', 'chroma_correlate', 'colourfulness_correlate'
]


class Hunt_InductionFactors(
        namedtuple('Hunt_InductionFactors', ('N_c', 'N_b', 'N_cb', 'N_bb'))):
    """
    *Hunt* colour appearance model induction factors.

    Parameters
    ----------
    N_c : numeric or array_like
        Chromatic surround induction factor :math:`N_c`.
    N_b : numeric or array_like
        *Brightness* surround induction factor :math:`N_b`.
    N_cb : numeric or array_like, optional
        Chromatic background induction factor :math:`N_{cb}`, approximated
        using tristimulus values :math:`Y_w` and :math:`Y_b` of
        respectively the reference white and the background if not specified.
    N_bb : numeric or array_like, optional
        *Brightness* background induction factor :math:`N_{bb}`, approximated
        using tristimulus values :math:`Y_w` and :math:`Y_b` of
        respectively the reference white and the background if not specified.

    References
    ----------
    :cite:`Fairchild2013u`, :cite:`Hunt2004b`
    """

    def __new__(cls, N_c, N_b, N_cb=None, N_bb=None):
        """
        Returns a new instance of the
        :class:`colour.appearance.Hunt_InductionFactors` class.
        """

        return super(Hunt_InductionFactors, cls).__new__(
            cls, N_c, N_b, N_cb, N_bb)


HUNT_VIEWING_CONDITIONS = CaseInsensitiveMapping({
    'Small Areas, Uniform Background & Surrounds':
        Hunt_InductionFactors(1, 300),
    'Normal Scenes':
        Hunt_InductionFactors(1, 75),
    'Television & CRT, Dim Surrounds':
        Hunt_InductionFactors(1, 25),
    'Large Transparencies On Light Boxes':
        Hunt_InductionFactors(0.7, 25),
    'Projected Transparencies, Dark Surrounds':
        Hunt_InductionFactors(0.7, 10)
})
HUNT_VIEWING_CONDITIONS.__doc__ = """
Reference *Hunt* colour appearance model viewing conditions.

References
----------
:cite:`Fairchild2013u`, :cite:`Hunt2004b`

HUNT_VIEWING_CONDITIONS : CaseInsensitiveMapping
    **{'Small Areas, Uniform Background & Surrounds',
    'Normal Scenes',
    'Television & CRT, Dim Surrounds',
    'Large Transparencies On Light Boxes',
    'Projected Transparencies, Dark Surrounds'}**

Aliases:

-   'small_uniform': 'Small Areas, Uniform Background & Surrounds'
-   'normal': 'Normal Scenes'
-   'tv_dim': 'Television & CRT, Dim Surrounds'
-   'light_boxes': 'Large Transparencies On Light Boxes'
-   'projected_dark': 'Projected Transparencies, Dark Surrounds'

"""
HUNT_VIEWING_CONDITIONS['small_uniform'] = (
    HUNT_VIEWING_CONDITIONS['Small Areas, Uniform Background & Surrounds'])
HUNT_VIEWING_CONDITIONS['normal'] = (HUNT_VIEWING_CONDITIONS['Normal Scenes'])
HUNT_VIEWING_CONDITIONS['tv_dim'] = (
    HUNT_VIEWING_CONDITIONS['Television & CRT, Dim Surrounds'])
HUNT_VIEWING_CONDITIONS['light_boxes'] = (
    HUNT_VIEWING_CONDITIONS['Large Transparencies On Light Boxes'])
HUNT_VIEWING_CONDITIONS['projected_dark'] = (
    HUNT_VIEWING_CONDITIONS['Projected Transparencies, Dark Surrounds'])

HUE_DATA_FOR_HUE_QUADRATURE = {
    'h_s': np.array([20.14, 90.00, 164.25, 237.53]),
    'e_s': np.array([0.8, 0.7, 1.0, 1.2])
}

XYZ_TO_HPE_MATRIX = np.array([
    [0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18340, 0.04641],
    [0.00000, 0.00000, 1.00000],
])
"""
*Hunt* colour appearance model *CIE XYZ* tristimulus values to
*Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace matrix.

XYZ_TO_HPE_MATRIX : array_like, (3, 3)
"""

HPE_TO_XYZ_MATRIX = np.linalg.inv(XYZ_TO_HPE_MATRIX)
"""
*Hunt* colour appearance model *Hunt-Pointer-Estevez*
:math:`\\rho\\gamma\\beta` colourspace to *CIE XYZ* tristimulus values matrix.

HPE_TO_XYZ_MATRIX : array_like, (3, 3)
"""


class Hunt_ReferenceSpecification(
        namedtuple('Hunt_ReferenceSpecification',
                   ('J', 'C_94', 'h_S', 's', 'Q', 'M_94', 'H', 'H_C'))):
    """
    Defines the *Hunt* colour appearance model reference specification.

    This specification has field names consistent with *Fairchild (2013)*
    reference.

    Parameters
    ----------
    J : numeric or array_like
        Correlate of *Lightness* :math:`J`.
    C_94 : numeric or array_like
        Correlate of *chroma* :math:`C_94`.
    h_S : numeric or array_like
        *Hue* angle :math:`h_S` in degrees.
    s : numeric or array_like
        Correlate of *saturation* :math:`s`.
    Q : numeric or array_like
        Correlate of *brightness* :math:`Q`.
    M_94 : numeric or array_like
        Correlate of *colourfulness* :math:`M_94`.
    H : numeric or array_like
        *Hue* :math:`h` quadrature :math:`H`.
    H_C : numeric or array_like
        *Hue* :math:`h` composition :math:`H_C`.

    References
    ----------
    :cite:`Fairchild2013u`, :cite:`Hunt2004b`
    """


class Hunt_Specification(
        namedtuple('Hunt_Specification',
                   ('J', 'C', 'h', 's', 'Q', 'M', 'H', 'HC'))):
    """
    Defines the *Hunt* colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    *Fairchild (2013)* reference.

    Parameters
    ----------
    J : numeric or array_like
        Correlate of *Lightness* :math:`J`.
    C : numeric or array_like
        Correlate of *chroma* :math:`C_94`.
    h : numeric or array_like
        *Hue* angle :math:`h_S` in degrees.
    s : numeric or array_like
        Correlate of *saturation* :math:`s`.
    Q : numeric or array_like
        Correlate of *brightness* :math:`Q`.
    M : numeric or array_like
        Correlate of *colourfulness* :math:`M_94`.
    H : numeric or array_like
        *Hue* :math:`h` quadrature :math:`H`.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H_C`.

    Notes
    -----
    -   This specification is the one used in the current model implementation.

    References
    ----------
    :cite:`Fairchild2013u`, :cite:`Hunt2004b`
    """


def XYZ_to_Hunt(XYZ,
                XYZ_w,
                XYZ_b,
                L_A,
                surround=HUNT_VIEWING_CONDITIONS['Normal Scenes'],
                L_AS=None,
                CCT_w=None,
                XYZ_p=None,
                p=None,
                S=None,
                S_w=None,
                helson_judd_effect=False,
                discount_illuminant=True):
    """
    Computes the *Hunt* colour appearance model correlates.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_w : array_like
        *CIE XYZ* tristimulus values of reference white.
    XYZ_b : array_like
        *CIE XYZ* tristimulus values of background.
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    surround : Hunt_InductionFactors, optional
         Surround viewing conditions induction factors.
    L_AS : numeric or array_like, optional
        Scotopic luminance :math:`L_{AS}` of the illuminant, approximated if
        not specified.
    CCT_w : numeric or array_like, optional
        Correlated color temperature :math:`T_{cp}`: of the illuminant, needed
        to approximate :math:`L_{AS}`.
    XYZ_p : array_like, optional
        *CIE XYZ* tristimulus values of proximal field, assumed to be equal to
        background if not specified.
    p : numeric or array_like, optional
        Simultaneous contrast / assimilation factor :math:`p` with value
        normalised to domain [-1, 0] when simultaneous contrast occurs and
        normalised to domain [0, 1] when assimilation occurs.
    S : numeric or array_like, optional
        Scotopic response :math:`S` to the stimulus, approximated using
        tristimulus values :math:`Y` of the stimulus if not specified.
    S_w : numeric or array_like, optional
        Scotopic response :math:`S_w` for the reference white, approximated
        using the tristimulus values :math:`Y_w` of the reference white if not
        specified.
    helson_judd_effect : bool, optional
        Truth value indicating whether the *Helson-Judd* effect should be
        accounted for.
    discount_illuminant : bool, optional
       Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    Hunt_Specification
        *Hunt* colour appearance model specification.

    Raises
    ------
    ValueError
        If an illegal arguments combination is specified.

    Notes
    -----

    +--------------------------+-----------------------+---------------+
    | **Domain**               | **Scale - Reference** | **Scale - 1** |
    +==========================+=======================+===============+
    | ``XYZ``                  | [0, 100]              | [0, 1]        |
    +--------------------------+-----------------------+---------------+
    | ``XYZ_w``                | [0, 100]              | [0, 1]        |
    +--------------------------+-----------------------+---------------+
    | ``XYZ_b``                | [0, 100]              | [0, 1]        |
    +--------------------------+-----------------------+---------------+
    | ``XYZ_p``                | [0, 100]              | [0, 1]        |
    +--------------------------+-----------------------+---------------+

    +--------------------------+-----------------------+---------------+
    | **Range**                | **Scale - Reference** | **Scale - 1** |
    +==========================+=======================+===============+
    | ``Hunt_Specification.h`` | [0, 360]              | [0, 1]        |
    +--------------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2013u`, :cite:`Hunt2004b`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> XYZ_b = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> surround = HUNT_VIEWING_CONDITIONS['Normal Scenes']
    >>> CCT_w = 6504.0
    >>> XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)
    ... # doctest: +ELLIPSIS
    Hunt_Specification(J=30.0462678..., C=0.1210508..., h=269.2737594..., \
s=0.0199093..., Q=22.2097654..., M=0.1238964..., H=None, HC=None)
    """
    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    XYZ_b = to_domain_100(XYZ_b)
    _X, Y, _Z = tsplit(XYZ)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    X_b, Y_b, _Z_b = tsplit(XYZ_b)

    # Arguments handling.
    if XYZ_p is not None:
        X_p, Y_p, Z_p = tsplit(to_domain_100(XYZ_p))
    else:
        X_p = X_b
        Y_p = Y_b
        Z_p = Y_b
        usage_warning('Unspecified proximal field "XYZ_p" argument, using '
                      'background "XYZ_b" as approximation!')

    if surround.N_cb is None:
        N_cb = 0.725 * spow(Y_w / Y_b, 0.2)
        usage_warning('Unspecified "N_cb" argument, using approximation: '
                      '"{0}"'.format(N_cb))
    if surround.N_bb is None:
        N_bb = 0.725 * spow(Y_w / Y_b, 0.2)
        usage_warning('Unspecified "N_bb" argument, using approximation: '
                      '"{0}"'.format(N_bb))

    if L_AS is None and CCT_w is None:
        raise ValueError('Either the scotopic luminance "L_AS" of the '
                         'illuminant or its correlated colour temperature '
                         '"CCT_w" must be specified!')
    if L_AS is None:
        L_AS = illuminant_scotopic_luminance(L_A, CCT_w)
        usage_warning(
            'Unspecified "L_AS" argument, using approximation from "CCT": '
            '"{0}"'.format(L_AS))

    if (S is None and S_w is not None) or (S is not None and S_w is None):
        raise ValueError('Either both stimulus scotopic response "S" and '
                         'reference white scotopic response "S_w" arguments '
                         'need to be specified or none of them!')
    elif S is None and S_w is None:
        S = Y
        S_w = Y_w
        usage_warning(
            'Unspecified stimulus scotopic response "S" and reference '
            'white scotopic response "S_w" arguments, using '
            'approximation: "{0}", "{1}"'.format(S, S_w))

    if p is None:
        usage_warning(
            'Unspecified simultaneous contrast / assimilation "p" '
            'argument, model will not account for simultaneous chromatic '
            'contrast!')

    XYZ_p = tstack([X_p, Y_p, Z_p])

    # Computing luminance level adaptation factor :math:`F_L`.
    F_L = luminance_level_adaptation_factor(L_A)

    # Computing test sample chromatic adaptation.
    rgb_a = chromatic_adaptation(XYZ, XYZ_w, XYZ_b, L_A, F_L, XYZ_p, p,
                                 helson_judd_effect, discount_illuminant)

    # Computing reference white chromatic adaptation.
    rgb_aw = chromatic_adaptation(XYZ_w, XYZ_w, XYZ_b, L_A, F_L, XYZ_p, p,
                                  helson_judd_effect, discount_illuminant)

    # Computing opponent colour dimensions.
    # Computing achromatic post adaptation signals.
    A_a = achromatic_post_adaptation_signal(rgb_a)
    A_aw = achromatic_post_adaptation_signal(rgb_aw)

    # Computing colour difference signals.
    C = colour_difference_signals(rgb_a)
    C_w = colour_difference_signals(rgb_aw)

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`h_s`.
    # -------------------------------------------------------------------------
    h = hue_angle(C)
    # hue_w = hue_angle(C_w)
    # TODO: Implement hue quadrature & composition computation.

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s`.
    # -------------------------------------------------------------------------
    # Computing eccentricity factors.
    e_s = eccentricity_factor(h)

    # Computing low luminance tritanopia factor :math:`F_t`.
    F_t = low_luminance_tritanopia_factor(L_A)

    M_yb = yellowness_blueness_response(C, e_s, surround.N_c, N_cb, F_t)
    M_rg = redness_greenness_response(C, e_s, surround.N_c, N_cb)
    M_yb_w = yellowness_blueness_response(C_w, e_s, surround.N_c, N_cb, F_t)
    M_rg_w = redness_greenness_response(C_w, e_s, surround.N_c, N_cb)

    # Computing overall chromatic response.
    M = overall_chromatic_response(M_yb, M_rg)
    M_w = overall_chromatic_response(M_yb_w, M_rg_w)

    s = saturation_correlate(M, rgb_a)

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`Q`.
    # -------------------------------------------------------------------------
    # Computing achromatic signal :math:`A`.
    A = achromatic_signal(L_AS, S, S_w, N_bb, A_a)
    A_w = achromatic_signal(L_AS, S_w, S_w, N_bb, A_aw)

    Q = brightness_correlate(A, A_w, M, surround.N_b)
    brightness_w = brightness_correlate(A_w, A_w, M_w, surround.N_b)
    # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`J`.
    # -------------------------------------------------------------------------
    J = lightness_correlate(Y_b, Y_w, Q, brightness_w)

    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`C_{94}`.
    # -------------------------------------------------------------------------
    C_94 = chroma_correlate(s, Y_b, Y_w, Q, brightness_w)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`M_{94}`.
    # -------------------------------------------------------------------------
    M_94 = colourfulness_correlate(F_L, C_94)

    return Hunt_Specification(J, C_94, from_range_degrees(h), s, Q, M_94, None,
                              None)


def luminance_level_adaptation_factor(L_A):
    """
    Returns the *luminance* level adaptation factor :math:`F_L`.

    Parameters
    ----------
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    numeric or ndarray
        *Luminance* level adaptation factor :math:`F_L`

    Examples
    --------
    >>> luminance_level_adaptation_factor(318.31)  # doctest: +ELLIPSIS
    1.1675444...
    """

    L_A = as_float_array(L_A)

    k = 1 / (5 * L_A + 1)
    k4 = k ** 4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)

    return F_L


def illuminant_scotopic_luminance(L_A, CCT):
    """
    Returns the approximate scotopic luminance :math:`L_{AS}` of the
    illuminant.

    Parameters
    ----------
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    CCT : numeric or array_like
        Correlated color temperature :math:`T_{cp}` of the illuminant.

    Returns
    -------
    numeric or ndarray
        Approximate scotopic luminance :math:`L_{AS}`.

    Examples
    --------
    >>> illuminant_scotopic_luminance(318.31, 6504.0)  # doctest: +ELLIPSIS
    769.9376286...
    """

    L_A = as_float_array(L_A)
    CCT = as_float_array(CCT)

    CCT = 2.26 * L_A * spow((CCT / 4000) - 0.4, 1 / 3)

    return CCT


def XYZ_to_rgb(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to *Hunt-Pointer-Estevez*
    :math:`\\rho\\gamma\\beta` colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_to_rgb(XYZ)  # doctest: +ELLIPSIS
    array([ 19.4743367...,  20.3101217...,  21.78     ])
    """

    return dot_vector(XYZ_TO_HPE_MATRIX, XYZ)


def f_n(x):
    """
    Defines the nonlinear response function of the *Hunt* colour appearance
    model used to model the nonlinear behaviour of various visual responses.

    Parameters
    ----------
    x : numeric or array_like or array_like
        Visual response variable :math:`x`.

    Returns
    -------
    numeric or array_like
        Modeled visual response variable :math:`x`.


    Examples
    --------
    >>> x = np.array([0.23350512, 0.23351103, 0.23355179])
    >>> f_n(x)  # doctest: +ELLIPSIS
    array([ 5.8968592...,  5.8969521...,  5.8975927...])
    """

    x = as_float_array(x)

    x_p = spow(x, 0.73)
    x_m = 40 * (x_p / (x_p + 2))

    return x_m


def chromatic_adaptation(XYZ,
                         XYZ_w,
                         XYZ_b,
                         L_A,
                         F_L,
                         XYZ_p=None,
                         p=None,
                         helson_judd_effect=False,
                         discount_illuminant=True):
    """
    Applies chromatic adaptation to given *CIE XYZ* tristimulus values.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of test sample.
    XYZ_b : array_like
        *CIE XYZ* tristimulus values of background.
    XYZ_w : array_like
        *CIE XYZ* tristimulus values of reference white.
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    F_L : numeric or array_like
        Luminance adaptation factor :math:`F_L`.
    XYZ_p : array_like, optional
        *CIE XYZ* tristimulus values of proximal field, assumed to be equal to
        background if not specified.
    p : numeric or array_like, optional
        Simultaneous contrast / assimilation factor :math:`p` with value
        normalised to  domain [-1, 0] when simultaneous contrast occurs and
        normalised to domain [0, 1] when assimilation occurs.
    helson_judd_effect : bool, optional
        Truth value indicating whether the *Helson-Judd* effect should be
        accounted for.
    discount_illuminant : bool, optional
       Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    ndarray
        Adapted *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_b = np.array([95.05, 100.00, 108.88])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> F_L = 1.16754446415
    >>> chromatic_adaptation(XYZ, XYZ_w, XYZ_b, L_A, F_L)  # doctest: +ELLIPSIS
    array([ 6.8959454...,  6.8959991...,  6.8965708...])
    """

    XYZ_w = as_float_array(XYZ_w)
    XYZ_b = as_float_array(XYZ_b)
    L_A = as_float_array(L_A)
    F_L = as_float_array(F_L)

    rgb = XYZ_to_rgb(XYZ)
    rgb_w = XYZ_to_rgb(XYZ_w)
    Y_w = XYZ_w[..., 1]
    Y_b = XYZ_b[..., 1]

    h_rgb = 3 * rgb_w / np.sum(rgb_w, axis=-1)[..., np.newaxis]

    # Computing chromatic adaptation factors.
    if not discount_illuminant:
        L_A_p = spow(L_A, 1 / 3)
        F_rgb = ((1 + L_A_p + h_rgb) / (1 + L_A_p + (1 / h_rgb)))
    else:
        F_rgb = np.ones(h_rgb.shape)

    # Computing Helson-Judd effect parameters.
    if helson_judd_effect:
        D_rgb = (f_n((Y_b / Y_w) * F_L * F_rgb[..., 1]) - f_n(
            (Y_b / Y_w) * F_L * F_rgb))
    else:
        D_rgb = np.zeros(F_rgb.shape)

    # Computing cone bleach factors.
    B_rgb = (10 ** 7) / ((10 ** 7) + 5 * L_A[..., np.newaxis] * (rgb_w / 100))

    # Computing adjusted reference white signals.
    if XYZ_p is not None and p is not None:
        rgb_p = XYZ_to_rgb(XYZ_p)
        rgb_w = adjusted_reference_white_signals(rgb_p, B_rgb, rgb_w, p)

    # Computing adapted cone responses.
    rgb_a = 1
    rgb_a += B_rgb * (f_n(F_L[..., np.newaxis] * F_rgb * rgb / rgb_w) + D_rgb)

    return rgb_a


def adjusted_reference_white_signals(rgb_p, rgb_b, rgb_w, p):
    """
    Adjusts the white point for simultaneous chromatic contrast.

    Parameters
    ----------
    rgb_p :  array_like
        Cone signals *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
        colourspace array of the proximal field.
    rgb_b :  array_like
        Cone signals *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
        colourspace array of the background.
    rgb_w :  array_like
        Cone signals array *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
        colourspace array of the reference white.
    p : numeric or array_like
        Simultaneous contrast / assimilation factor :math:`p` with value
        normalised to domain [-1, 0] when simultaneous contrast occurs and
        normalised to domain [0, 1] when assimilation occurs.

    Returns
    -------
    ndarray
        Adjusted cone signals *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
        colourspace array of the reference white.

    Examples
    --------
    >>> rgb_p = np.array([98.07193550, 101.13755950, 100.00000000])
    >>> rgb_b = np.array([0.99984505, 0.99983840, 0.99982674])
    >>> rgb_w = np.array([97.37325710, 101.54968030, 108.88000000])
    >>> p = 0.1
    >>> adjusted_reference_white_signals(rgb_p, rgb_b, rgb_w, p)
    ... # doctest: +ELLIPSIS
    array([ 88.0792742...,  91.8569553...,  98.4876543...])
    """

    rgb_p = as_float_array(rgb_p)
    rgb_b = as_float_array(rgb_b)
    rgb_w = as_float_array(rgb_w)
    p = as_float_array(p)

    p_rgb = rgb_p / rgb_b
    rgb_w = (rgb_w * (spow((1 - p) * p_rgb + (1 + p) / p_rgb, 0.5)) / (spow(
        (1 + p) * p_rgb + (1 - p) / p_rgb, 0.5)))

    return rgb_w


def achromatic_post_adaptation_signal(rgb):
    """
    Returns the achromatic post adaptation signal :math:`A` from given
    *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace array.

    Parameters
    ----------
    rgb : array_like
        *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace array.

    Returns
    -------
    numeric or ndarray
        Achromatic post adaptation signal :math:`A`.

    Examples
    --------
    >>> rgb = np.array([6.89594549, 6.89599915, 6.89657085])
    >>> achromatic_post_adaptation_signal(rgb)  # doctest: +ELLIPSIS
    18.9827186...
    """

    r, g, b = tsplit(rgb)

    A = 2 * r + g + (1 / 20) * b - 3.05 + 1

    return A


def colour_difference_signals(rgb):
    """
    Returns the colour difference signals :math:`C_1`, :math:`C_2` and
    :math:`C_3` from given *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
    colourspace array.

    Parameters
    ----------
    rgb : array_like
        *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace array.

    Returns
    -------
    ndarray
        Colour difference signals :math:`C_1`, :math:`C_2` and :math:`C_3`.

    Examples
    --------
    >>> rgb = np.array([6.89594549, 6.89599915, 6.89657085])
    >>> colour_difference_signals(rgb)  # doctest: +ELLIPSIS
    array([ -5.3660000...e-05,  -5.7170000...e-04,   6.2536000...e-04])
    """

    r, g, b = tsplit(rgb)

    C_1 = r - g
    C_2 = g - b
    C_3 = b - r

    C = tstack([C_1, C_2, C_3])

    return C


def hue_angle(C):
    """
    Returns the *hue* angle :math:`h` in degrees from given colour difference
    signals :math:`C`.

    Parameters
    ----------
    C : array_like
        Colour difference signals :math:`C`.

    Returns
    -------
    numeric or ndarray
        *Hue* angle :math:`h` in degrees.

    Examples
    --------
    >>> C = np.array([
    ...     -5.365865581996587e-05,
    ...     -0.000571699383647,
    ...     0.000625358039467
    ... ])
    >>> hue_angle(C)  # doctest: +ELLIPSIS
    269.2737594...
    """

    C_1, C_2, C_3 = tsplit(C)

    hue = (180 * np.arctan2(0.5 * (C_2 - C_3) / 4.5, C_1 -
                            (C_2 / 11)) / np.pi) % 360
    return hue


def eccentricity_factor(hue):
    """
    Returns eccentricity factor :math:`e_s` from given hue angle :math:`h`
    in degrees.

    Parameters
    ----------
    hue : numeric or array_like
        Hue angle :math:`h` in degrees.

    Returns
    -------
    numeric or ndarray
        Eccentricity factor :math:`e_s`.

    Examples
    --------
    >>> eccentricity_factor(269.273759)  # doctest: +ELLIPSIS
    array(1.1108365...)
    """

    hue = as_float_array(hue)

    h_s = HUE_DATA_FOR_HUE_QUADRATURE['h_s']
    e_s = HUE_DATA_FOR_HUE_QUADRATURE['e_s']

    x = np.interp(hue, h_s, e_s)
    x = np.where(hue < 20.14, 0.856 - (hue / 20.14) * 0.056, x)
    x = np.where(hue > 237.53, 0.856 + 0.344 * (360 - hue) / (360 - 237.53), x)

    return x


def low_luminance_tritanopia_factor(L_A):
    """
    Returns the low luminance tritanopia factor :math:`F_t` from given adapting
    field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Parameters
    ----------
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    numeric or ndarray
        Low luminance tritanopia factor :math:`F_t`.

    Examples
    --------
    >>> low_luminance_tritanopia_factor(318.31)  # doctest: +ELLIPSIS
    0.9996859...
    """

    L_A = as_float_array(L_A)

    F_t = L_A / (L_A + 0.1)

    return F_t


def yellowness_blueness_response(C, e_s, N_c, N_cb, F_t):
    """
    Returns the yellowness / blueness response :math:`M_{yb}`.

    Parameters
    ----------
    C : array_like
        Colour difference signals :math:`C`.
    e_s : numeric or array_like
        Eccentricity factor :math:`e_s`.
    N_c : numeric or array_like
         Chromatic surround induction factor :math:`N_c`.
    N_cb : numeric or array_like
         Chromatic background induction factor :math:`N_{cb}`.
    F_t : numeric or array_like
        Low luminance tritanopia factor :math:`F_t`.

    Returns
    -------
    numeric or ndarray
        Yellowness / blueness response :math:`M_{yb}`.

    Examples
    --------
    >>> C = np.array([
    ...     -5.365865581996587e-05,
    ...     -0.000571699383647,
    ...     0.000625358039467
    ... ])
    >>> e_s = 1.110836504862630
    >>> N_c = 1.0
    >>> N_cb = 0.725000000000000
    >>> F_t = 0.99968593951195
    >>> yellowness_blueness_response(C, e_s, N_c, N_cb, F_t)
    ... # doctest: +ELLIPSIS
    -0.0082372...
    """

    _C_1, C_2, C_3 = tsplit(C)
    e_s = as_float_array(e_s)
    N_c = as_float_array(N_c)
    N_cb = as_float_array(N_cb)
    F_t = as_float_array(F_t)

    M_yb = (
        100 * (0.5 * (C_2 - C_3) / 4.5) * (e_s * (10 / 13) * N_c * N_cb * F_t))

    return M_yb


def redness_greenness_response(C, e_s, N_c, N_cb):
    """
    Returns the redness / greenness response :math:`M_{yb}`.

    Parameters
    ----------
    C : array_like
        Colour difference signals :math:`C`.
    e_s : numeric or array_like
        Eccentricity factor :math:`e_s`.
    N_c : numeric or array_like
         Chromatic surround induction factor :math:`N_c`.
    N_cb : numeric or array_like
         Chromatic background induction factor :math:`N_{cb}`.

    Returns
    -------
    numeric or ndarray
        Redness / greenness response :math:`M_{rg}`.

    Examples
    --------
    >>> C = np.array([
    ...     -5.365865581996587e-05,
    ...     -0.000571699383647,
    ...     0.000625358039467
    ... ])
    >>> e_s = 1.110836504862630
    >>> N_c = 1.0
    >>> N_cb = 0.725000000000000
    >>> redness_greenness_response(C, e_s, N_c, N_cb)  # doctest: +ELLIPSIS
    -0.0001044...
    """

    C_1, C_2, _C_3 = tsplit(C)
    e_s = as_float_array(e_s)
    N_c = as_float_array(N_c)
    N_cb = as_float_array(N_cb)

    M_rg = 100 * (C_1 - (C_2 / 11)) * (e_s * (10 / 13) * N_c * N_cb)

    return M_rg


def overall_chromatic_response(M_yb, M_rg):
    """
    Returns the overall chromatic response :math:`M`.

    Parameters
    ----------
    M_yb : numeric or array_like
         Yellowness / blueness response :math:`M_{yb}`.
    M_rg : numeric or array_like
         Redness / greenness response :math:`M_{rg}`.

    Returns
    -------
    numeric or ndarray
        Overall chromatic response :math:`M`.

    Examples
    --------
    >>> M_yb = -0.008237223618825
    >>> M_rg = -0.000104447583276
    >>> overall_chromatic_response(M_yb, M_rg)  # doctest: +ELLIPSIS
    0.0082378...
    """

    M_yb = as_float_array(M_yb)
    M_rg = as_float_array(M_rg)

    M = spow((M_yb ** 2) + (M_rg ** 2), 0.5)

    return M


def saturation_correlate(M, rgb_a):
    """
    Returns the *saturation* correlate :math:`s`.

    Parameters
    ----------
    M : numeric or array_like
         Overall chromatic response :math:`M`.
    rgb_a : array_like
        Adapted *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace
        array.

    Returns
    -------
    numeric or ndarray
        *Saturation* correlate :math:`s`.

    Examples
    --------
    >>> M = 0.008237885787274
    >>> rgb_a = np.array([6.89594549, 6.89599915, 6.89657085])
    >>> saturation_correlate(M, rgb_a)  # doctest: +ELLIPSIS
    0.0199093...
    """

    M = as_float_array(M)
    rgb_a = as_float_array(rgb_a)

    s = 50 * M / np.sum(rgb_a, axis=-1)

    return s


def achromatic_signal(L_AS, S, S_w, N_bb, A_a):
    """
    Returns the achromatic signal :math:`A`.

    Parameters
    ----------
    L_AS : numeric or array_like
        Scotopic luminance :math:`L_{AS}` of the illuminant.
    S : numeric or array_like
        Scotopic response :math:`S` to the stimulus.
    S_w : numeric or array_like
        Scotopic response :math:`S_w` for the reference white.
    N_bb : numeric or array_like
        Brightness background induction factor :math:`N_{bb}`.
    A_a: numeric or array_like
        Achromatic post adaptation signal of the stimulus :math:`A_a`.

    Returns
    -------
    numeric or ndarray
        Achromatic signal :math:`A`.

    Examples
    --------
    >>> L_AS = 769.9376286541402
    >>> S = 20.0
    >>> S_w = 100.0
    >>> N_bb = 0.725000000000000
    >>> A_a = 18.982718664838487
    >>> achromatic_signal(L_AS, S, S_w, N_bb, A_a)  # doctest: +ELLIPSIS
    15.5068546...
    """

    L_AS = as_float_array(L_AS)
    S = as_float_array(S)
    S_w = as_float_array(S_w)
    N_bb = as_float_array(N_bb)
    A_a = as_float_array(A_a)

    j = 0.00001 / ((5 * L_AS / 2.26) + 0.00001)

    # Computing scotopic luminance level adaptation factor :math:`F_{LS}`.
    F_LS = 3800 * (j ** 2) * (5 * L_AS / 2.26)
    F_LS += 0.2 * (spow(1 - (j ** 2), 0.4)) * (spow(5 * L_AS / 2.26, 1 / 6))

    # Computing cone bleach factors :math:`B_S`.
    B_S = 0.5 / (1 + 0.3 * spow((5 * L_AS / 2.26) * (S / S_w), 0.3))
    B_S += 0.5 / (1 + 5 * (5 * L_AS / 2.26))

    # Computing adapted scotopic signal :math:`A_S`.
    A_S = (f_n(F_LS * S / S_w) * 3.05 * B_S) + 0.3

    # Computing achromatic signal :math:`A`.
    A = N_bb * (A_a - 1 + A_S - 0.3 + np.sqrt((1 + (0.3 ** 2))))

    return A


def brightness_correlate(A, A_w, M, N_b):
    """
    Returns the *brightness* correlate :math:`Q`.

    Parameters
    ----------
    A : numeric or array_like
         Achromatic signal :math:`A`.
    A_w: numeric or array_like
        Achromatic post adaptation signal of the reference white :math:`A_w`.
    M : numeric or array_like
        Overall chromatic response :math:`M`.
    N_b : numeric or array_like
         Brightness surround induction factor :math:`N_b`.

    Returns
    -------
    numeric or ndarray
        *Brightness* correlate :math:`Q`.

    Examples
    --------
    >>> A = 15.506854623621885
    >>> A_w = 35.718916676317086
    >>> M = 0.008237885787274
    >>> N_b = 75.0
    >>> brightness_correlate(A, A_w, M, N_b)  # doctest: +ELLIPSIS
    22.2097654...
    """

    A = as_float_array(A)
    A_w = as_float_array(A_w)
    M = as_float_array(M)
    N_b = as_float_array(N_b)

    N_1 = (spow(7 * A_w, 0.5)) / (5.33 * spow(N_b, 0.13))
    N_2 = (7 * A_w * spow(N_b, 0.362)) / 200

    Q = spow(7 * (A + (M / 100)), 0.6) * N_1 - N_2

    return Q


def lightness_correlate(Y_b, Y_w, Q, Q_w):
    """
    Returns the *Lightness* correlate :math:`J`.

    Parameters
    ----------
    Y_b : numeric or array_like
         Tristimulus values :math:`Y_b` the background.
    Y_w : numeric or array_like
         Tristimulus values :math:`Y_b` the reference white.
    Q : numeric or array_like
        *Brightness* correlate :math:`Q` of the stimulus.
    Q_w : numeric or array_like
        *Brightness* correlate :math:`Q` of the reference white.

    Returns
    -------
    numeric or ndarray
        *Lightness* correlate :math:`J`.

    Examples
    --------
    >>> Y_b = 100.0
    >>> Y_w = 100.0
    >>> Q = 22.209765491265024
    >>> Q_w = 40.518065821226081
    >>> lightness_correlate(Y_b, Y_w, Q, Q_w)  # doctest: +ELLIPSIS
    30.0462678...
    """

    Y_b = as_float_array(Y_b)
    Y_w = as_float_array(Y_w)
    Q = as_float_array(Q)
    Q_w = as_float_array(Q_w)

    Z = 1 + spow(Y_b / Y_w, 0.5)
    J = 100 * spow(Q / Q_w, Z)

    return J


def chroma_correlate(s, Y_b, Y_w, Q, Q_w):
    """
    Returns the *chroma* correlate :math:`C_94`.

    Parameters
    ----------
    s : numeric or array_like
        *Saturation* correlate :math:`s`.
    Y_b : numeric or array_like
         Tristimulus values :math:`Y_b` the background.
    Y_w : numeric or array_like
         Tristimulus values :math:`Y_b` the reference white.
    Q : numeric or array_like
        *Brightness* correlate :math:`Q` of the stimulus.
    Q_w : numeric or array_like
        *Brightness* correlate :math:`Q` of the reference white.

    Returns
    -------
    numeric or ndarray
        *Chroma* correlate :math:`C_94`.

    Examples
    --------
    >>> s = 0.0199093206929
    >>> Y_b = 100.0
    >>> Y_w = 100.0
    >>> Q = 22.209765491265024
    >>> Q_w = 40.518065821226081
    >>> chroma_correlate(s, Y_b, Y_w, Q, Q_w)  # doctest: +ELLIPSIS
    0.1210508...
    """

    s = as_float_array(s)
    Y_b = as_float_array(Y_b)
    Y_w = as_float_array(Y_w)
    Q = as_float_array(Q)
    Q_w = as_float_array(Q_w)

    C_94 = (2.44 * spow(s, 0.69) * (spow(Q / Q_w, Y_b / Y_w)) *
            (1.64 - spow(0.29, Y_b / Y_w)))

    return C_94


def colourfulness_correlate(F_L, C_94):
    """
    Returns the *colourfulness* correlate :math:`M_94`.

    Parameters
    ----------
    F_L : numeric or array_like
        Luminance adaptation factor :math:`F_L`.
    C_94 : numeric
        *Chroma* correlate :math:`C_94`.

    Returns
    -------
    numeric
        *Colourfulness* correlate :math:`M_94`.

    Examples
    --------
    >>> F_L = 1.16754446414718
    >>> C_94 = 0.121050839936176
    >>> colourfulness_correlate(F_L, C_94)  # doctest: +ELLIPSIS
    0.1238964...
    """

    F_L = as_float_array(F_L)
    C_94 = as_float_array(C_94)

    M_94 = spow(F_L, 0.15) * C_94

    return M_94
