# -*- coding: utf-8 -*-
"""
Nayatani (1995) Colour Appearance Model
=======================================

Defines the *Nayatani (1995)* colour appearance model objects:

-   :class:`colour.CAM_Specification_Nayatani95`
-   :func:`colour.XYZ_to_Nayatani95`

References
----------
-   :cite:`Fairchild2013ba` : Fairchild, M. D. (2013). The Nayatani et al.
    Model. In Color Appearance Models (3rd ed., pp. 4810-5085). Wiley.
    ISBN:B00DAYO8E2
-   :cite:`Nayatani1995a` : Nayatani, Y., Sobagaki, H., & Yano, K. H. T.
    (1995). Lightness dependency of chroma scales of a nonlinear
    color-appearance model and its latest formulation. Color Research &
    Application, 20(3), 156-167. doi:10.1002/col.5080200305
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union

from colour.algebra import spow, vector_dot
from colour.adaptation.cie1994 import (
    MATRIX_XYZ_TO_RGB_CIE1994,
    beta_1,
    exponential_factors,
    intermediate_values,
)
from colour.models import XYZ_to_xy
from colour.utilities import (
    MixinDataclassArray,
    as_float,
    as_float_array,
    from_range_degrees,
    to_domain_100,
    tsplit,
    tstack,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_XYZ_TO_RGB_NAYATANI95',
    'CAM_ReferenceSpecification_Nayatani95',
    'CAM_Specification_Nayatani95',
    'XYZ_to_Nayatani95',
    'illuminance_to_luminance',
    'XYZ_to_RGB_Nayatani95',
    'scaling_coefficient',
    'achromatic_response',
    'tritanopic_response',
    'protanopic_response',
    'brightness_correlate',
    'ideal_white_brightness_correlate',
    'achromatic_lightness_correlate',
    'normalised_achromatic_lightness_correlate',
    'hue_angle',
    'saturation_components',
    'saturation_correlate',
    'chroma_components',
    'chroma_correlate',
    'colourfulness_components',
    'colourfulness_correlate',
    'chromatic_strength_function',
]

MATRIX_XYZ_TO_RGB_NAYATANI95 = MATRIX_XYZ_TO_RGB_CIE1994
"""
*Nayatani (1995)* colour appearance model *CIE XYZ* tristimulus values to cone
responses matrix.

MATRIX_XYZ_TO_RGB_NAYATANI95 : array_like, (3, 3)
"""


@dataclass
class CAM_ReferenceSpecification_Nayatani95(MixinDataclassArray):
    """
    Defines the *Nayatani (1995)* colour appearance model reference
    specification.

    This specification has field names consistent with *Fairchild (2013)*
    reference.

    Parameters
    ----------
    L_star_P : numeric or array_like
        Correlate of *achromatic Lightness* :math:`L_p^\\star`.
    C : numeric or array_like
        Correlate of *chroma* :math:`C`.
    theta : numeric or array_like
        *Hue* angle :math:`\\theta` in degrees.
    S : numeric or array_like
        Correlate of *saturation* :math:`S`.
    B_r : numeric or array_like
        Correlate of *brightness* :math:`B_r`.
    M : numeric or array_like
        Correlate of *colourfulness* :math:`M`.
    H : numeric or array_like
        *Hue* :math:`h` quadrature :math:`H`.
    H_C : numeric or array_like
        *Hue* :math:`h` composition :math:`H_C`.
    L_star_N : numeric or array_like
        Correlate of *normalised achromatic Lightness* :math:`L_n^\\star`.

    References
    ----------
    :cite:`Fairchild2013ba`, :cite:`Nayatani1995a`
    """

    L_star_P: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    C: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    theta: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    S: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    B_r: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    M: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    H: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    H_C: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    L_star_N: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)


@dataclass
class CAM_Specification_Nayatani95(MixinDataclassArray):
    """
    Defines the *Nayatani (1995)* colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    *Fairchild (2013)* reference.

    Parameters
    ----------
    L_star_P : numeric or array_like
        Correlate of *achromatic Lightness* :math:`L_p^\\star`.
    C : numeric or array_like
        Correlate of *chroma* :math:`C`.
    h : numeric or array_like
        *Hue* angle :math:`\\theta` in degrees.
    s : numeric or array_like
        Correlate of *saturation* :math:`S`.
    Q : numeric or array_like
        Correlate of *brightness* :math:`B_r`.
    M : numeric or array_like
        Correlate of *colourfulness* :math:`M`.
    H : numeric or array_like
        *Hue* :math:`h` quadrature :math:`H`.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H_C`.
    L_star_N : numeric or array_like
        Correlate of *normalised achromatic Lightness* :math:`L_n^\\star`.

    Notes
    -----
    -   This specification is the one used in the current model implementation.

    References
    ----------
    :cite:`Fairchild2013ba`, :cite:`Nayatani1995a`
    """

    L_star_P: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    C: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    h: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    s: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    Q: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    M: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    H: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    HC: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    L_star_N: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)


def XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, n=1):
    """
    Computes the *Nayatani (1995)* colour appearance model correlates.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_n : array_like
        *CIE XYZ* tristimulus values of reference white.
    Y_o : numeric or array_like
        Luminance factor :math:`Y_o` of achromatic background as percentage
        normalised to domain [0.18, 1.0] in **'Reference'** domain-range scale.
    E_o : numeric or array_like
        Illuminance :math:`E_o` of the viewing field in lux.
    E_or : numeric or array_like
        Normalising illuminance :math:`E_{or}` in lux usually normalised to
        domain [1000, 3000].
    n : numeric or array_like, optional
        Noise term used in the non-linear chromatic adaptation model.

    Returns
    -------
    CAM_Specification_Nayatani95
        *Nayatani (1995)* colour appearance model specification.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_n``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------------------------------+-----------------------\
+---------------+
    | **Range**                          | **Scale - Reference** \
| **Scale - 1** |
    +====================================+=======================\
+===============+
    | ``CAM_Specification_Nayatani95.h`` | [0, 360]              \
| [0, 1]        |
    +------------------------------------+-----------------------\
+---------------+

    References
    ----------
    :cite:`Fairchild2013ba`, :cite:`Nayatani1995a`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_n = np.array([95.05, 100.00, 108.88])
    >>> Y_o = 20.0
    >>> E_o = 5000.0
    >>> E_or = 1000.0
    >>> XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)  # doctest: +ELLIPSIS
    CAM_Specification_Nayatani95(L_star_P=49.9998829..., C=0.0133550..., \
h=257.5232268..., s=0.0133550..., Q=62.6266734..., M=0.0167262..., \
H=None, HC=None, L_star_N=50.0039154...)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_n = to_domain_100(XYZ_n)
    Y_o = as_float_array(Y_o)
    E_o = as_float_array(E_o)
    E_or = as_float_array(E_or)

    # Computing adapting luminance :math:`L_o` and normalising luminance
    # :math:`L_{or}` in in :math:`cd/m^2`.
    # L_o = illuminance_to_luminance(E_o, Y_o)
    L_or = illuminance_to_luminance(E_or, Y_o)

    # Computing :math:`\\xi` :math:`\\eta`, :math:`\\zeta` values.
    xez = intermediate_values(XYZ_to_xy(XYZ_n / 100))
    xi, eta, _zeta = tsplit(xez)

    # Computing adapting field cone responses.
    RGB_o = ((
        (Y_o[..., np.newaxis] * E_o[..., np.newaxis]) / (100 * np.pi)) * xez)

    # Computing stimulus cone responses.
    RGB = XYZ_to_RGB_Nayatani95(XYZ)
    R, G, _B = tsplit(RGB)

    # Computing exponential factors of the chromatic adaptation.
    bRGB_o = exponential_factors(RGB_o)
    bL_or = beta_1(L_or)

    # Computing scaling coefficients :math:`e(R)` and :math:`e(G)`
    eR = scaling_coefficient(R, xi)
    eG = scaling_coefficient(G, eta)

    # Computing opponent colour dimensions.
    # Computing achromatic response :math:`Q`:
    Q_response = achromatic_response(RGB, bRGB_o, xez, bL_or, eR, eG, n)

    # Computing tritanopic response :math:`t`:
    t_response = tritanopic_response(RGB, bRGB_o, xez, n)

    # Computing protanopic response :math:`p`:
    p_response = protanopic_response(RGB, bRGB_o, xez, n)

    # Computing the correlate of *brightness* :math:`B_r`.
    B_r = brightness_correlate(bRGB_o, bL_or, Q_response)

    # Computing *brightness* :math:`B_{rw}` of ideal white.
    brightness_ideal_white = ideal_white_brightness_correlate(
        bRGB_o, xez, bL_or, n)

    # Computing the correlate of achromatic *Lightness* :math:`L_p^\\star`.
    L_star_P = (achromatic_lightness_correlate(Q_response))

    # Computing the correlate of normalised achromatic *Lightness*
    # :math:`L_n^\\star`.
    L_star_N = (normalised_achromatic_lightness_correlate(
        B_r, brightness_ideal_white))

    # Computing the *hue* angle :math:`\\theta`.
    theta = hue_angle(p_response, t_response)
    # TODO: Implement hue quadrature & composition computation.

    # Computing the correlate of *saturation* :math:`S`.
    S_RG, S_YB = tsplit(
        saturation_components(theta, bL_or, t_response, p_response))
    S = saturation_correlate(S_RG, S_YB)

    # Computing the correlate of *chroma* :math:`C`.
    # C_RG, C_YB = tsplit(chroma_components(L_star_P, S_RG, S_YB))
    C = chroma_correlate(L_star_P, S)

    # Computing the correlate of *colourfulness* :math:`M`.
    # TODO: Investigate components usage.
    # M_RG, M_YB = tsplit(colourfulness_components(C_RG, C_YB,
    # brightness_ideal_white))
    M = colourfulness_correlate(C, brightness_ideal_white)

    return CAM_Specification_Nayatani95(
        L_star_P,
        C,
        as_float(from_range_degrees(theta)),
        S,
        B_r,
        M,
        None,
        None,
        L_star_N,
    )


def illuminance_to_luminance(E, Y_f):
    """
    Converts given *illuminance* :math:`E` value in lux to *luminance* in
    :math:`cd/m^2`.

    Parameters
    ----------
    E : numeric or array_like
        *Illuminance* :math:`E` in lux.
    Y_f : numeric or array_like
        *Luminance* factor :math:`Y_f` in :math:`cd/m^2`.

    Returns
    -------
    numeric or ndarray
        *Luminance* :math:`Y` in :math:`cd/m^2`.

    Examples
    --------
    >>> illuminance_to_luminance(5000.0, 20.0)  # doctest: +ELLIPSIS
    318.3098861...
    """

    E = as_float_array(E)
    Y_f = as_float_array(Y_f)

    return Y_f * E / (100 * np.pi)


def XYZ_to_RGB_Nayatani95(XYZ):
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
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_to_RGB_Nayatani95(XYZ)  # doctest: +ELLIPSIS
    array([ 20.0005206...,  19.999783 ...,  19.9988316...])
    """

    return vector_dot(MATRIX_XYZ_TO_RGB_NAYATANI95, XYZ)


def scaling_coefficient(x, y):
    """
    Returns the scaling coefficient :math:`e(R)` or :math:`e(G)`.

    Parameters
    ----------
    x: numeric or array_like
        Cone response.
    y: numeric or array_like
        Intermediate value.

    Returns
    -------
    numeric or ndarray
        Scaling coefficient :math:`e(R)` or :math:`e(G)`.

    Examples
    --------
    >>> x = 20.000520600000002
    >>> y = 1.000042192
    >>> scaling_coefficient(x, y)
    1.0
    """

    x = as_float_array(x)
    y = as_float_array(y)

    return as_float(np.where(x >= (20 * y), 1.758, 1))


def achromatic_response(RGB, bRGB_o, xez, bL_or, eR, eG, n=1):
    """
    Returns the achromatic response :math:`Q` from given stimulus cone
    responses.

    Parameters
    ----------
    RGB: ndarray
         Stimulus cone responses.
    bRGB_o: ndarray
         Chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
         :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)`.
    xez: ndarray
        Intermediate values :math:`\\xi`, :math:`\\eta`, :math:`\\zeta`.
    bL_or: numeric or array_like
         Normalising chromatic adaptation exponential factor
         :math:`\\beta_1(B_or)`.
    eR: numeric or array_like
         Scaling coefficient :math:`e(R)`.
    eG: numeric or array_like
         Scaling coefficient :math:`e(G)`.
    n : numeric or array_like, optional
        Noise term used in the non-linear chromatic adaptation model.

    Returns
    -------
    numeric or ndarray
        Achromatic response :math:`Q`.

    Examples
    --------
    >>> RGB = np.array([20.00052060, 19.99978300, 19.99883160])
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> xez = np.array([1.00004219, 0.99998001, 0.99975794])
    >>> bL_or = 3.681021495604089
    >>> eR = 1.0
    >>> eG = 1.758
    >>> n = 1.0
    >>> achromatic_response(RGB, bRGB_o, xez, bL_or, eR, eG, n)
    ... # doctest: +ELLIPSIS
    -0.0001169...
    """

    R, G, _B = tsplit(RGB)
    bR_o, bG_o, _bB_o = tsplit(bRGB_o)
    xi, eta, _zeta = tsplit(xez)
    bL_or = as_float_array(bL_or)
    eR = as_float_array(eR)
    eG = as_float_array(eG)

    Q = (2 / 3) * bR_o * eR * np.log10((R + n) / (20 * xi + n))
    Q += (1 / 3) * bG_o * eG * np.log10((G + n) / (20 * eta + n))
    Q *= 41.69 / bL_or

    return Q


def tritanopic_response(RGB, bRGB_o, xez, n):
    """
    Returns the tritanopic response :math:`t` from given stimulus cone
    responses.

    Parameters
    ----------
    RGB: ndarray
         Stimulus cone responses.
    bRGB_o: ndarray
         Chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
         :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)`.
    xez: ndarray
        Intermediate values :math:`\\xi`, :math:`\\eta`, :math:`\\zeta`.
    n : numeric or array_like, optional
        Noise term used in the non-linear chromatic adaptation model.

    Returns
    -------
    numeric or ndarray
        Tritanopic response :math:`t`.

    Examples
    --------
    >>> RGB = np.array([20.00052060, 19.99978300, 19.99883160])
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> xez = np.array([1.00004219, 0.99998001, 0.99975794])
    >>> n = 1.0
    >>> tritanopic_response(RGB, bRGB_o, xez, n)  # doctest: +ELLIPSIS
    -1.7703650...e-05
    """

    R, G, B = tsplit(RGB)
    bR_o, bG_o, bB_o = tsplit(bRGB_o)
    xi, eta, zeta = tsplit(xez)

    t = bR_o * np.log10((R + n) / (20 * xi + n))
    t += -(12 / 11) * bG_o * np.log10((G + n) / (20 * eta + n))
    t += (1 / 11) * bB_o * np.log10((B + n) / (20 * zeta + n))

    return t


def protanopic_response(RGB, bRGB_o, xez, n):
    """
    Returns the protanopic response :math:`p` from given stimulus cone
    responses.

    Parameters
    ----------
    RGB: ndarray
         Stimulus cone responses.
    bRGB_o: ndarray
         Chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
         :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)`.
    xez: ndarray
        Intermediate values :math:`\\xi`, :math:`\\eta`, :math:`\\zeta`.
    n : numeric or array_like, optional
        Noise term used in the non-linear chromatic adaptation model.

    Returns
    -------
    numeric or ndarray
        Protanopic response :math:`p`.

    Examples
    --------
    >>> RGB = np.array([20.00052060, 19.99978300, 19.99883160])
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> xez = np.array([1.00004219, 0.99998001, 0.99975794])
    >>> n = 1.0
    >>> protanopic_response(RGB, bRGB_o, xez, n)  # doctest: +ELLIPSIS
    -8.0021426...e-05
    """

    R, G, B = tsplit(RGB)
    bR_o, bG_o, bB_o = tsplit(bRGB_o)
    xi, eta, zeta = tsplit(xez)

    p = (1 / 9) * bR_o * np.log10((R + n) / (20 * xi + n))
    p += (1 / 9) * bG_o * np.log10((G + n) / (20 * eta + n))
    p += -(2 / 9) * bB_o * np.log10((B + n) / (20 * zeta + n))

    return p


def brightness_correlate(bRGB_o, bL_or, Q):
    """
    Returns the *brightness* correlate :math:`B_r`.

    Parameters
    ----------
    bRGB_o: ndarray
         Chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
         :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)`.
    bL_or: numeric or array_like
         Normalising chromatic adaptation exponential factor
         :math:`\\beta_1(B_or)`.
    Q : numeric or array_like
        Achromatic response :math:`Q`.
    Returns
    -------
    numeric or ndarray
        *Brightness* correlate :math:`B_r`.

    Examples
    --------
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> bL_or = 3.681021495604089
    >>> Q = -0.000117024294955
    >>> brightness_correlate(bRGB_o, bL_or, Q)  # doctest: +ELLIPSIS
    62.6266734...
    """

    bR_o, bG_o, _bB_o = tsplit(bRGB_o)
    bL_or = as_float_array(bL_or)
    Q = as_float_array(Q)

    B_r = (50 / bL_or) * ((2 / 3) * bR_o + (1 / 3) * bG_o) + Q

    return B_r


def ideal_white_brightness_correlate(bRGB_o, xez, bL_or, n):
    """
    Returns the ideal white *brightness* correlate :math:`B_{rw}`.

    Parameters
    ----------
    bRGB_o: ndarray
         Chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
         :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)`.
    xez: ndarray
        Intermediate values :math:`\\xi`, :math:`\\eta`, :math:`\\zeta`.
    bL_or: numeric or array_like
         Normalising chromatic adaptation exponential factor
         :math:`\\beta_1(B_or)`.
    n : numeric or array_like, optional
        Noise term used in the non-linear chromatic adaptation model.

    Returns
    -------
    numeric or ndarray
        Ideal white *brightness* correlate :math:`B_{rw}`.

    Examples
    --------
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> xez = np.array([1.00004219, 0.99998001, 0.99975794])
    >>> bL_or = 3.681021495604089
    >>> n = 1.0
    >>> ideal_white_brightness_correlate(bRGB_o, xez, bL_or, n)
    ... # doctest: +ELLIPSIS
    125.2435392...
    """

    bR_o, bG_o, _bB_o = tsplit(bRGB_o)
    xi, eta, _zeta = tsplit(xez)
    bL_or = as_float_array(bL_or)

    B_rw = (2 / 3) * bR_o * 1.758 * np.log10((100 * xi + n) / (20 * xi + n))
    B_rw += (1 / 3) * bG_o * 1.758 * np.log10((100 * eta + n) / (20 * eta + n))
    B_rw *= 41.69 / bL_or
    B_rw += (50 / bL_or) * (2 / 3) * bR_o
    B_rw += (50 / bL_or) * (1 / 3) * bG_o

    return B_rw


def achromatic_lightness_correlate(Q):
    """
    Returns the *achromatic Lightness* correlate :math:`L_p^\\star`.

    Parameters
    ----------
    Q : numeric or array_like
        Achromatic response :math:`Q`.

    Returns
    -------
    numeric or ndarray
        *Achromatic Lightness* correlate :math:`L_p^\\star`.

    Examples
    --------
    >>> Q = -0.000117024294955
    >>> achromatic_lightness_correlate(Q)  # doctest: +ELLIPSIS
    49.9998829...
    """

    Q = as_float_array(Q)

    return Q + 50


def normalised_achromatic_lightness_correlate(B_r, B_rw):
    """
    Returns the *normalised achromatic Lightness* correlate :math:`L_n^\\star`.

    Parameters
    ----------
    B_r : numeric or array_like
        *Brightness* correlate :math:`B_r`.
    B_rw : numeric or array_like
        Ideal white *brightness* correlate :math:`B_{rw}`.

    Returns
    -------
    numeric or ndarray
        *Normalised achromatic Lightness* correlate :math:`L_n^\\star`.

    Examples
    --------
    >>> B_r = 62.626673467230766
    >>> B_rw = 125.24353925846037
    >>> normalised_achromatic_lightness_correlate(B_r, B_rw)
    ... # doctest: +ELLIPSIS
    50.0039154...
    """

    B_r = as_float_array(B_r)
    B_rw = as_float_array(B_rw)

    return 100 * (B_r / B_rw)


def hue_angle(p, t):
    """
    Returns the *hue* angle :math:`h` in degrees.

    Parameters
    ----------
    p : numeric or array_like
        Protanopic response :math:`p`.
    t : numeric or array_like
        Tritanopic response :math:`t`.

    Returns
    -------
    numeric or ndarray
        *Hue* angle :math:`h` in degrees.

    Examples
    --------
    >>> p = -8.002142682085493e-05
    >>> t = -0.000017703650669
    >>> hue_angle(p, t)  # doctest: +ELLIPSIS
    257.5250300...
    """

    p = as_float_array(p)
    t = as_float_array(t)

    h_L = np.degrees(np.arctan2(p, t)) % 360

    return h_L


def chromatic_strength_function(theta):
    """
    Defines the chromatic strength function :math:`E_s(\\theta)` used to
    correct saturation scale as function of hue angle :math:`\\theta` in
    degrees.

    Parameters
    ----------
    theta : numeric or array_like
        Hue angle :math:`\\theta` in degrees.

    Returns
    -------
    numeric or ndarray
        Corrected saturation scale.

    Examples
    --------
    >>> h = 257.52322689806243
    >>> chromatic_strength_function(h)  # doctest: +ELLIPSIS
    1.2267869...
    """

    theta = np.radians(theta)

    E_s = 0.9394
    E_s += -0.2478 * np.sin(1 * theta)
    E_s += -0.0743 * np.sin(2 * theta)
    E_s += +0.0666 * np.sin(3 * theta)
    E_s += -0.0186 * np.sin(4 * theta)
    E_s += -0.0055 * np.cos(1 * theta)
    E_s += -0.0521 * np.cos(2 * theta)
    E_s += -0.0573 * np.cos(3 * theta)
    E_s += -0.0061 * np.cos(4 * theta)

    return E_s


def saturation_components(h, bL_or, t, p):
    """
    Returns the *saturation* components :math:`S_{RG}` and :math:`S_{YB}`.

    Parameters
    ----------
    h: numeric or array_like
        Correlate of *hue* :math:`h` in degrees.
    bL_or: numeric or array_like
         Normalising chromatic adaptation exponential factor
         :math:`\\beta_1(B_or)`.
    t : numeric or array_like
        Tritanopic response :math:`t`.
    p : numeric or array_like
        Protanopic response :math:`p`.

    Returns
    -------
    numeric or ndarray
        *Saturation* components :math:`S_{RG}` and :math:`S_{YB}`.

    Examples
    --------
    >>> h = 257.52322689806243
    >>> bL_or = 3.681021495604089
    >>> t = -0.000017706764677
    >>> p = -0.000080023561356
    >>> saturation_components(h, bL_or, t, p)  # doctest: +ELLIPSIS
    array([-0.0028852..., -0.0130396...])
    """

    h = as_float_array(h)
    bL_or = as_float_array(bL_or)
    t = as_float_array(t)
    p = as_float_array(p)

    E_s = chromatic_strength_function(h)
    S_RG = (488.93 / bL_or) * E_s * t
    S_YB = (488.93 / bL_or) * E_s * p

    return tstack([S_RG, S_YB])


def saturation_correlate(S_RG, S_YB):
    """
    Returns the correlate of *saturation* :math:`S`.

    Parameters
    ----------
    S_RG : numeric or array_like
        *Saturation* component :math:`S_{RG}`.
    S_YB : numeric or array_like
        *Saturation* component :math:`S_{YB}`.

    Returns
    -------
    numeric or ndarray
        Correlate of *saturation* :math:`S`.

    Examples
    --------
    >>> S_RG = -0.002885271638197
    >>> S_YB = -0.013039632941332
    >>> saturation_correlate(S_RG, S_YB)  # doctest: +ELLIPSIS
    0.0133550...
    """

    S_RG = as_float_array(S_RG)
    S_YB = as_float_array(S_YB)

    S = np.hypot(S_RG, S_YB)

    return S


def chroma_components(L_star_P, S_RG, S_YB):
    """
    Returns the *chroma* components :math:`C_{RG}` and :math:`C_{YB}`.

    Parameters
    ----------
    L_star_P : numeric or array_like
        *Achromatic Lightness* correlate :math:`L_p^\\star`.
    S_RG : numeric or array_like
        *Saturation* component :math:`S_{RG}`.
    S_YB : numeric or array_like
        *Saturation* component :math:`S_{YB}`.

    Returns
    -------
    ndarray
        *Chroma* components :math:`C_{RG}` and :math:`C_{YB}`.

    Examples
    --------
    >>> L_star_P = 49.99988297570504
    >>> S_RG = -0.002885271638197
    >>> S_YB = -0.013039632941332
    >>> chroma_components(L_star_P, S_RG, S_YB)  # doctest: +ELLIPSIS
    array([-0.00288527, -0.01303961])
    """

    L_star_P = as_float_array(L_star_P)
    S_RG = as_float_array(S_RG)
    S_YB = as_float_array(S_YB)

    C_RG = spow(L_star_P / 50, 0.7) * S_RG
    C_YB = spow(L_star_P / 50, 0.7) * S_YB

    return tstack([C_RG, C_YB])


def chroma_correlate(L_star_P, S):
    """
    Returns the correlate of *chroma* :math:`C`.

    Parameters
    ----------
    L_star_P : numeric or array_like
        *Achromatic Lightness* correlate :math:`L_p^\\star`.
    S : numeric or array_like
        Correlate of *saturation* :math:`S`.

    Returns
    -------
    numeric or ndarray
        Correlate of *chroma* :math:`C`.

    Examples
    --------
    >>> L_star_P = 49.99988297570504
    >>> S = 0.013355029751778
    >>> chroma_correlate(L_star_P, S)  # doctest: +ELLIPSIS
    0.0133550...
    """

    L_star_P = as_float_array(L_star_P)
    S = as_float_array(S)

    C = spow(L_star_P / 50, 0.7) * S

    return C


def colourfulness_components(C_RG, C_YB, B_rw):
    """
    Returns the *colourfulness* components :math:`M_{RG}` and :math:`M_{YB}`.

    Parameters
    ----------
    C_RG : numeric or array_like
        *Chroma* component :math:`C_{RG}`.
    C_YB : numeric or array_like
        *Chroma* component :math:`C_{YB}`.
    B_rw : numeric or array_like
        Ideal white *brightness* correlate :math:`B_{rw}`.

    Returns
    -------
    numeric or ndarray
        *Colourfulness* components :math:`M_{RG}` and :math:`M_{YB}`.

    Examples
    --------
    >>> C_RG = -0.002885271638197
    >>> C_YB = -0.013039632941332
    >>> B_rw = 125.24353925846037
    >>> colourfulness_components(C_RG, C_YB, B_rw)  # doctest: +ELLIPSIS
    (-0.0036136..., -0.0163312...)
    """

    C_RG = as_float_array(C_RG)
    C_YB = as_float_array(C_YB)
    B_rw = as_float_array(B_rw)

    M_RG = C_RG * B_rw / 100
    M_YB = C_YB * B_rw / 100

    return M_RG, M_YB


def colourfulness_correlate(C, B_rw):
    """
    Returns the correlate of *colourfulness* :math:`M`.

    Parameters
    ----------
    C : numeric or array_like
        Correlate of *chroma* :math:`C`.
    B_rw : numeric or array_like
        Ideal white *brightness* correlate :math:`B_{rw}`.

    Returns
    -------
    numeric or ndarray
        Correlate of *colourfulness* :math:`M`.

    Examples
    --------
    >>> C = 0.013355007871689
    >>> B_rw = 125.24353925846037
    >>> colourfulness_correlate(C, B_rw)  # doctest: +ELLIPSIS
    0.0167262...
    """

    C = as_float_array(C)
    B_rw = as_float_array(B_rw)

    M = C * B_rw / 100

    return M
