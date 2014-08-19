#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nayatani (1995) Colour Appearance Model
=======================================

Defines *Nayatani (1995)* colour appearance model objects:

-   :func:`Nayatani95_Specification`
-   :func:`XYZ_to_Nayatani95`

References
----------
.. [1]  **Mark D. Fairchild**, *Color Appearance Models, 3nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published June 2013, ASIN: B00DAYO8E2,
        Locations 4810-5085.
.. [2]  **Y. Nayatani, H. Sobagaki & K. H. T. Yano**,
        *Lightness dependency of chroma scales of a nonlinear color-appearance
        model and its latest formulation*,
        *Color Research & Application, Volume 20, Issue 3, pages 156–167,
        June 1995*
"""

from __future__ import division, unicode_literals

import math
import numpy as np
from collections import namedtuple
from colour.models import XYZ_to_xy

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['NAYATANI95_XYZ_TO_RGB_MATRIX',
           'Nayatani95_Specification',
           'XYZ_to_Nayatani95',
           'illuminance_to_luminance',
           'get_intermediate_values',
           'XYZ_to_RGB_Nayatani95',
           'get_beta_1',
           'get_beta_2',
           'get_chromatic_adaptation_exponential_factors',
           'get_scaling_coefficient',
           'get_achromatic_response',
           'get_tritanopic_response',
           'get_protanopic_response',
           'get_brightness_correlate',
           'get_ideal_white_brightness_correlate',
           'get_achromatic_lightness_correlate',
           'get_normalised_achromatic_lightness_correlate',
           'get_hue_angle',
           'get_saturation_components',
           'get_saturation_correlate',
           'get_chroma_components',
           'get_chroma_correlate',
           'get_colourfulness_components',
           'get_colourfulness_correlate',
           'chromatic_strength_function', ]

NAYATANI95_XYZ_TO_RGB_MATRIX = np.array(
    [[0.40024, 0.70760, -0.08081],
     [-0.22630, 1.16532, 0.04570],
     [0.00000, 0.00000, 0.91822]])

Nayatani95_Specification = namedtuple(
    'Nayatani95_Specification',
    ('B_r', 'L_star_P', 'L_star_N', 'theta', 'S', 'C', 'M'))
"""
Defines the *Nayatani (1995)* colour appearance model specification.

Parameters
----------
B_r : float
    Correlate of *brightness* :math:`B_r`.
L_star_P : float
    Correlate of *achromatic Lightness* :math:`L_p^\star`.
L_star_N : float
    Correlate of *normalised achromatic Lightness* :math:`L_n^\star`.
theta : float
    *Hue* angle :math:`\\theta` in degrees.
S : float
    Correlate of *saturation* :math:`S`.
C : float
    Correlate of *chroma* :math:`C`.
M : float
    Correlate of *colourfulness* :math:`M`.
"""


def XYZ_to_Nayatani95(XYZ,
                      XYZ_n,
                      Y_o,
                      E_o,
                      E_or,
                      n=1):
    """
    Computes the *Nayatani (1995)* colour appearance model correlates.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    XYZ_n : array_like, (3,)
        *CIE XYZ* colourspace matrix of reference white in domain [0, 100].
    Y_o : float
        Luminance factor :math:`Y_o` of achromatic background as percentage in
        domain [0.18,]
    E_o : float
        Illuminance :math:`E_o` of the viewing field in lux.
    E_or : float
        Normalising illuminance :math:`E_{or}` in lux usually in domain
        [1000, 3000]
    n : float, optional
        Noise term used in the non linear chromatic adaptation model.

    Returns
    -------
    Nayatani95_Specification
        *Nayatani (1995)* colour appearance model specification.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    -   Input *CIE XYZ_n* colourspace matrix is in domain [0, 100].

    Raises
    ------
    ValueError
        If Luminance factor :math:`Y_o` is not greater or equal than 18%.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20, 21.78])
    >>> XYZ_n = np.array([95.05, 100, 108.88])
    >>> Y_o = 20.0
    >>> E_o = 5000.0
    >>> E_or = 1000.0
    >>> colour.XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)
    Nayatani95_Specification(B_r=62.626673450940629, L_star_P=49.999882975705042, L_star_N=50.003915441089511, theta=257.52322689806243, S=0.013355029751777615, C=0.013355007871688761, M=0.016726284522665977)
    """

    if np.any(Y_o < 0.18):
        raise ValueError(
            'Luminance factor "Y_o" of achromatic background must '
            'be greater or equal than 18%!')

    X, Y, Z = np.ravel(XYZ)

    # Computing adapting luminance :math:`L_o` and normalising luminance
    # :math:`L_{or}` in in :math:`cd/m^2`.
    L_o = illuminance_to_luminance(E_o, Y_o)
    L_or = illuminance_to_luminance(E_or, Y_o)

    # Computing :math:`\xi`, :math:`\eta`, :math:`\zeta` values.
    xi, eta, zeta = x_e_z = get_intermediate_values(XYZ_n)

    # Computing adapting field cone responses.
    RGB_o = ((Y_o * E_o) / (100 * np.pi)) * x_e_z

    # Computing stimulus cone responses.
    R, G, B = RGB = XYZ_to_RGB_Nayatani95(np.array([X, Y, Z]))

    # Computing exponential factors of the chromatic adaptation.
    bRGB_o = get_chromatic_adaptation_exponential_factors(RGB_o)
    bL_or = get_beta_1(L_or)

    # Computing scaling coefficients :math:`e(R)` and :math:`e(G)`
    eR = get_scaling_coefficient(R, xi)
    eG = get_scaling_coefficient(G, eta)

    # Computing opponent colour dimensions.
    # Computing achromatic response :math:`Q`:
    achromatic_response = get_achromatic_response(RGB, bRGB_o, x_e_z,
                                                  bL_or, eR, eG, n)

    # Computing tritanopic response :math:`t`:
    tritanopic_response = get_tritanopic_response(RGB, bRGB_o, x_e_z, n)

    protanopic_response = get_protanopic_response(RGB, bRGB_o, x_e_z, n)

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`B_r`.
    # -------------------------------------------------------------------------
    brightness = get_brightness_correlate(bRGB_o, bL_or, achromatic_response)

    # Computing *brightness* :math:`B_{rw}` of ideal white.
    brightness_ideal_white = get_ideal_white_brightness_correlate(bRGB_o,
                                                                  x_e_z,
                                                                  bL_or,
                                                                  n)


    # -------------------------------------------------------------------------
    # Computing the correlate of achromatic *Lightness* :math:`L_p^\star`.
    # -------------------------------------------------------------------------
    lightness_achromatic = (
        get_achromatic_lightness_correlate(achromatic_response))

    # -------------------------------------------------------------------------
    # Computing the correlate of normalised achromatic *Lightness*
    # :math:`L_n^\star`.
    # -------------------------------------------------------------------------
    lightness_achromatic_normalised = (
        get_normalised_achromatic_lightness_correlate(brightness,
                                                      brightness_ideal_white))

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`\\theta`.
    # -------------------------------------------------------------------------
    hue = get_hue_angle(protanopic_response, tritanopic_response)
    # TODO: Implement hue quadrature & composition computation.

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`S`.
    # -------------------------------------------------------------------------
    S_RG, S_YB = get_saturation_components(hue, bL_or,
                                           tritanopic_response,
                                           protanopic_response)
    saturation = get_saturation_correlate(S_RG, S_YB)

    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`C`.
    # -------------------------------------------------------------------------
    C_RG, C_YB = get_chroma_components(lightness_achromatic, S_RG, S_YB)
    chroma = get_chroma_correlate(lightness_achromatic, saturation)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`M`.
    # -------------------------------------------------------------------------
    # TODO: Investigate components usage?
    M_RG, M_YB = get_colourfulness_components(C_RG, C_YB,
                                              brightness_ideal_white)
    colorfulness = get_colourfulness_correlate(chroma, brightness_ideal_white)

    return Nayatani95_Specification(brightness,
                                    lightness_achromatic,
                                    lightness_achromatic_normalised,
                                    hue,
                                    saturation,
                                    chroma,
                                    colorfulness)


def illuminance_to_luminance(E, Y_f):
    """
    Converts given *illuminance* :math:`E` value in lux to *luminance* in
    :math:`cd/m^2`.

    Parameters
    ----------
    E : float
        *Illuminance* :math:`E` in lux.
    Y_f : float
        *Luminance* factor :math:`Y_f` in :math:`cd/m^2`.

    Returns
    -------
    float
        *Luminance* :math:`Y` in :math:`cd/m^2`.

    Examples
    --------
    >>> colour.appearance.nayatani95.illuminance_to_luminance(5000.0, 20.0)
    318.3098861837907
    """

    return Y_f * E / (100 * np.pi)


def get_intermediate_values(XYZ_n):
    """
    Returns the intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.

    Parameters
    ----------
    XYZ_n : array_like, (3,)
        *CIE XYZ* colourspace matrix of reference white in domain [0, 100].

    Returns
    -------
    ndarray, (3,)
        Intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.

    Examples
    --------
    >>> XYZ_n = np.array([95.05, 100, 108.88])
    >>> colour.appearance.nayatani95.get_intermediate_values(XYZ_n)
    array([ 1.00004219,  0.99998001,  0.99975794])
    """

    # Illuminant chromaticity coordinates.
    x_o, y_o = XYZ_to_xy(XYZ_n)

    # Computing :math:`\xi`, :math:`\eta`, :math:`\zeta` values.
    xi = (0.48105 * x_o + 0.78841 * y_o - 0.08081) / y_o
    eta = (-0.27200 * x_o + 1.11962 * y_o + 0.04570) / y_o
    zeta = (0.91822 * (1 - x_o - y_o)) / y_o

    return np.array([xi, eta, zeta])


def XYZ_to_RGB_Nayatani95(XYZ):
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
    >>> colour.appearance.nayatani95.XYZ_to_RGB(XYZ)
    array([ 20.0005206  19.999783   19.9988316])
    """

    return NAYATANI95_XYZ_TO_RGB_MATRIX.dot(XYZ)


def get_beta_1(x):
    """
    Computes the exponent :math:`\beta_1` for the middle and long-wavelength
    sensitive cones.

    Parameters
    ----------
    x: float
        Middle and long-wavelength sensitive cone response.

    Returns
    -------
    float
        Exponent :math:`\beta_1`.

    Examples
    --------
    >>> colour.appearance.nayatani95.get_beta_1(318.323316315)
    4.61062222647
    """

    return (6.469 + 6.362 * (x ** 0.4495)) / (6.469 + (x ** 0.4495))


def get_beta_2(x):
    """
    Computes the exponent :math:`\beta_2` for the short-wavelength sensitive
    cones.

    Parameters
    ----------
    x: float
        Short-wavelength sensitive cone response.

    Returns
    -------
    float
        Exponent :math:`\beta_2`.

    Examples
    --------
    >>> colour.appearance.nayatani95.get_beta_2(318.323316315)
    4.6520698609530138
    """

    return 0.7844 * (8.414 + 8.091 * (x ** 0.5128)) / (8.414 + (x ** 0.5128))


def get_chromatic_adaptation_exponential_factors(RGB_o):
    """
    Returns the chromatic adaptation exponential factors :math:`\beta_1(R_o)`,
    `math:`\beta_1(G_o)` and :math:`\beta_2(B_o)` of given cone responses.

    Parameters
    ----------
    RGB_o: ndarray, (3,)
         Cone responses.

    Returns
    -------
    ndarray, (3,)
        Chromatic adaptation exponential factors :math:`\beta_1(R_o)`,
        `math:`\beta_1(G_o)` and :math:`\beta_2(B_o)`.

    Examples
    --------
    >>> RGB_o = np.array([318.32331631, 318.30352317, 318.23283482])
    >>> colour.appearance.nayatani95.get_chromatic_adaptation_exponential_factors(RGB_o)
    array([ 4.61062223,  4.61058926,  4.65206986])
    """

    R_o, G_o, B_o = np.ravel(RGB_o)

    bR_o = get_beta_1(R_o)
    bG_o = get_beta_1(G_o)
    bB_o = get_beta_2(B_o)

    return np.array([bR_o, bG_o, bB_o])


def get_scaling_coefficient(x, y):
    """
    Returns the scaling coefficient :math:`e(R)` or :math:`e(G)`.

    Parameters
    ----------
    x: float
        Cone response.
    y: float
        Intermediate value.

    Returns
    -------
    float
        Scaling coefficient :math:`e(R)` or :math:`e(G)`.

    Examples
    --------
    >>> x = 20.000520600000002
    >>> y = 1.000042192
    >>> colour.appearance.nayatani95.get_scaling_coefficient(x, y)
    1.0
    """

    return 1.758 if x >= (20 * y) else 1


def get_achromatic_response(RGB, bRGB_o, x_e_z, bL_or, eR, eG, n=1):
    """
    Returns the achromatic response :math:`Q` from given stimulus cone
    responses.

    Parameters
    ----------
    RGB: ndarray, (3,)
         Stimulus cone responses.
    bRGB_o: ndarray, (3,)
         Chromatic adaptation exponential factors :math:`\beta_1(R_o)`,
         `math:`\beta_1(G_o)` and :math:`\beta_2(B_o)`.
    x_e_z: ndarray, (3,)
        Intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.
    bL_or: float
         Normalising chromatic adaptation exponential factor
         :math:`\beta_1(B_or)`.
    eR: float
         Scaling coefficient :math:`e(R)`.
    eG: float
         Scaling coefficient :math:`e(G)`.
    n : float, optional
        Noise term used in the non linear chromatic adaptation model.

    Returns
    -------
    float
        Achromatic response :math:`Q`.

    Examples
    --------
    >>> RGB = np.array([20.0005206, 19.999783, 19.9988316])
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> x_e_z = np.array([1.00004219, 0.99998001, 0.99975794])
    >>> bL_or = 3.6810214956040888
    >>> eR = 1.0
    >>> eG = 1.758
    >>> n = 1.0
    >>> colour.appearance.nayatani95.get_achromatic_response(RGB, bRGB_o, x_e_z, bL_or, eR, eG, n)
    -0.000117024294955
    """

    R, G, B = RGB
    bR_o, bG_o, bB_o = bRGB_o
    xi, eta, zeta = x_e_z

    Q = (2 / 3) * bR_o * eR * np.log10((R + n) / (20 * xi + n))
    Q += (1 / 3) * bG_o * eG * np.log10((G + n) / (20 * eta + n))
    Q *= 41.69 / bL_or

    return Q


def get_tritanopic_response(RGB, bRGB_o, x_e_z, n):
    """
    Returns the tritanopic response :math:`t` from given stimulus cone
    responses.

    Parameters
    ----------
    RGB: ndarray, (3,)
         Stimulus cone responses.
    bRGB_o: ndarray, (3,)
         Chromatic adaptation exponential factors :math:`\beta_1(R_o)`,
         `math:`\beta_1(G_o)` and :math:`\beta_2(B_o)`.
    x_e_z: ndarray, (3,)
        Intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.
    n : float, optional
        Noise term used in the non linear chromatic adaptation model.

    Returns
    -------
    float
        Tritanopic response :math:`t`.

    Examples
    --------
    >>> RGB = np.array([20.0005206, 19.999783, 19.9988316])
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> x_e_z = np.array([1.00004219, 0.99998001, 0.99975794])
    >>> n = 1.0
    >>> colour.appearance.nayatani95.get_tritanopic_response(RGB, bRGB_o, x_e_z, n)
    -1.7703650668990973e-05
    """

    R, G, B = RGB
    bR_o, bG_o, bB_o = bRGB_o
    xi, eta, zeta = x_e_z

    t = (1 / 1) * bR_o * np.log10((R + n) / (20 * xi + n))
    t += - (12 / 11) * bG_o * np.log10((G + n) / (20 * eta + n))
    t += (1 / 11) * bB_o * np.log10((B + n) / (20 * zeta + n))

    return t


def get_protanopic_response(RGB, bRGB_o, x_e_z, n):
    """
    Returns the protanopic response :math:`p` from given stimulus cone
    responses.

    Parameters
    ----------
    RGB: ndarray, (3,)
         Stimulus cone responses.
    bRGB_o: ndarray, (3,)
         Chromatic adaptation exponential factors :math:`\beta_1(R_o)`,
         `math:`\beta_1(G_o)` and :math:`\beta_2(B_o)`.
    x_e_z: ndarray, (3,)
        Intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.
    n : float, optional
        Noise term used in the non linear chromatic adaptation model.

    Returns
    -------
    float
        Protanopic response :math:`p`.

    Examples
    --------
    >>> RGB = np.array([20.0005206, 19.999783, 19.9988316])
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> x_e_z = np.array([1.00004219, 0.99998001, 0.99975794])
    >>> n = 1.0
    >>> colour.appearance.nayatani95.get_protanopic_response(RGB, bRGB_o, x_e_z, n)
    -8.002142682085493e-05
    """

    R, G, B = RGB
    bR_o, bG_o, bB_o = bRGB_o
    xi, eta, zeta = x_e_z

    p = (1 / 9) * bR_o * np.log10((R + n) / (20 * xi + n))
    p += (1 / 9) * bG_o * np.log10((G + n) / (20 * eta + n))
    p += - (2 / 9) * bB_o * np.log10((B + n) / (20 * zeta + n))

    return p


def get_brightness_correlate(bRGB_o, bL_or, Q):
    """
    Returns the *brightness* correlate :math:`B_r`.

    Parameters
    ----------
    bRGB_o: ndarray, (3,)
         Chromatic adaptation exponential factors :math:`\beta_1(R_o)`,
         `math:`\beta_1(G_o)` and :math:`\beta_2(B_o)`.
    bL_or: float
         Normalising chromatic adaptation exponential factor
         :math:`\beta_1(B_or)`.
    Q : float
        Achromatic response :math:`Q`.
    Returns
    -------
    float
        *Brightness* correlate :math:`B_r`.

    Examples
    --------
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> bL_or = 3.6810214956040888
    >>> Q = -0.000117024294955
    >>> colour.appearance.nayatani95.get_brightness_correlate(bRGB_o, bL_or, Q)
    62.626673467230766
    """

    bR_o, bG_o, bB_o = bRGB_o

    B_r = (50 / bL_or) * ((2 / 3) * bR_o + (1 / 3) * bG_o) + Q

    return B_r


def get_ideal_white_brightness_correlate(bRGB_o, x_e_z, bL_or, n):
    """
    Returns the ideal white *brightness* correlate :math:`B_{rw}`.

    Parameters
    ----------
    bRGB_o: ndarray, (3,)
         Chromatic adaptation exponential factors :math:`\beta_1(R_o)`,
         `math:`\beta_1(G_o)` and :math:`\beta_2(B_o)`.
    x_e_z: ndarray, (3,)
        Intermediate values :math:`\\xi`, :math:`\eta`, :math:`\zeta`.
    bL_or: float
         Normalising chromatic adaptation exponential factor
         :math:`\beta_1(B_or)`.
    n : float, optional
        Noise term used in the non linear chromatic adaptation model.

    Returns
    -------
    float
        Ideal white *brightness* correlate :math:`B_{rw}`.

    Examples
    --------
    >>> bRGB_o = np.array([4.61062223, 4.61058926, 4.65206986])
    >>> x_e_z = np.array([1.00004219, 0.99998001, 0.99975794])
    >>> bL_or = 3.6810214956040888
    >>> n = 1.0
    >>> colour.appearance.nayatani95.get_ideal_white_brightness_correlate(bRGB_o, x_e_z, bL_or, n)
    125.24353925846037
    """

    bR_o, bG_o, bB_o = bRGB_o
    xi, eta, zeta = x_e_z

    B_rw = (2 / 3) * bR_o * 1.758 * np.log10((100 * xi + n) / (20 * xi + n))
    B_rw += (1 / 3) * bG_o * 1.758 * np.log10((100 * eta + n) / (20 * eta + n))
    B_rw *= 41.69 / bL_or
    B_rw += (50 / bL_or) * (2 / 3) * bR_o
    B_rw += (50 / bL_or) * (1 / 3) * bG_o

    return B_rw


def get_achromatic_lightness_correlate(Q):
    """
    Returns the *achromatic Lightness* correlate :math:`L_p^\star`.

    Parameters
    ----------
    Q : float
        Achromatic response :math:`Q`.

    Returns
    -------
    float
        *Achromatic Lightness* correlate :math:`L_p^\star`.

    Examples
    --------
    >>> Q = -0.000117024294955
    >>> colour.appearance.nayatani95.get_achromatic_lightness_correlate(Q)
    49.99988297570504
    """

    return Q + 50


def get_normalised_achromatic_lightness_correlate(B_r, B_rw):
    """
    Returns the *normalised achromatic Lightness* correlate :math:`L_n^\star`.

    Parameters
    ----------
    B_r : float
        *Brightness* correlate :math:`B_r`.
    B_rw : float
        Ideal white *brightness* correlate :math:`B_{rw}`.

    Returns
    -------
    float
        *Normalised achromatic Lightness* correlate :math:`L_n^\star`.

    Examples
    --------
    >>> B_r = 62.626673467230766
    >>> B_rw = 125.24353925846037
    >>> colour.appearance.nayatani95.get_normalised_achromatic_lightness_correlate(B_r, B_rw)
    50.003915441889944
    """

    return 100 * (B_r / B_rw)


def get_hue_angle(p, t):
    """
    Returns the *hue* angle :math:`h` in degrees.

    Parameters
    ----------
    p : float
        Protanopic response :math:`p`.
    t : float
        Tritanopic response :math:`t`.

    Returns
    -------
    float
        *Hue* angle :math:`h` in degrees.

    Examples
    --------
    >>> p = -8.002142682085493e-05
    >>> t = -1.7703650668990973e-05
    >>> colour.appearance.nayatani95.get_hue_correlate(p, t)
    257.52503009852325
    """

    h_L = math.degrees(np.arctan2(p, t)) % 360
    return h_L


def chromatic_strength_function(theta):
    """
    Defines the chromatic strength function :math:`E_s(\\theta)` used to correct
    saturation scale as function of hue angle :math:`\\theta`

    Parameters
    ----------
    theta : float
        Hue angle :math:`\\theta`

    Returns
    -------
    float
        Corrected saturation scale.

    Examples
    --------
    >>> colour.appearance.nayatani95.chromatic_strength_function(4.49462820973)
    1.22678698241
    """

    E_s = 0.9394
    E_s += - 0.2478 * np.sin(1 * theta)
    E_s += - 0.0743 * np.sin(2 * theta)
    E_s += + 0.0666 * np.sin(3 * theta)
    E_s += - 0.0186 * np.sin(4 * theta)
    E_s += - 0.0055 * np.cos(1 * theta)
    E_s += - 0.0521 * np.cos(2 * theta)
    E_s += - 0.0573 * np.cos(3 * theta)
    E_s += - 0.0061 * np.cos(4 * theta)

    return E_s


def get_saturation_components(h, bL_or, t, p):
    """
    Returns the *saturation* components :math:`S_{RG}` and :math:`S_{YB}`.

    Parameters
    ----------
    h: float
        Correlate of *hue* :math:`h` in degrees.
    bL_or: float
         Normalising chromatic adaptation exponential factor
         :math:`\beta_1(B_or)`.
    t : float
        Tritanopic response :math:`t`.
    p : float
        Protanopic response :math:`p`.

    Returns
    -------
    float
        *Saturation* components :math:`S_{RG}` and :math:`S_{YB}`.

    Examples
    --------
    >>> h = 257.52322689806243
    >>> bL_or = 3.6810214956040888
    >>> t = -1.7706764677181658e-05
    >>> p = -8.0023561356363753e-05
    >>> colour.appearance.nayatani95.get_saturation_components(h, bL_or, t, p)
    (-0.0028852716381965863, -0.013039632941332499)
    """

    E_s = chromatic_strength_function(math.radians(h))
    S_RG = (488.93 / bL_or) * E_s * t
    S_YB = (488.93 / bL_or) * E_s * p

    return S_RG, S_YB


def get_saturation_correlate(S_RG, S_YB):
    """
    Returns the correlate of *saturation* :math:`S`.

    Parameters
    ----------
    S_RG : float
        *Saturation* component :math:`S_{RG}`.
    S_YB : float
        *Saturation* component :math:`S_{YB}`.

    Returns
    -------
    float
        Correlate of *saturation* :math:`S`.

    Examples
    --------
    >>> S_RG = -0.0028852716381965863
    >>> S_YB = -0.013039632941332499
    >>> colour.appearance.nayatani95.get_saturation_correlate(S_RG, S_YB)
    0.013355029751777615
    """

    S = np.sqrt((S_RG ** 2) + (S_YB ** 2))

    return S


def get_chroma_components(L_star_p, S_RG, S_YB):
    """
    Returns the *chroma* components :math:`C_{RG}` and :math:`C_{YB}`.

    Parameters
    ----------
    L_star_p : float
        *Achromatic Lightness* correlate :math:`L_p^\star`.
    S_RG : float
        *Saturation* component :math:`S_{RG}`.
    S_YB : float
        *Saturation* component :math:`S_{YB}`.

    Returns
    -------
    float
        *Chroma* components :math:`C_{RG}` and :math:`C_{YB}`.

    Examples
    --------
    >>> L_star_p = 49.99988297570504
    >>> S_RG = -0.0028852716381965863
    >>> S_YB = -0.013039632941332499
    >>> colour.appearance.nayatani95.get_chroma_components(L_star_p, S_RG, S_YB)
    (-0.0028852716381965863, -0.013039632941332499)
    """

    C_RG = ((L_star_p / 50) ** 0.7) * S_RG
    C_YB = ((L_star_p / 50) ** 0.7) * S_YB

    return C_RG, C_YB


def get_chroma_correlate(L_star_p, S):
    """
    Returns the correlate of *chroma* :math:`C`.

    Parameters
    ----------
    L_star_p : float
        *Achromatic Lightness* correlate :math:`L_p^\star`.
    S : float
        Correlate of *saturation* :math:`S`.

    Returns
    -------
    float
        Correlate of *chroma* :math:`C`.

    Examples
    --------
    >>> L_star_p = 49.99988297570504
    >>> S = 0.013355029751777615
    >>> colour.appearance.nayatani95.get_chroma_correlate(L_star_p, S)
    0.013355007871688761
    """

    C = ((L_star_p / 50) ** 0.7) * S
    return C


def get_colourfulness_components(C_RG, C_YB, B_rw):
    """
    Returns the *colourfulness* components :math:`M_{RG}` and :math:`M_{YB}`.

    Parameters
    ----------
    C_RG : float
        *Chroma* component :math:`C_{RG}`.
    C_YB : float
        *Chroma* component :math:`C_{YB}`.
    B_rw : float
        Ideal white *brightness* correlate :math:`B_{rw}`.

    Returns
    -------
    float
        *Colourfulness* components :math:`M_{RG}` and :math:`M_{YB}`.

    Examples
    --------
    >>> C_RG = -0.0028852716381965863
    >>> C_YB = -0.013039632941332499
    >>> B_rw = 125.24353925846037
    >>> colour.appearance.nayatani95.get_colourfulness_components(C_RG, C_YB, B_rw)
    (-0.0036136163168979645, -0.0163312978020369)
    """

    M_RG = C_RG * B_rw / 100
    M_YB = C_YB * B_rw / 100

    return M_RG, M_YB


def get_colourfulness_correlate(C, B_rw):
    """
    Returns the correlate of *colourfulness* :math:`M`.

    Parameters
    ----------
    C : float
        Correlate of *chroma* :math:`C`.
    B_rw : float
        Ideal white *brightness* correlate :math:`B_{rw}`.

    Returns
    -------
    float
        Correlate of *colourfulness* :math:`M`.

    Examples
    --------
    >>> C = 0.013355007871688761
    >>> B_rw = 125.24353925846037
    >>> colour.appearance.nayatani95.get_colourfulness_correlate(C, B_rw)
    0.016726284526748986
    """

    M = C * B_rw / 100
    return M

