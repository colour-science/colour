#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLAB(l:c) Colour Appearance Model
=================================

Defines *LLAB(l:c)* colour appearance model objects:

-   :attr:`LLAB_Specification`:
-   :func:`XYZ_to_LLAB`

References
----------
.. [1]  **Mark D. Fairchild**, *Color Appearance Models, 3nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published June 2013, ASIN: B00DAYO8E2,
        locations 6019-6178.
.. [2]  **Luo, M. R., & Morovic, J.**,
        *Two Unsolved Issues in Colour Management – Colour Appearance and
        Gamut Mapping*,
        *5th International Conference on High Technology: Imaging Science and
        Technology – Evolution & Promise*
        published 1996, pp. 136–147.
.. [3]  **Luo, M. R., Lo, M. C., & Kuo, W. G.**,
        *The LLAB (l:c) colour model*,
        *Color Research & Application, Volume 21, Issue 6, pages 412–429,
        December 1996*
"""

from __future__ import division, unicode_literals

import math
import numpy as np
from collections import namedtuple

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LLAB_InductionFactors',
           'LLAB_VIEWING_CONDITIONS',
           'LLAB_XYZ_TO_RGB_MATRIX',
           'LLAB_XYZ_TO_RGB_INVERSE_MATRIX',
           'LLAB_Specification',
           'XYZ_to_LLAB',
           'XYZ_to_RGB_LLAB',
           'get_chromatic_adaptation',
           'f',
           'get_opponent_colour_dimensions',
           'get_hue_angle',
           'get_chroma_correlate',
           'get_colourfulness_correlate',
           'get_saturation_correlate',
           'get_final_opponent_signals']

LLAB_InductionFactors = namedtuple('LLAB_InductionFactors',
                                   ('D', 'F_S', 'F_L', 'F_C'))

LLAB_VIEWING_CONDITIONS = {
    'Reference Samples & Images, Average Surround, Subtending > 4': \
        LLAB_InductionFactors(1, 3, 0, 1),
    'Reference Samples & Images, Average Surround, Subtending < 4': \
        LLAB_InductionFactors(1, 3, 1, 1),
    'Television & VDU Displays, Dim Surround': \
        LLAB_InductionFactors(0.7, 3.5, 1, 1),
    'Cut Sheet Transparency, Dim Surround': \
        LLAB_InductionFactors(1, 5, 1, 1.1),
    '35mm Projection Transparency, Dark Surround': \
        LLAB_InductionFactors(0.7, 4, 1, 1)}
"""
Reference *LLAB(l:c)* colour appearance model viewing conditions.

LLAB_VIEWING_CONDITIONS : dict
('Reference Samples & Images, Average Surround, Subtending > 4',
'Reference Samples & Images, Average Surround, Subtending < 4',
'Television & VDU Displays, Dim Surround',
'Cut Sheet Transparency, Dim Surround':,
'35mm Projection Transparency, Dark Surround')
"""

LLAB_XYZ_TO_RGB_MATRIX = np.array(
    [[0.8951, 0.2664, -0.1614],
     [-0.7502, 1.7135, 0.0367],
     [0.0389, -0.0685, 1.0296]])

# LLAB_XYZ_TO_RGB_INVERSE_MATRIX = np.linalg.inv(LLAB_XYZ_TO_RGB_MATRIX)
# TODO: Investigate rounding issues.
LLAB_XYZ_TO_RGB_INVERSE_MATRIX = np.array(
    [[0.987, -0.1471, 0.1600],
     [0.4323, 0.5184, 0.0493],
     [-0.0085, 0.0400, 0.9685]])

LLAB_Specification = namedtuple(
    'LLAB_Specification',
    ('h_L', 'Ch_L', 's_L', 'L_L', 'C_L', 'A_L', 'B_L'))
"""
Defines the *LLAB(l:c)* colour appearance model specification.

Parameters
----------
h_L : float
    *Hue* angle :math:`h_L` in degrees.
Ch_L : float
    Correlate of *chroma* :math:`Ch_L`.
s_L : float
    Correlate of *saturation* :math:`s_L`.
L_L : float
    Correlate of *Lightness* :math:`L_L`.
C_L : float
    Correlate of *colourfulness* :math:`C_L`.
A_L : float
    Opponent signal :math:`A_L`.
B_L : float
    Opponent signal :math:`B_L`.
"""


def XYZ_to_LLAB(XYZ,
                XYZ_0,
                Y_b,
                F_S,
                F_L,
                F_C,
                L,
                D=1):
    """
    Computes the *LLAB(L:c)* colour appearance model correlates.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    XYZ_0 : array_like, (3,)
        *CIE XYZ* colourspace matrix of reference white in domain [0, 100].
    Y_b : float
        Luminance factor of the background in :math:`cd/m^2`.
    F_S : float
        Surround induction factor :math:`F_S`.
    F_L : float
        Lightness induction factor :math:`F_L`.
    F_C : float
        Chroma induction factor :math:`F_C`.
    L : float
        Absolute luminance :math:`L` of reference white in :math:`cd/m^2`.
    D : float, optional
         *Discounting-the-Illuminant* factor in domain [0, 1].

    Returns
    -------
    LLAB_Specification
        *LLAB(L:c)* colour appearance model specification.

    Warning
    -------
    The output domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    -   Input *CIE XYZ_0* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> XYZ = np.array([ 19.01,  20  ,  21.78])
    >>> XYZ_0 = np.array([  95.05,  100  ,  108.88])
    >>> Y_b = 20.0
    >>> F_S = 3.0
    >>> F_L = 1.0
    >>> F_C = 1.0
    >>> L = 318.31
    >>> colour.XYZ_to_LLAB(XYZ, XYZ_0, Y_b, F_S, F_L, F_C, L)
    LLAB_Specification(h_L=229.46357270056598, Ch_L=0.0086506620517144972, s_L=0.00023149890432782482, L_L=37.368047493928195, C_L=0.018383289914270105, A_L=-0.011947876709772017, B_L=-0.013971169965331907)
    """

    X, Y, Z = np.ravel(XYZ)
    RGB = XYZ_to_RGB_LLAB(XYZ)
    RGB_0 = XYZ_to_RGB_LLAB(XYZ_0)

    # Reference illuminant *CIE Standard Illuminant D Series* *D65*.
    XYZ_0r = np.array([95.05, 100, 108.88])
    RGB_0r = XYZ_to_RGB_LLAB(XYZ_0r)

    # Computing chromatic adaptation.
    XYZ_r = get_chromatic_adaptation(RGB, RGB_0, RGB_0r, Y, D)

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`L_L`.
    # -------------------------------------------------------------------------
    # Computing opponent colour dimensions.
    lightness, a, b = get_opponent_colour_dimensions(XYZ_r, Y_b, F_S, F_L)

    # Computing perceptual correlates.
    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`Ch_L`.
    # -------------------------------------------------------------------------
    chroma = get_chroma_correlate(a, b)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`C_L`.
    # -------------------------------------------------------------------------
    colourfulness = get_colourfulness_correlate(L, lightness, chroma, F_C)

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`S_L`.
    # -------------------------------------------------------------------------
    saturation = get_saturation_correlate(chroma, lightness)

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`h_L`.
    # -------------------------------------------------------------------------
    hue = get_hue_angle(a, b)
    h_Lr = math.radians(hue)
    # TODO: Implement hue quadrature & composition computation.

    # -------------------------------------------------------------------------
    # Computing final opponent signals.
    # -------------------------------------------------------------------------
    A_L, B_L = get_final_opponent_signals(colourfulness, h_Lr)

    return LLAB_Specification(hue,
                              chroma,
                              saturation,
                              lightness,
                              colourfulness,
                              A_L,
                              B_L)


def XYZ_to_RGB_LLAB(XYZ):
    """
    Converts from *CIE XYZ* colourspace to normalised cone responses.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        Normalised cone responses.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20, 21.78])
    >>> colour.appearance.llab.XYZ_to_RGB(XYZ)
    array([ 0.94142795,  1.0404012 ,  1.08970885])
    """

    return LLAB_XYZ_TO_RGB_MATRIX.dot(XYZ / XYZ[1])


def get_chromatic_adaptation(RGB, RGB_0, RGB_0r, Y, D=1):
    """
    Applies chromatic adaptation to given *RGB* normalised cone responses
    matrix.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* normalised cone responses matrix of test sample / stimulus.
    RGB_0 : array_like, (3,)
        *RGB* normalised cone responses matrix of reference white.
    RGB_0r : array_like, (3,)
        *RGB* normalised cone responses matrix of reference illuminant
        *CIE Standard Illuminant D Series* *D65*.
    Y : float
        Tristimulus values :math:`Y` of the stimulus.
    D : float, optional
         *Discounting-the-Illuminant* factor in domain [0, 1].

    Returns
    -------
    ndarray, (3,)
        Adapted *CIE XYZ* colourspace matrix.

    Examples
    --------
    >>> RGB = np.array([0.94142795, 1.0404012, 1.08970885])
    >>> RGB_0 = np.array([0.94146023, 1.04039386, 1.08950293])
    >>> RGB_0r = np.array([0.94146023, 1.04039386, 1.08950293])
    >>> Y = 20.0
    >>> colour.appearance.llab.get_chromatic_adaptation(RGB, RGB_0, RGB_0r, Y)
    array([ 19.00999572,  20.00091862,  21.77993863])
    """

    R, G, B = np.ravel(RGB)
    R_0, G_0, B_0 = np.ravel(RGB_0)
    R_0r, G_0r, B_0r = np.ravel(RGB_0r)

    beta = (B_0 / B_0r) ** 0.0834

    R_r = (D * (R_0r / R_0) + 1 - D) * R
    G_r = (D * (G_0r / G_0) + 1 - D) * G
    B_r = (D * (B_0r / (B_0 ** beta)) + 1 - D) * (abs(B) ** beta)

    RGB_r = np.array([R_r, G_r, B_r])

    X_r, Y_r, Z_r = LLAB_XYZ_TO_RGB_INVERSE_MATRIX.dot(RGB_r * Y)

    return np.array([X_r, Y_r, Z_r])


def f(x, F_S):
    """
    Defines the nonlinear response function of the *LLAB(L:c)* colour
    appearance model used to model the nonlinear behavior of various visual
    responses.

    Parameters
    ----------
    x : float or array_like
        Visual response variable :math:`x`.
    F_S : float
        Surround induction factor :math:`F_S`.

    Returns
    -------
    float or array_like
        Modeled visual response variable :math:`x`.

    Examples
    --------
    >>> x = np.array([0.23350512, 0.23351103, 0.23355179]
    >>> colour.appearance.llab.f(0.20000918623399996, 3)
    array(0.5848125010758818)
    """

    x_m = np.where(x > 0.008856,
                   x ** (1 / F_S),
                   ((((0.008856 ** (1 / F_S)) -
                      (16 / 116)) / 0.008856) * x + (16 / 116)))
    return x_m


def get_opponent_colour_dimensions(XYZ, Y_b, F_S, F_L):
    """
    Returns opponent colour dimensions from given adapted *CIE XYZ* colourspace
    matrix.

    The opponent colour dimensions are based on a modified *CIE Lab*
    colourspace formulae.

    Parameters
    ----------
    XYZ : array_like, (3,)
        Adapted *CIE XYZ* colourspace matrix.
    Y_b : float
        Luminance factor of the background in :math:`cd/m^2`.
    F_S : float
        Surround induction factor :math:`F_S`.
    F_L : float
        Lightness induction factor :math:`F_L`.

    Returns
    -------
    ndarray, (3,)
        Opponent colour dimensions.

    Examples
    --------
    >>> XYZ = np.array([19.00999572, 20.00091862, 21.77993863])
    >>> Y_b = 20.0
    >>> F_S = 3.0
    >>> F_L = 1.0
    >>> colour.appearance.llab.get_opponent_colour_dimensions(XYZ, Y_b, F_S, F_L)
    array([  3.73680475e+01,  -4.49864756e-03,  -5.26046353e-03])
    """

    X, Y, Z = np.ravel(XYZ)

    # Account for background lightness contrast.
    z = 1 + F_L * ((Y_b / 100) ** 0.5)

    # Computing modified *CIE Lab* colourspace matrix.
    L = 116 * (f(Y / 100, F_S) ** z) - 16
    a = 500 * (f(X / 95.05, F_S) - f(Y / 100, F_S))
    b = 200 * (f(Y / 100, F_S) - f(Z / 108.88, F_S))

    return np.array([L, a, b])


def get_hue_angle(a, b):
    """
    Returns the *hue* angle :math:`h_L` in degrees.

    Parameters
    ----------
    a : float
        Opponent colour dimension :math:`a`.
    b : float
        Opponent colour dimension :math:`b`.

    Returns
    -------
    float
        *Hue* angle :math:`h_L` in degrees.

    Examples
    --------
    >>> colour.appearance.llab.get_hue_correlate(-4.49864756e-03, -5.26046353e-03)
    229.4635727085839
    """

    h_L = math.degrees(np.arctan2(b, a)) % 360
    return h_L


def get_chroma_correlate(a, b):
    """
    Returns the correlate of *chroma* :math:`Ch_L`.

    Parameters
    ----------
    a : float
        Opponent colour dimension :math:`a`.
    b : float
        Opponent colour dimension :math:`b`.

    Returns
    -------
    float
        Correlate of *chroma* :math:`Ch_L`.

    Examples
    --------
    >>> colour.appearance.llab.get_chroma_correlate(-4.49864756e-03, -5.26046353e-03)
    0.0086506620569251902
    """

    c = (a ** 2 + b ** 2) ** 0.5
    Ch_L = 25 * np.log(1 + 0.05 * c)
    return Ch_L


def get_colourfulness_correlate(L, L_L, Ch_L, F_C):
    """
    Returns the correlate of *colourfulness* :math:`C_L`.

    Parameters
    ----------
    L : float
        Absolute luminance :math:`L` of reference white in :math:`cd/m^2`.
    L_L : float
        Correlate of *Lightness* :math:`L_L`.
    Ch_L : float
        Correlate of *chroma* :math:`Ch_L`.
    F_C : float
        Chroma induction factor :math:`F_C`.

    Returns
    -------
    float
        Correlate of *colourfulness* :math:`C_L`.

    Examples
    --------
    >>> L = 318.31
    >>> L_L = 37.368047493928195
    >>> Ch_L = 0.0086506620517144972
    >>> F_C = 1.0
    >>> colour.appearance.llab.get_colourfulness_correlate(L, L_L, Ch_L, F_C)
    0.0183832899143
    """

    S_C = 1 + 0.47 * np.log10(L) - 0.057 * np.log10(L) ** 2
    S_M = 0.7 + 0.02 * L_L - 0.0002 * L_L ** 2
    C_L = Ch_L * S_M * S_C * F_C

    return C_L


def get_saturation_correlate(Ch_L, L_L):
    """
    Returns the correlate of *saturation* :math:`S_L`.

    Parameters
    ----------
    Ch_L : float
        Correlate of *chroma* :math:`Ch_L`.
    L_L : float
        Correlate of *Lightness* :math:`L_L`.

    Returns
    -------
    float
        Correlate of *saturation* :math:`S_L`.

    Examples
    --------
    >>> Ch_L = 0.0086506620517144972
    >>> L_L = 37.368047493928195
    >>> colour.appearance.llab.get_saturation_correlate(Ch_L, L_L)
    0.00023149890432782482
    """

    S_L = Ch_L / L_L

    return S_L


def get_final_opponent_signals(C_L, h_L):
    """
    Returns the final opponent signals :math:`A_L` and :math:`B_L`.

    Parameters
    ----------
    C_L : float
        Correlate of *colourfulness* :math:`C_L`.
    h_L : float
        Correlate of *hue* :math:`h_L` in radians.

    Returns
    -------
    tuple
        Final opponent signals :math:`A_L` and :math:`B_L`.

    Examples
    --------
    >>> C_L = 0.0183832899143
    >>> h_L = 4.004894857014253
    >>> colour.appearance.llab.get_final_opponent_signals(C_L, h_L)
    (-0.01194787670977202, -0.013971169965331903)
    """

    A_L = C_L * np.cos(h_L)
    B_L = C_L * np.sin(h_L)

    return A_L, B_L
