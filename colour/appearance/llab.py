#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLAB(l:c) Colour Appearance Model
=================================

Defines *LLAB(l:c)* colour appearance model objects:

-   :class:`LLAB_InductionFactors`
-   :attr:`LLAB_VIEWING_CONDITIONS`
-   :class:`LLAB_Specification`
-   :func:`XYZ_to_LLAB`

See Also
--------
`LLAB(l:c) Colour Appearance Model IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/appearance/llab.ipynb>`_  # noqa

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
        December 1996*,
        DOI: http://dx.doi.org/10.1002/(SICI)1520-6378(199612)21:6<412::AID-COL4>3.0.CO;2-Z  # noqa
"""

from __future__ import division, unicode_literals

import math
import numpy as np
from collections import namedtuple

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LLAB_InductionFactors',
           'LLAB_VIEWING_CONDITIONS',
           'LLAB_XYZ_TO_RGB_MATRIX',
           'LLAB_RGB_TO_XYZ_MATRIX',
           'LLAB_ReferenceSpecification',
           'LLAB_Specification',
           'XYZ_to_LLAB',
           'XYZ_to_RGB_LLAB',
           'chromatic_adaptation',
           'f',
           'opponent_colour_dimensions',
           'hue_angle',
           'chroma_correlate',
           'colourfulness_correlate',
           'saturation_correlate',
           'final_opponent_signals']


class LLAB_InductionFactors(
    namedtuple('LLAB_InductionFactors',
               ('D', 'F_S', 'F_L', 'F_C'))):
    """
    *LLAB(l:c)* colour appearance model induction factors.

    Parameters
    ----------
    D : numeric
         *Discounting-the-Illuminant* factor :math:`D` in domain [0, 1].
    F_S : numeric
        Surround induction factor :math:`F_S`.
    F_L : numeric
        *Lightness* induction factor :math:`F_L`.
    F_C : numeric
        *Chroma* induction factor :math:`F_C`.
    """


LLAB_VIEWING_CONDITIONS = CaseInsensitiveMapping(
    {'Reference Samples & Images, Average Surround, Subtending > 4': (
        LLAB_InductionFactors(1, 3, 0, 1)),
     'Reference Samples & Images, Average Surround, Subtending < 4': (
         LLAB_InductionFactors(1, 3, 1, 1)),
     'Television & VDU Displays, Dim Surround': (
         LLAB_InductionFactors(0.7, 3.5, 1, 1)),
     'Cut Sheet Transparency, Dim Surround': (
         LLAB_InductionFactors(1, 5, 1, 1.1)),
     '35mm Projection Transparency, Dark Surround': (
         LLAB_InductionFactors(0.7, 4, 1, 1))})
"""
Reference *LLAB(l:c)* colour appearance model viewing conditions.

LLAB_VIEWING_CONDITIONS : CaseInsensitiveMapping
    {'Reference Samples & Images, Average Surround, Subtending > 4',
    'Reference Samples & Images, Average Surround, Subtending < 4',
    'Television & VDU Displays, Dim Surround',
    'Cut Sheet Transparency, Dim Surround':,
    '35mm Projection Transparency, Dark Surround'}

Aliases:

-   'ref_average_4_plus':
    'Reference Samples & Images, Average Surround, Subtending > 4'
-   'ref_average_4_minus':
    'Reference Samples & Images, Average Surround, Subtending < 4'
-   'tv_dim': 'Television & VDU Displays, Dim Surround'
-   'sheet_dim': 'Cut Sheet Transparency, Dim Surround'
-   'projected_dark': '35mm Projection Transparency, Dark Surround'
"""
LLAB_VIEWING_CONDITIONS['ref_average_4_plus'] = (
    LLAB_VIEWING_CONDITIONS[
        'Reference Samples & Images, Average Surround, Subtending > 4'])
LLAB_VIEWING_CONDITIONS['ref_average_4_minus'] = (
    LLAB_VIEWING_CONDITIONS[
        'Reference Samples & Images, Average Surround, Subtending < 4'])
LLAB_VIEWING_CONDITIONS['tv_dim'] = (
    LLAB_VIEWING_CONDITIONS[
        'Television & VDU Displays, Dim Surround'])
LLAB_VIEWING_CONDITIONS['sheet_dim'] = (
    LLAB_VIEWING_CONDITIONS[
        'Cut Sheet Transparency, Dim Surround'])
LLAB_VIEWING_CONDITIONS['projected_dark'] = (
    LLAB_VIEWING_CONDITIONS[
        '35mm Projection Transparency, Dark Surround'])

LLAB_XYZ_TO_RGB_MATRIX = np.array(
    [[0.8951, 0.2664, -0.1614],
     [-0.7502, 1.7135, 0.0367],
     [0.0389, -0.0685, 1.0296]])
"""
*LLAB(l:c)* colour appearance model *CIE XYZ* colourspace matrix to normalised
cone responses matrix.

LLAB_XYZ_TO_RGB_MATRIX : array_like, (3, 3)
"""

LLAB_RGB_TO_XYZ_MATRIX = np.around(
    np.linalg.inv(LLAB_XYZ_TO_RGB_MATRIX),
    decimals=4)
"""
*LLAB(l:c)* colour appearance model normalised cone responses to *CIE XYZ*
colourspace matrix.

Notes
-----
-   This matrix has been rounded on purpose to 4 decimals so that we keep
    consistency with **Mark D. Fairchild** implementation results.

LLAB_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""


class LLAB_ReferenceSpecification(
    namedtuple('LLAB_ReferenceSpecification',
               ('L_L', 'Ch_L', 'h_L', 's_L', 'C_L', 'HC', 'A_L', 'B_L'))):
    """
    Defines the *LLAB(l:c)* colour appearance model reference specification.

    This specification has field names consistent with **Mark D. Fairchild**
    reference.

    Parameters
    ----------
    L_L : numeric
        Correlate of *Lightness* :math:`L_L`.
    Ch_L : numeric
        Correlate of *chroma* :math:`Ch_L`.
    h_L : numeric
        *Hue* angle :math:`h_L` in degrees.
    s_L : numeric
        Correlate of *saturation* :math:`s_L`.
    C_L : numeric
        Correlate of *colourfulness* :math:`C_L`.
    HC : numeric
        *Hue* :math:`h` composition :math:`H^C`.
    A_L : numeric
        Opponent signal :math:`A_L`.
    B_L : numeric
        Opponent signal :math:`B_L`.
    """


class LLAB_Specification(
    namedtuple('LLAB_Specification',
               ('J', 'C', 'h', 's', 'M', 'HC', 'a', 'b'))):
    """
    Defines the *LLAB(l:c)* colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    **Mark D. Fairchild** reference.

    Parameters
    ----------
    J : numeric
        Correlate of *Lightness* :math:`L_L`.
    C : numeric
        Correlate of *chroma* :math:`Ch_L`.
    h : numeric
        *Hue* angle :math:`h_L` in degrees.
    s : numeric
        Correlate of *saturation* :math:`s_L`.
    M : numeric
        Correlate of *colourfulness* :math:`C_L`.
    HC : numeric
        *Hue* :math:`h` composition :math:`H^C`.
    a : numeric
        Opponent signal :math:`A_L`.
    b : numeric
        Opponent signal :math:`B_L`.
    """


def XYZ_to_LLAB(
        XYZ,
        XYZ_0,
        Y_b,
        L,
        surround=LLAB_VIEWING_CONDITIONS.get(
            'Reference Samples & Images, Average Surround, Subtending < 4')):
    """
    Computes the *LLAB(L:c)* colour appearance model correlates.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    XYZ_0 : array_like, (3,)
        *CIE XYZ* colourspace matrix of reference white in domain [0, 100].
    Y_b : numeric
        Luminance factor of the background in :math:`cd/m^2`.
    L : numeric
        Absolute luminance :math:`L` of reference white in :math:`cd/m^2`.
    surround : LLAB_InductionFactors, optional
         Surround viewing conditions induction factors.

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
    >>> XYZ = np.array([19.01, 20, 21.78])
    >>> XYZ_0 = np.array([95.05, 100, 108.88])
    >>> Y_b = 20.0
    >>> L = 318.31
    >>> surround = LLAB_VIEWING_CONDITIONS['ref_average_4_minus']
    >>> XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)  # doctest: +ELLIPSIS
    LLAB_Specification(J=37.3680474..., C=0.0086506..., h=229.4635727..., s=0.0002314..., M=0.0183832..., HC=None, a=-0.0119478..., b=-0.0139711...)
    """

    X, Y, Z = np.ravel(XYZ)
    RGB = XYZ_to_RGB_LLAB(XYZ)
    RGB_0 = XYZ_to_RGB_LLAB(XYZ_0)

    # Reference illuminant *CIE Standard Illuminant D Series* *D65*.
    XYZ_0r = np.array([95.05, 100, 108.88])
    RGB_0r = XYZ_to_RGB_LLAB(XYZ_0r)

    # Computing chromatic adaptation.
    XYZ_r = chromatic_adaptation(RGB, RGB_0, RGB_0r, Y, surround.D)

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`L_L`.
    # -------------------------------------------------------------------------
    # Computing opponent colour dimensions.
    L_L, a, b = opponent_colour_dimensions(XYZ_r,
                                           Y_b,
                                           surround.F_S,
                                           surround.F_L)

    # Computing perceptual correlates.
    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`Ch_L`.
    # -------------------------------------------------------------------------
    Ch_L = chroma_correlate(a, b)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`C_L`.
    # -------------------------------------------------------------------------
    C_L = colourfulness_correlate(L, L_L, Ch_L, surround.F_C)

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s_L`.
    # -------------------------------------------------------------------------
    s_L = saturation_correlate(Ch_L, L_L)

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`h_L`.
    # -------------------------------------------------------------------------
    h_L = hue_angle(a, b)
    h_Lr = math.radians(h_L)
    # TODO: Implement hue composition computation.

    # -------------------------------------------------------------------------
    # Computing final opponent signals.
    # -------------------------------------------------------------------------
    A_L, B_L = final_opponent_signals(C_L, h_Lr)

    return LLAB_Specification(L_L, Ch_L, h_L, s_L, C_L, None, A_L, B_L)


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
    >>> XYZ_to_RGB_LLAB(XYZ)  # doctest: +ELLIPSIS
    array([ 0.9414279...,  1.0404012...,  1.0897088...])
    """

    return LLAB_XYZ_TO_RGB_MATRIX.dot(XYZ / XYZ[1])


def chromatic_adaptation(RGB, RGB_0, RGB_0r, Y, D=1):
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
    Y : numeric
        Tristimulus values :math:`Y` of the stimulus.
    D : numeric, optional
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
    >>> chromatic_adaptation(RGB, RGB_0, RGB_0r, Y)  # doctest: +ELLIPSIS
    array([ 19.0099957...,  20.0009186...,  21.7799386...])
    """

    R, G, B = np.ravel(RGB)
    R_0, G_0, B_0 = np.ravel(RGB_0)
    R_0r, G_0r, B_0r = np.ravel(RGB_0r)

    beta = (B_0 / B_0r) ** 0.0834

    R_r = (D * (R_0r / R_0) + 1 - D) * R
    G_r = (D * (G_0r / G_0) + 1 - D) * G
    B_r = (D * (B_0r / (B_0 ** beta)) + 1 - D) * (abs(B) ** beta)

    RGB_r = np.array([R_r, G_r, B_r])

    X_r, Y_r, Z_r = LLAB_RGB_TO_XYZ_MATRIX.dot(RGB_r * Y)

    return np.array([X_r, Y_r, Z_r])


def f(x, F_S):
    """
    Defines the nonlinear response function of the *LLAB(L:c)* colour
    appearance model used to model the nonlinear behavior of various visual
    responses.

    Parameters
    ----------
    x : numeric or array_like
        Visual response variable :math:`x`.
    F_S : numeric
        Surround induction factor :math:`F_S`.

    Returns
    -------
    numeric or array_like
        Modeled visual response variable :math:`x`.

    Examples
    --------
    >>> x = np.array([0.23350512, 0.23351103, 0.23355179])
    >>> f(0.20000918623399996, 3)  # doctest: +ELLIPSIS
    array(0.5848125...)
    """

    x_m = np.where(x > 0.008856,
                   x ** (1 / F_S),
                   ((((0.008856 ** (1 / F_S)) -
                      (16 / 116)) / 0.008856) * x + (16 / 116)))
    return x_m


def opponent_colour_dimensions(XYZ, Y_b, F_S, F_L):
    """
    Returns opponent colour dimensions from given adapted *CIE XYZ* colourspace
    matrix.

    The opponent colour dimensions are based on a modified *CIE Lab*
    colourspace formulae.

    Parameters
    ----------
    XYZ : array_like, (3,)
        Adapted *CIE XYZ* colourspace matrix.
    Y_b : numeric
        Luminance factor of the background in :math:`cd/m^2`.
    F_S : numeric
        Surround induction factor :math:`F_S`.
    F_L : numeric
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
    >>> opponent_colour_dimensions(XYZ, Y_b, F_S, F_L)  # doctest: +ELLIPSIS
    array([  3.7368047...e+01,  -4.4986443...e-03,  -5.2604647...e-03])
    """

    X, Y, Z = np.ravel(XYZ)

    # Account for background lightness contrast.
    z = 1 + F_L * ((Y_b / 100) ** 0.5)

    # Computing modified *CIE Lab* colourspace matrix.
    L = 116 * (f(Y / 100, F_S) ** z) - 16
    a = 500 * (f(X / 95.05, F_S) - f(Y / 100, F_S))
    b = 200 * (f(Y / 100, F_S) - f(Z / 108.88, F_S))

    return np.array([L, a, b])


def hue_angle(a, b):
    """
    Returns the *hue* angle :math:`h_L` in degrees.

    Parameters
    ----------
    a : numeric
        Opponent colour dimension :math:`a`.
    b : numeric
        Opponent colour dimension :math:`b`.

    Returns
    -------
    numeric
        *Hue* angle :math:`h_L` in degrees.

    Examples
    --------
    >>> hue_angle(-4.49864756e-03, -5.26046353e-03)  # doctest: +ELLIPSIS
    229.4635727...
    """

    h_L = math.degrees(np.arctan2(b, a)) % 360
    return h_L


def chroma_correlate(a, b):
    """
    Returns the correlate of *chroma* :math:`Ch_L`.

    Parameters
    ----------
    a : numeric
        Opponent colour dimension :math:`a`.
    b : numeric
        Opponent colour dimension :math:`b`.

    Returns
    -------
    numeric
        Correlate of *chroma* :math:`Ch_L`.

    Examples
    --------
    >>> a = -4.49864756e-03
    >>> b = -5.26046353e-03
    >>> chroma_correlate(a, b)  # doctest: +ELLIPSIS
    0.0086506...
    """

    c = (a ** 2 + b ** 2) ** 0.5
    Ch_L = 25 * np.log(1 + 0.05 * c)
    return Ch_L


def colourfulness_correlate(L, L_L, Ch_L, F_C):
    """
    Returns the correlate of *colourfulness* :math:`C_L`.

    Parameters
    ----------
    L : numeric
        Absolute luminance :math:`L` of reference white in :math:`cd/m^2`.
    L_L : numeric
        Correlate of *Lightness* :math:`L_L`.
    Ch_L : numeric
        Correlate of *chroma* :math:`Ch_L`.
    F_C : numeric
        Chroma induction factor :math:`F_C`.

    Returns
    -------
    numeric
        Correlate of *colourfulness* :math:`C_L`.

    Examples
    --------
    >>> L = 318.31
    >>> L_L = 37.368047493928195
    >>> Ch_L = 0.0086506620517144972
    >>> F_C = 1.0
    >>> colourfulness_correlate(L, L_L, Ch_L, F_C)  # doctest: +ELLIPSIS
    0.0183832...
    """

    S_C = 1 + 0.47 * np.log10(L) - 0.057 * np.log10(L) ** 2
    S_M = 0.7 + 0.02 * L_L - 0.0002 * L_L ** 2
    C_L = Ch_L * S_M * S_C * F_C

    return C_L


def saturation_correlate(Ch_L, L_L):
    """
    Returns the correlate of *saturation* :math:`S_L`.

    Parameters
    ----------
    Ch_L : numeric
        Correlate of *chroma* :math:`Ch_L`.
    L_L : numeric
        Correlate of *Lightness* :math:`L_L`.

    Returns
    -------
    numeric
        Correlate of *saturation* :math:`S_L`.

    Examples
    --------
    >>> Ch_L = 0.0086506620517144972
    >>> L_L = 37.368047493928195
    >>> saturation_correlate(Ch_L, L_L)  # doctest: +ELLIPSIS
    0.0002314...
    """

    S_L = Ch_L / L_L

    return S_L


def final_opponent_signals(C_L, h_L):
    """
    Returns the final opponent signals :math:`A_L` and :math:`B_L`.

    Parameters
    ----------
    C_L : numeric
        Correlate of *colourfulness* :math:`C_L`.
    h_L : numeric
        Correlate of *hue* :math:`h_L` in radians.

    Returns
    -------
    tuple
        Final opponent signals :math:`A_L` and :math:`B_L`.

    Examples
    --------
    >>> C_L = 0.0183832899143
    >>> h_L = 4.004894857014253
    >>> final_opponent_signals(C_L, h_L)  # doctest: +ELLIPSIS
    (-0.0119478..., -0.0139711...)
    """

    A_L = C_L * np.cos(h_L)
    B_L = C_L * np.sin(h_L)

    return A_L, B_L
