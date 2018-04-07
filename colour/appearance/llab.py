# -*- coding: utf-8 -*-
"""
LLAB(l:c) Colour Appearance Model
=================================

Defines *LLAB(l:c)* colour appearance model objects:

-   :class:`colour.appearance.LLAB_InductionFactors`
-   :attr:`colour.LLAB_VIEWING_CONDITIONS`
-   :class:`colour.LLAB_Specification`
-   :func:`colour.XYZ_to_LLAB`

See Also
--------
`LLAB(l:c) Colour Appearance Model Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/appearance/llab.ipynb>`_

References
----------
-   :cite:`Fairchild2013x` : Fairchild, M. D. (2013). LLAB Model. In Color
    Appearance Models (3rd ed., pp. 6025-6178). Wiley. ISBN:B00DAYO8E2
-   :cite:`Luo1996b` : Luo, M. R., Lo, M.-C., & Kuo, W.-G. (1996). The LLAB
    (l:c) colour model. Color Research & Application, 21(6), 412-429.
    doi:10.1002/(SICI)1520-6378(199612)21:6<412::AID-COL4>3.0.CO;2-Z
-   :cite:`Luo1996c` : Luo, M. R., & Morovic, J. (1996). Two Unsolved Issues in
    Colour Management - Colour Appearance and Gamut Mapping. In Conference:
    5th International Conference on High Technology: Imaging Science and
    Technology - Evolution & Promise (pp. 136-147). Retrieved from
    http://www.researchgate.net/publication/\236348295_Two_Unsolved_Issues_in\
_Colour_Management__Colour_Appearance_and_Gamut_Mapping
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.algebra import polar_to_cartesian
from colour.utilities import (CaseInsensitiveMapping, dot_vector,
                              to_domain_100, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'LLAB_InductionFactors', 'LLAB_VIEWING_CONDITIONS',
    'LLAB_XYZ_TO_RGB_MATRIX', 'LLAB_RGB_TO_XYZ_MATRIX',
    'LLAB_ReferenceSpecification', 'LLAB_Specification', 'XYZ_to_LLAB',
    'XYZ_to_RGB_LLAB', 'chromatic_adaptation', 'f',
    'opponent_colour_dimensions', 'hue_angle', 'chroma_correlate',
    'colourfulness_correlate', 'saturation_correlate', 'final_opponent_signals'
]


class LLAB_InductionFactors(
        namedtuple('LLAB_InductionFactors', ('D', 'F_S', 'F_L', 'F_C'))):
    """
    *LLAB(l:c)* colour appearance model induction factors.

    Parameters
    ----------
    D : numeric or array_like
         *Discounting-the-Illuminant* factor :math:`D` normalised to domain
         [0, 1].
    F_S : numeric or array_like
        Surround induction factor :math:`F_S`.
    F_L : numeric or array_like
        *Lightness* induction factor :math:`F_L`.
    F_C : numeric or array_like
        *Chroma* induction factor :math:`F_C`.

    References
    ----------
    -   :cite:`Fairchild2013x`
    -   :cite:`Luo1996b`
    -   :cite:`Luo1996c`
    """


LLAB_VIEWING_CONDITIONS = CaseInsensitiveMapping({
    'Reference Samples & Images, Average Surround, Subtending > 4': (
        LLAB_InductionFactors(1, 3, 0, 1)),
    'Reference Samples & Images, Average Surround, Subtending < 4': (
        LLAB_InductionFactors(1, 3, 1, 1)),
    'Television & VDU Displays, Dim Surround': (LLAB_InductionFactors(
        0.7, 3.5, 1, 1)),
    'Cut Sheet Transparency, Dim Surround': (LLAB_InductionFactors(
        1, 5, 1, 1.1)),
    '35mm Projection Transparency, Dark Surround': (LLAB_InductionFactors(
        0.7, 4, 1, 1))
})
LLAB_VIEWING_CONDITIONS.__doc__ = """
Reference *LLAB(l:c)* colour appearance model viewing conditions.

References
----------
-   :cite:`Fairchild2013x`
-   :cite:`Luo1996b`
-   :cite:`Luo1996c`

LLAB_VIEWING_CONDITIONS : CaseInsensitiveMapping
    **{'Reference Samples & Images, Average Surround, Subtending > 4',
    'Reference Samples & Images, Average Surround, Subtending < 4',
    'Television & VDU Displays, Dim Surround',
    'Cut Sheet Transparency, Dim Surround':,
    '35mm Projection Transparency, Dark Surround'}**

Aliases:

-   'ref_average_4_plus':
    'Reference Samples & Images, Average Surround, Subtending > 4'
-   'ref_average_4_minus':
    'Reference Samples & Images, Average Surround, Subtending < 4'
-   'tv_dim': 'Television & VDU Displays, Dim Surround'
-   'sheet_dim': 'Cut Sheet Transparency, Dim Surround'
-   'projected_dark': '35mm Projection Transparency, Dark Surround'
"""
LLAB_VIEWING_CONDITIONS['ref_average_4_plus'] = (  # yapf: disable
    LLAB_VIEWING_CONDITIONS['Reference Samples & Images, '
                            'Average Surround, Subtending > 4'])
LLAB_VIEWING_CONDITIONS['ref_average_4_minus'] = (  # yapf: disable
    LLAB_VIEWING_CONDITIONS['Reference Samples & Images, '
                            'Average Surround, Subtending < 4'])
LLAB_VIEWING_CONDITIONS['tv_dim'] = (
    LLAB_VIEWING_CONDITIONS['Television & VDU Displays, Dim Surround'])
LLAB_VIEWING_CONDITIONS['sheet_dim'] = (
    LLAB_VIEWING_CONDITIONS['Cut Sheet Transparency, Dim Surround'])
LLAB_VIEWING_CONDITIONS['projected_dark'] = (
    LLAB_VIEWING_CONDITIONS['35mm Projection Transparency, Dark Surround'])

LLAB_XYZ_TO_RGB_MATRIX = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
])
"""
LLAB(l:c) colour appearance model *CIE XYZ* tristimulus values to normalised
cone responses matrix.

LLAB_XYZ_TO_RGB_MATRIX : array_like, (3, 3)
"""

LLAB_RGB_TO_XYZ_MATRIX = np.linalg.inv(LLAB_XYZ_TO_RGB_MATRIX)
"""
LLAB(l:c) colour appearance model normalised cone responses to *CIE XYZ*
tristimulus values matrix.

LLAB_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""


class LLAB_ReferenceSpecification(
        namedtuple('LLAB_ReferenceSpecification',
                   ('L_L', 'Ch_L', 'h_L', 's_L', 'C_L', 'HC', 'A_L', 'B_L'))):
    """
    Defines the *LLAB(l:c)* colour appearance model reference specification.

    This specification has field names consistent with *Fairchild (2013)*
    reference.

    Parameters
    ----------
    L_L : numeric or array_like
        Correlate of *Lightness* :math:`L_L`.
    Ch_L : numeric or array_like
        Correlate of *chroma* :math:`Ch_L`.
    h_L : numeric or array_like
        *Hue* angle :math:`h_L` in degrees.
    s_L : numeric or array_like
        Correlate of *saturation* :math:`s_L`.
    C_L : numeric or array_like
        Correlate of *colourfulness* :math:`C_L`.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H^C`.
    A_L : numeric or array_like
        Opponent signal :math:`A_L`.
    B_L : numeric or array_like
        Opponent signal :math:`B_L`.

    References
    ----------
    -   :cite:`Fairchild2013x`
    -   :cite:`Luo1996b`
    -   :cite:`Luo1996c`
    """


class LLAB_Specification(
        namedtuple('LLAB_Specification', ('J', 'C', 'h', 's', 'M', 'HC', 'a',
                                          'b'))):
    """
    Defines the *LLAB(l:c)* colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    *Fairchild (2013)* reference.

    Parameters
    ----------
    J : numeric or array_like
        Correlate of *Lightness* :math:`L_L`.
    C : numeric or array_like
        Correlate of *chroma* :math:`Ch_L`.
    h : numeric or array_like
        *Hue* angle :math:`h_L` in degrees.
    s : numeric or array_like
        Correlate of *saturation* :math:`s_L`.
    M : numeric or array_like
        Correlate of *colourfulness* :math:`C_L`.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H^C`.
    a : numeric or array_like
        Opponent signal :math:`A_L`.
    b : numeric or array_like
        Opponent signal :math:`B_L`.

    Notes
    -----
    -   This specification is the one used in the current model implementation.

    References
    ----------
    -   :cite:`Fairchild2013x`
    -   :cite:`Luo1996b`
    -   :cite:`Luo1996c`
    """


def XYZ_to_LLAB(
        XYZ,
        XYZ_0,
        Y_b,
        L,
        surround=LLAB_VIEWING_CONDITIONS[
            'Reference Samples & Images, Average Surround, Subtending < 4']):
    """
    Computes the *LLAB(l:c)* colour appearance model correlates.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of test sample / stimulus normalised to
        domain [0, 100].
    XYZ_0 : array_like
        *CIE XYZ* tristimulus values of reference white normalised to domain
        [0, 100].
    Y_b : numeric or array_like
        Luminance factor of the background in :math:`cd/m^2`.
    L : numeric or array_like
        Absolute luminance :math:`L` of reference white in :math:`cd/m^2`.
    surround : LLAB_InductionFactors, optional
         Surround viewing conditions induction factors.

    Returns
    -------
    LLAB_Specification
        *LLAB(l:c)* colour appearance model specification.

    Warning
    -------
    The output range of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are normalised to domain [0, 100].
    -   Input *CIE XYZ_0* tristimulus values are normalised to domain [0, 100].

    References
    ----------
    -   :cite:`Fairchild2013x`
    -   :cite:`Luo1996b`
    -   :cite:`Luo1996c`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_0 = np.array([95.05, 100.00, 108.88])
    >>> Y_b = 20.0
    >>> L = 318.31
    >>> surround = LLAB_VIEWING_CONDITIONS['ref_average_4_minus']
    >>> XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)  # doctest: +ELLIPSIS
    LLAB_Specification(J=37.3668650..., C=0.0089496..., h=270..., \
s=0.0002395..., M=0.0190185..., HC=None, a=..., b=-0.0190185...)
    """

    _X, Y, _Z = tsplit(to_domain_100(XYZ))
    RGB = XYZ_to_RGB_LLAB(to_domain_100(XYZ))
    RGB_0 = XYZ_to_RGB_LLAB(to_domain_100(XYZ_0))

    # Reference illuminant *CIE Standard Illuminant D Series* *D65*.
    XYZ_0r = np.array([95.05, 100.00, 108.88])
    RGB_0r = XYZ_to_RGB_LLAB(XYZ_0r)

    # Computing chromatic adaptation.
    XYZ_r = chromatic_adaptation(RGB, RGB_0, RGB_0r, Y, surround.D)

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`L_L`.
    # -------------------------------------------------------------------------
    # Computing opponent colour dimensions.
    L_L, a, b = tsplit(
        opponent_colour_dimensions(XYZ_r, Y_b, surround.F_S, surround.F_L))

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
    # TODO: Implement hue composition computation.

    # -------------------------------------------------------------------------
    # Computing final opponent signals.
    # -------------------------------------------------------------------------
    A_L, B_L = tsplit(final_opponent_signals(C_L, h_L))

    return LLAB_Specification(L_L, Ch_L, h_L, s_L, C_L, None, A_L, B_L)


def XYZ_to_RGB_LLAB(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to normalised cone responses.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        Normalised cone responses.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_to_RGB_LLAB(XYZ)  # doctest: +ELLIPSIS
    array([ 0.9414279...,  1.0404012...,  1.0897088...])
    """

    _X, Y, _Z = tsplit(XYZ)

    Y = tstack((Y, Y, Y))
    XYZ_n = XYZ / Y

    return dot_vector(LLAB_XYZ_TO_RGB_MATRIX, XYZ_n)


def chromatic_adaptation(RGB, RGB_0, RGB_0r, Y, D=1):
    """
    Applies chromatic adaptation to given *RGB* normalised cone responses
    array.

    Parameters
    ----------
    RGB : array_like
        *RGB* normalised cone responses array of test sample / stimulus.
    RGB_0 : array_like
        *RGB* normalised cone responses array of reference white.
    RGB_0r : array_like
        *RGB* normalised cone responses array of reference illuminant
        *CIE Standard Illuminant D Series* *D65*.
    Y : numeric or array_like
        Tristimulus values :math:`Y` of the stimulus.
    D : numeric or array_like, optional
         *Discounting-the-Illuminant* factor normalised to domain [0, 1].

    Returns
    -------
    ndarray
        Adapted *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> RGB = np.array([0.94142795, 1.04040120, 1.08970885])
    >>> RGB_0 = np.array([0.94146023, 1.04039386, 1.08950293])
    >>> RGB_0r = np.array([0.94146023, 1.04039386, 1.08950293])
    >>> Y = 20.0
    >>> chromatic_adaptation(RGB, RGB_0, RGB_0r, Y)  # doctest: +ELLIPSIS
    array([ 19.01,  20.  ,  21.78])
    """

    R, G, B = tsplit(RGB)
    R_0, G_0, B_0 = tsplit(RGB_0)
    R_0r, G_0r, B_0r = tsplit(RGB_0r)
    Y = np.asarray(Y)

    beta = (B_0 / B_0r) ** 0.0834

    R_r = (D * (R_0r / R_0) + 1 - D) * R
    G_r = (D * (G_0r / G_0) + 1 - D) * G
    B_r = (D * (B_0r / (B_0 ** beta)) + 1 - D) * (abs(B) ** beta)

    RGB_r = tstack((R_r, G_r, B_r))

    Y = tstack((Y, Y, Y))

    XYZ_r = dot_vector(LLAB_RGB_TO_XYZ_MATRIX, RGB_r * Y)

    return XYZ_r


def f(x, F_S):
    """
    Defines the nonlinear response function of the *LLAB(l:c)* colour
    appearance model used to model the nonlinear behaviour of various visual
    responses.

    Parameters
    ----------
    x : numeric or array_like or array_like
        Visual response variable :math:`x`.
    F_S : numeric or array_like
        Surround induction factor :math:`F_S`.

    Returns
    -------
    numeric or array_like
        Modeled visual response variable :math:`x`.

    Examples
    --------
    >>> x = np.array([0.23350512, 0.23351103, 0.23355179])
    >>> f(0.200009186234000, 3)  # doctest: +ELLIPSIS
    array(0.5848125...)
    """

    x = np.asarray(x)
    F_S = np.asarray(F_S)

    x_m = np.where(x > 0.008856,
                   x ** (1 / F_S),
                   ((((0.008856 ** (1 / F_S)) - (16 / 116)) / 0.008856) * x +
                    (16 / 116)))

    return x_m


def opponent_colour_dimensions(XYZ, Y_b, F_S, F_L):
    """
    Returns opponent colour dimensions from given adapted *CIE XYZ* tristimulus
    values.

    The opponent colour dimensions are based on a modified *CIE L\*a\*b\**
    colourspace formulae.

    Parameters
    ----------
    XYZ : array_like
        Adapted *CIE XYZ* tristimulus values.
    Y_b : numeric or array_like
        Luminance factor of the background in :math:`cd/m^2`.
    F_S : numeric or array_like
        Surround induction factor :math:`F_S`.
    F_L : numeric or array_like
        Lightness induction factor :math:`F_L`.

    Returns
    -------
    ndarray
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

    X, Y, Z = tsplit(XYZ)
    Y_b = np.asarray(Y_b)
    F_S = np.asarray(F_S)
    F_L = np.asarray(F_L)

    # Account for background lightness contrast.
    z = 1 + F_L * ((Y_b / 100) ** 0.5)

    # Computing modified *CIE L\*a\*b\** colourspace array.
    L = 116 * (f(Y / 100, F_S) ** z) - 16
    a = 500 * (f(X / 95.05, F_S) - f(Y / 100, F_S))
    b = 200 * (f(Y / 100, F_S) - f(Z / 108.88, F_S))

    Lab = tstack((L, a, b))

    return Lab


def hue_angle(a, b):
    """
    Returns the *hue* angle :math:`h_L` in degrees.

    Parameters
    ----------
    a : numeric or array_like
        Opponent colour dimension :math:`a`.
    b : numeric or array_like
        Opponent colour dimension :math:`b`.

    Returns
    -------
    numeric or ndarray
        *Hue* angle :math:`h_L` in degrees.

    Examples
    --------
    >>> hue_angle(-4.49864756e-03, -5.26046353e-03)  # doctest: +ELLIPSIS
    229.4635727...
    """

    a = np.asarray(a)
    b = np.asarray(b)

    h_L = np.degrees(np.arctan2(b, a)) % 360

    return h_L


def chroma_correlate(a, b):
    """
    Returns the correlate of *chroma* :math:`Ch_L`.

    Parameters
    ----------
    a : numeric or array_like
        Opponent colour dimension :math:`a`.
    b : numeric or array_like
        Opponent colour dimension :math:`b`.

    Returns
    -------
    numeric or ndarray
        Correlate of *chroma* :math:`Ch_L`.

    Examples
    --------
    >>> a = -4.49864756e-03
    >>> b = -5.26046353e-03
    >>> chroma_correlate(a, b)  # doctest: +ELLIPSIS
    0.0086506...
    """

    a = np.asarray(a)
    b = np.asarray(b)

    c = (a ** 2 + b ** 2) ** 0.5
    Ch_L = 25 * np.log(1 + 0.05 * c)

    return Ch_L


def colourfulness_correlate(L, L_L, Ch_L, F_C):
    """
    Returns the correlate of *colourfulness* :math:`C_L`.

    Parameters
    ----------
    L : numeric or array_like
        Absolute luminance :math:`L` of reference white in :math:`cd/m^2`.
    L_L : numeric or array_like
        Correlate of *Lightness* :math:`L_L`.
    Ch_L : numeric or array_like
        Correlate of *chroma* :math:`Ch_L`.
    F_C : numeric or array_like
        Chroma induction factor :math:`F_C`.

    Returns
    -------
    numeric or ndarray
        Correlate of *colourfulness* :math:`C_L`.

    Examples
    --------
    >>> L = 318.31
    >>> L_L = 37.368047493928195
    >>> Ch_L = 0.008650662051714
    >>> F_C = 1.0
    >>> colourfulness_correlate(L, L_L, Ch_L, F_C)  # doctest: +ELLIPSIS
    0.0183832...
    """

    L = np.asarray(L)
    L_L = np.asarray(L_L)
    Ch_L = np.asarray(Ch_L)
    F_C = np.asarray(F_C)

    S_C = 1 + 0.47 * np.log10(L) - 0.057 * np.log10(L) ** 2
    S_M = 0.7 + 0.02 * L_L - 0.0002 * L_L ** 2
    C_L = Ch_L * S_M * S_C * F_C

    return C_L


def saturation_correlate(Ch_L, L_L):
    """
    Returns the correlate of *saturation* :math:`S_L`.

    Parameters
    ----------
    Ch_L : numeric or array_like
        Correlate of *chroma* :math:`Ch_L`.
    L_L : numeric or array_like
        Correlate of *Lightness* :math:`L_L`.

    Returns
    -------
    numeric or ndarray
        Correlate of *saturation* :math:`S_L`.

    Examples
    --------
    >>> Ch_L = 0.008650662051714
    >>> L_L = 37.368047493928195
    >>> saturation_correlate(Ch_L, L_L)  # doctest: +ELLIPSIS
    0.0002314...
    """

    Ch_L = np.asarray(Ch_L)
    L_L = np.asarray(L_L)

    S_L = Ch_L / L_L

    return S_L


def final_opponent_signals(C_L, h_L):
    """
    Returns the final opponent signals :math:`A_L` and :math:`B_L`.

    Parameters
    ----------
    C_L : numeric or array_like
        Correlate of *colourfulness* :math:`C_L`.
    h_L : numeric or array_like
        Correlate of *hue* :math:`h_L` in degrees.

    Returns
    -------
    ndarray
        Final opponent signals :math:`A_L` and :math:`B_L`.

    Examples
    --------
    >>> C_L = 0.0183832899143
    >>> h_L = 229.46357270858391
    >>> final_opponent_signals(C_L, h_L)  # doctest: +ELLIPSIS
    array([-0.0119478..., -0.0139711...])
    """

    AB_L = polar_to_cartesian(tstack((C_L, np.radians(h_L))))

    return AB_L
