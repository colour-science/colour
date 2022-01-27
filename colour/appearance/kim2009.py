# -*- coding: utf-8 -*-
"""
Kim, Weyrich and Kautz (2009) Colour Appearance Model
=====================================================

Defines the *Kim, Weyrich and Kautz (2009)* colour appearance model objects:

-   :class:`colour.appearance.InductionFactors_Kim2009`
-   :attr:`colour.VIEWING_CONDITIONS_KIM2009`
-   :class:`colour.appearance.MediaParameters_Kim2009`
-   :attr:`colour.MEDIA_PARAMETERS_KIM2009`
-   :class:`colour.CAM_Specification_Kim2009`
-   :func:`colour.XYZ_to_Kim2009`
-   :func:`colour.Kim2009_to_XYZ`

References
----------
-   :cite:`Kim2009` : Kim, M., Weyrich, T., & Kautz, J. (2009). Modeling Human
    Color Perception under Extended Luminance Levels. ACM Transactions on
    Graphics, 28(3), 27:1--27:9. doi:10.1145/1531326.1531333
"""

from __future__ import annotations

import numpy as np
from collections import namedtuple
from dataclasses import astuple, dataclass, field

from colour.adaptation import CAT_CAT02
from colour.appearance.ciecam02 import (
    VIEWING_CONDITIONS_CIECAM02,
    CAT_INVERSE_CAT02,
    RGB_to_rgb,
    degree_of_adaptation,
    full_chromatic_adaptation_forward,
    full_chromatic_adaptation_inverse,
    hue_quadrature,
    rgb_to_RGB,
)
from colour.algebra import vector_dot, spow
from colour.hints import (
    ArrayLike,
    Boolean,
    Floating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
    Optional,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    MixinDataclassArray,
    as_float,
    as_float_array,
    from_range_100,
    from_range_degrees,
    has_only_nan,
    ones,
    to_domain_100,
    to_domain_degrees,
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
    'InductionFactors_Kim2009',
    'VIEWING_CONDITIONS_KIM2009',
    'MediaParameters_Kim2009',
    'MEDIA_PARAMETERS_KIM2009',
    'CAM_Specification_Kim2009',
    'XYZ_to_Kim2009',
    'Kim2009_to_XYZ',
]


class InductionFactors_Kim2009(
        namedtuple('InductionFactors_Kim2009', ('F', 'c', 'N_c'))):
    """
    *Kim, Weyrich and Kautz (2009)* colour appearance model induction factors.

    Parameters
    ----------
    F
        Maximum degree of adaptation :math:`F`.
    c
        Exponential non-linearity :math:`c`.
    N_c
        Chromatic induction factor :math:`N_c`.

    Notes
    -----
    -   The *Kim, Weyrich and Kautz (2009)* colour appearance model induction
        factors are the same as *CIECAM02* colour appearance model.
    -   The *Kim, Weyrich and Kautz (2009)* colour appearance model separates
        the surround modelled by the
        :class:`colour.appearance.InductionFactors_Kim2009` class instance from
        the media, modeled with the
        :class:`colour.appearance.MediaParameters_Kim2009` class instance.

    References
    ----------
    :cite:`Kim2009`
    """


VIEWING_CONDITIONS_KIM2009: CaseInsensitiveMapping = CaseInsensitiveMapping(
    VIEWING_CONDITIONS_CIECAM02)
VIEWING_CONDITIONS_KIM2009.__doc__ = """
Reference *Kim, Weyrich and Kautz (2009)* colour appearance model viewing
conditions.

References
----------
:cite:`Kim2009`
"""


class MediaParameters_Kim2009(namedtuple('MediaParameters_Kim2009', ('E', ))):
    """
    *Kim, Weyrich and Kautz (2009)* colour appearance model media parameters.

    Parameters
    ----------
    E
        Lightness prediction modulating parameter :math:`E`.

    References
    ----------
    :cite:`Kim2009`
    """

    def __new__(cls, E):
        """
        Returns a new instance of the
        :class:`colour.appearance.MediaParameters_Kim2009` class.
        """

        return super(MediaParameters_Kim2009, cls).__new__(cls, E)


MEDIA_PARAMETERS_KIM2009: CaseInsensitiveMapping = CaseInsensitiveMapping({
    'High-luminance LCD Display': MediaParameters_Kim2009(1),
    'Transparent Advertising Media': MediaParameters_Kim2009(1.2175),
    'CRT Displays': MediaParameters_Kim2009(1.4572),
    'Reflective Paper': MediaParameters_Kim2009(1.7526)
})
MEDIA_PARAMETERS_KIM2009.__doc__ = """
Reference *Kim, Weyrich and Kautz (2009)* colour appearance model media
parameters.

References
----------
:cite:`Kim2009`

Aliases:

-   'bright_lcd_display': 'High-luminance LCD Display'
-   'advertising_transparencies': 'Transparent Advertising Media'
-   'crt': 'CRT Displays'
-   'paper': 'Reflective Paper'
"""
MEDIA_PARAMETERS_KIM2009['bright_lcd_display'] = (
    MEDIA_PARAMETERS_KIM2009['High-luminance LCD Display'])
MEDIA_PARAMETERS_KIM2009['advertising_transparencies'] = (
    MEDIA_PARAMETERS_KIM2009['Transparent Advertising Media'])
MEDIA_PARAMETERS_KIM2009['crt'] = (MEDIA_PARAMETERS_KIM2009['CRT Displays'])
MEDIA_PARAMETERS_KIM2009['paper'] = (
    MEDIA_PARAMETERS_KIM2009['Reflective Paper'])


@dataclass
class CAM_Specification_Kim2009(MixinDataclassArray):
    """
    Defines the *Kim, Weyrich and Kautz (2009)* colour appearance model
    specification.

    Parameters
    ----------
    J
        Correlate of *Lightness* :math:`J`.
    C
        Correlate of *chroma* :math:`C`.
    h
        *Hue* angle :math:`h` in degrees.
    s
        Correlate of *saturation* :math:`s`.
    Q
        Correlate of *brightness* :math:`Q`.
    M
        Correlate of *colourfulness* :math:`M`.
    H
        *Hue* :math:`h` quadrature :math:`H`.
    HC
        *Hue* :math:`h` composition :math:`H^C`.

    References
    ----------
    :cite:`Kim2009`
    """

    J: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    C: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    h: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    s: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    Q: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    M: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    H: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    HC: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)


def XYZ_to_Kim2009(
        XYZ: ArrayLike,
        XYZ_w: ArrayLike,
        L_A: FloatingOrArrayLike,
        media: MediaParameters_Kim2009 = MEDIA_PARAMETERS_KIM2009[
            'CRT Displays'],
        surround: InductionFactors_Kim2009 = VIEWING_CONDITIONS_KIM2009[
            'Average'],
        discount_illuminant: Boolean = False,
        n_c: Floating = 0.57) -> CAM_Specification_Kim2009:
    """
    Computes the *Kim, Weyrich and Kautz (2009)* colour appearance model
    correlates from given *CIE XYZ* tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    media
        Media parameters.
    surround
        Surround viewing conditions induction factors.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.
    n_c
        Cone response sigmoidal curve modulating factor :math:`n_c`.

    Returns
    -------
    :class:`colour.CAM_Specification_Kim2009`
       *Kim, Weyrich and Kautz (2009)* colour appearance model specification.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_w``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +---------------------------------+-----------------------+---------------+
    | **Range**                       | **Scale - Reference** | **Scale - 1** |
    +=================================+=======================+===============+
    | ``CAM_Specification_Kim2009.J`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.C`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.h`` | [0, 360]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.s`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.Q`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.M`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.H`` | [0, 400]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Kim2009`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> media = MEDIA_PARAMETERS_KIM2009['CRT Displays']
    >>> surround = VIEWING_CONDITIONS_KIM2009['Average']
    >>> XYZ_to_Kim2009(XYZ, XYZ_w, L_A, media, surround)
    ... # doctest: +ELLIPSIS
    CAM_Specification_Kim2009(J=28.8619089..., C=0.5592455..., \
h=219.0480667..., s=9.3837797..., Q=52.7138883..., M=0.4641738..., \
H=278.0602824..., HC=None)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = as_float_array(L_A)

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB = vector_dot(CAT_CAT02, XYZ)
    RGB_w = vector_dot(CAT_CAT02, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (degree_of_adaptation(surround.F, L_A)
         if not discount_illuminant else ones(L_A.shape))

    # Computing full chromatic adaptation.
    XYZ_c = full_chromatic_adaptation_forward(RGB, RGB_w, Y_w, D)
    XYZ_wc = full_chromatic_adaptation_forward(RGB_w, RGB_w, Y_w, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    LMS = RGB_to_rgb(XYZ_c)
    LMS_w = RGB_to_rgb(XYZ_wc)

    # Cones absolute response.
    LMS_n_c = spow(LMS, n_c)
    LMS_w_n_c = spow(LMS_w, n_c)
    L_A_n_c = spow(L_A, n_c)
    LMS_p = LMS_n_c / (LMS_n_c + L_A_n_c)
    LMS_wp = LMS_w_n_c / (LMS_w_n_c + L_A_n_c)

    # Achromatic signal :math:`A` and :math:`A_w`.
    v_A = np.array([40, 20, 1])
    A = np.sum(v_A * LMS_p, axis=-1) / 61
    A_w = np.sum(v_A * LMS_wp, axis=-1) / 61

    # Perceived *Lightness* :math:`J_p`.
    a_j, b_j, o_j, n_j = 0.89, 0.24, 0.65, 3.65
    A_A_w = A / A_w
    J_p = spow((-(A_A_w - b_j) * spow(o_j, n_j)) / (A_A_w - b_j - a_j),
               1 / n_j)

    # Computing the media dependent *Lightness* :math:`J`.
    J = 100 * (media.E * (J_p - 1) + 1)

    # Computing the correlate of *brightness* :math:`Q`.
    n_q = 0.1308
    Q = J * spow(Y_w, n_q)

    # Opponent signals :math:`a` and :math:`b`.
    a = (1 / 11) * np.sum(np.array([11, -12, 1]) * LMS_p, axis=-1)
    b = (1 / 9) * np.sum(np.array([1, 1, -2]) * LMS_p, axis=-1)

    # Computing the correlate of *chroma* :math:`C`.
    a_k, n_k = 456.5, 0.62
    C = a_k * spow(np.sqrt(a ** 2 + b ** 2), n_k)

    # Computing the correlate of *colourfulness* :math:`M`.
    a_m, b_m = 0.11, 0.61
    M = C * (a_m * np.log10(Y_w) + b_m)

    # Computing the correlate of *saturation* :math:`s`.
    s = 100 * np.sqrt(M / Q)

    # Computing the *hue* angle :math:`h`.
    h = np.degrees(np.arctan2(b, a)) % 360

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h)

    return CAM_Specification_Kim2009(
        as_float(from_range_100(J)),
        as_float(from_range_100(C)),
        as_float(from_range_degrees(h)),
        as_float(from_range_100(s)),
        as_float(from_range_100(Q)),
        as_float(from_range_100(M)),
        as_float(from_range_degrees(H, 400)),
        None,
    )


def Kim2009_to_XYZ(
        specification: CAM_Specification_Kim2009,
        XYZ_w: ArrayLike,
        L_A: FloatingOrArrayLike,
        media: MediaParameters_Kim2009 = MEDIA_PARAMETERS_KIM2009[
            'CRT Displays'],
        surround: InductionFactors_Kim2009 = VIEWING_CONDITIONS_KIM2009[
            'Average'],
        discount_illuminant: Boolean = False,
        n_c: Floating = 0.57) -> NDArray:
    """
    Converts from *Kim, Weyrich and Kautz (2009)* specification to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    specification
         *Kim, Weyrich and Kautz (2009)* colour appearance model specification.
         Correlate of *Lightness* :math:`J`, correlate of *chroma* :math:`C` or
         correlate of *colourfulness* :math:`M` and *hue* angle :math:`h` in
         degrees must be specified, e.g. :math:`JCh` or :math:`JMh`.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    media
        Media parameters.
    surroundl
        Surround viewing conditions induction factors.
    discount_illuminant
        Discount the illuminant.
    n_c
        Cone response sigmoidal curve modulating factor :math:`n_c`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If neither *C* or *M* correlates have been defined in the
        ``CAM_Specification_Kim2009`` argument.

    Notes
    -----

    +---------------------------------+-----------------------+---------------+
    | **Domain**                      | **Scale - Reference** | **Scale - 1** |
    +=================================+=======================+===============+
    | ``CAM_Specification_Kim2009.J`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.C`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.h`` | [0, 360]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.s`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.Q`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.M`` | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``CAM_Specification_Kim2009.H`` | [0, 360]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+
    | ``XYZ_w``                       | [0, 100]              | [0, 1]        |
    +---------------------------------+-----------------------+---------------+

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`Kim2009`

    Examples
    --------
    >>> specification = CAM_Specification_Kim2009(J=28.861908975839647,
    ...                                           C=0.5592455924373706,
    ...                                           h=219.04806677662953)
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> media = MEDIA_PARAMETERS_KIM2009['CRT Displays']
    >>> surround = VIEWING_CONDITIONS_KIM2009['Average']
    >>> Kim2009_to_XYZ(specification, XYZ_w, L_A, media, surround)
    ... # doctest: +ELLIPSIS
    array([ 19.0099995...,  19.9999999...,  21.7800000...])
    """

    J, C, h, _s, _Q, M, _H, _HC = astuple(specification)

    J = to_domain_100(J)
    C = to_domain_100(C)
    h = to_domain_degrees(h)
    M = to_domain_100(M)
    L_A = as_float_array(L_A)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB_w = vector_dot(CAT_CAT02, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (degree_of_adaptation(surround.F, L_A)
         if not discount_illuminant else ones(L_A.shape))

    # Computing full chromatic adaptation.
    XYZ_wc = full_chromatic_adaptation_forward(RGB_w, RGB_w, Y_w, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    LMS_w = RGB_to_rgb(XYZ_wc)

    # n_q = 0.1308
    # J = Q / spow(Y_w, n_q)
    if has_only_nan(C) and not has_only_nan(M):
        a_m, b_m = 0.11, 0.61
        C = M / (a_m * np.log10(Y_w) + b_m)
    elif has_only_nan(C):
        raise ValueError('Either "C" or "M" correlate must be defined in '
                         'the "CAM_Specification_Kim2009" argument!')

    # Cones absolute response.
    LMS_w_n_c = spow(LMS_w, n_c)
    L_A_n_c = spow(L_A, n_c)
    LMS_wp = LMS_w_n_c / (LMS_w_n_c + L_A_n_c)

    # Achromatic signal :math:`A_w`
    v_A = np.array([40, 20, 1])
    A_w = np.sum(v_A * LMS_wp, axis=-1) / 61

    # Perceived *Lightness* :math:`J_p`.
    J_p = (J / 100 - 1) / media.E + 1

    # Achromatic signal :math:`A`.
    a_j, b_j, n_j, o_j = 0.89, 0.24, 3.65, 0.65
    J_p_n_j = spow(J_p, n_j)
    A = A_w * ((a_j * J_p_n_j) / (J_p_n_j + spow(o_j, n_j)) + b_j)

    # Opponent signals :math:`a` and :math:`b`.
    a_k, n_k = 456.5, 0.62
    C_a_k_n_k = spow(C / a_k, 1 / n_k)
    hr = np.radians(h)
    a, b = np.cos(hr) * C_a_k_n_k, np.sin(hr) * C_a_k_n_k

    # Cones absolute response.
    M = np.array([
        [1.0000, 0.3215, 0.2053],
        [1.0000, -0.6351, -0.1860],
        [1.0000, -0.1568, -4.4904],
    ])
    LMS_p = vector_dot(M, tstack([A, a, b]))
    LMS = spow((-spow(L_A, n_c) * LMS_p) / (LMS_p - 1), 1 / n_c)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_c = rgb_to_RGB(LMS)

    # Applying inverse full chromatic adaptation.
    RGB = full_chromatic_adaptation_inverse(RGB_c, RGB_w, Y_w, D)

    XYZ = vector_dot(CAT_INVERSE_CAT02, RGB)

    return from_range_100(XYZ)
