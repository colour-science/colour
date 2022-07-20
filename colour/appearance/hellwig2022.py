"""
Hellwig and Fairchild (2022) Colour Appearance Model
====================================================

Defines the *Hellwig and Fairchild (2022)* colour appearance model objects:

-   :class:`colour.appearance.InductionFactors_Hellwig2022`
-   :attr:`colour.VIEWING_CONDITIONS_HELLWIG2022`
-   :class:`colour.CAM_Specification_Hellwig2022`
-   :func:`colour.XYZ_to_Hellwig2022`
-   :func:`colour.Hellwig2022_to_XYZ`

References
----------
-   :cite:`Fairchild2022` : Fairchild, M. D., & Hellwig, L. (2022). Private
    Discussion with Mansencal, T.
-   :cite:`Hellwig2022` : Hellwig, L., & Fairchild, M. D. (2022). Brightness,
    lightness, colorfulness, and chroma in CIECAM02 and CAM16. Color Research
    & Application, col.22792. doi:10.1002/col.22792
"""

from __future__ import annotations

import numpy as np
from collections import namedtuple
from dataclasses import astuple, dataclass, field

from colour.algebra import sdiv, sdiv_mode, spow, vector_dot
from colour.appearance.cam16 import MATRIX_16, MATRIX_INVERSE_16
from colour.appearance.ciecam02 import (
    InductionFactors_CIECAM02,
    VIEWING_CONDITIONS_CIECAM02,
    hue_quadrature,
    post_adaptation_non_linear_response_compression_forward,
    post_adaptation_non_linear_response_compression_inverse,
)
from colour.hints import (
    ArrayLike,
    Boolean,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
    Optional,
    Union,
)
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArithmetic,
    as_float,
    as_float_array,
    from_range_100,
    from_range_degrees,
    full,
    has_only_nan,
    ones,
    to_domain_100,
    to_domain_degrees,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "InductionFactors_Hellwig2022",
    "VIEWING_CONDITIONS_HELLWIG2022",
    "CAM_Specification_Hellwig2022",
    "XYZ_to_Hellwig2022",
    "Hellwig2022_to_XYZ",
]


class InductionFactors_Hellwig2022(
    namedtuple("InductionFactors_Hellwig2022", ("F", "c", "N_c"))
):
    """
    *Hellwig and Fairchild (2022)* colour appearance model induction factors.

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
    -   The *Hellwig and Fairchild (2022)* colour appearance model induction
        factors are the same as *CIECAM02* and *CAM16* colour appearance model.

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`
    """


VIEWING_CONDITIONS_HELLWIG2022: CanonicalMapping = CanonicalMapping(
    VIEWING_CONDITIONS_CIECAM02
)
VIEWING_CONDITIONS_HELLWIG2022.__doc__ = """
Reference *Hellwig and Fairchild (2022)* colour appearance model viewing
conditions.

References
----------
:cite:`Hellwig2022`
"""


@dataclass
class CAM_Specification_Hellwig2022(MixinDataclassArithmetic):
    """
    Define the *Hellwig and Fairchild (2022)* colour appearance model
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
    :cite:`Fairchild2022`, :cite:`Hellwig2022`
    """

    J: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    C: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    h: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    s: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    Q: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    M: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    H: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    HC: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)


def d_post_adaptation_non_linear_response_compression_forward(
    RGB: ArrayLike, F_L: FloatingOrArrayLike
) -> NDArray:
    F_L_RGB = spow(F_L[..., np.newaxis] * RGB / 100, 0.42)
    F_L_100 = spow(F_L[..., np.newaxis] / 100, 0.42)

    d_RGB_a = (
        400
        * ((0.42 * 27.13) * spow(RGB, -0.58) * F_L_100)
        / (F_L_RGB + 27.13) ** 2
    )

    return d_RGB_a


def XYZ_to_Hellwig2022(
    XYZ: ArrayLike,
    XYZ_w: ArrayLike,
    L_A: FloatingOrArrayLike,
    Y_b: FloatingOrArrayLike,
    surround: Union[
        InductionFactors_CIECAM02, InductionFactors_Hellwig2022
    ] = VIEWING_CONDITIONS_HELLWIG2022["Average"],
    L_B: FloatingOrArrayLike = 0.01,
    discount_illuminant: Boolean = False,
) -> CAM_Specification_Hellwig2022:
    """
    Compute the *Hellwig and Fairchild (2022)* colour appearance model
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
    Y_b
        Luminous factor of background :math:`Y_b` such as
        :math:`Y_b = 100 x L_b / L_w` where :math:`L_w` is the luminance of the
        light source and :math:`L_b` is the luminance of the background. For
        viewing images, :math:`Y_b` can be the average :math:`Y` value for the
        pixels in the entire image, or frequently, a :math:`Y` value of 20,
        approximate an :math:`L^*` of 50 is used.
    surround
        Surround viewing conditions induction factors.
    L_B
        Breaking point for the linear extension of the post-adaptation
        non-linear response compression.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    :class:`colour.CAM_Specification_Hellwig2022`
        *Hellwig and Fairchild (2022)* colour appearance model specification.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_w``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +-------------------------------------+-----------------------+-----------\
----+
    | **Range**                           | **Scale - Reference** | **Scale - \
1** |
    +=====================================+=======================+===========\
====+
    | ``CAM_Specification_Hellwig2022.J`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.C`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.h`` | [0, 360]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.s`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.Q`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.M`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.H`` | [0, 400]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_HELLWIG2022['Average']
    >>> XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround)
    ... # doctest: +ELLIPSIS
    CAM_Specification_Hellwig2022(J=41.7312079..., C=0.0257636..., \
h=217.0679597..., s=0.0608550..., Q=55.8523226..., M=0.0339889..., \
H=275.5949861..., HC=None)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    # Step 0
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB_w = vector_dot(MATRIX_16, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (
        np.clip(surround.F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92)), 0, 1)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    # Viewing conditions dependent parameters
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    n = sdiv(Y_b, Y_w)
    z = 1.48 + np.sqrt(n)

    D_RGB = (
        D[..., np.newaxis] * Y_w[..., np.newaxis] / RGB_w
        + 1
        - D[..., np.newaxis]
    )
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    F_L_RGB_w = spow(F_L[..., np.newaxis] * np.absolute(RGB_wc) / 100, 0.42)
    RGB_aw = (400 * np.sign(RGB_wc) * F_L_RGB_w) / (27.13 + F_L_RGB_w) + 0.1

    # Computing achromatic responses for the whitepoint.
    R_aw, G_aw, B_aw = tsplit(RGB_aw)
    A_w = 2 * R_aw + G_aw + 0.05 * B_aw - 0.305

    # Step 1
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB = vector_dot(MATRIX_16, XYZ)

    # Step 2
    RGB_c = D_RGB * RGB

    # Step 3
    # Applying forward post-adaptation non-linear response compression.
    RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L)
    RGB_a_l = d_post_adaptation_non_linear_response_compression_forward(
        full(3, L_B), F_L
    ) * (
        RGB_c - L_B
    ) + post_adaptation_non_linear_response_compression_forward(
        full(3, L_B), F_L
    )
    RGB_a = np.where(RGB_c < L_B, RGB_a_l, RGB_a)

    # Step 4
    # Converting to preliminary cartesian coordinates.
    R_a, G_a, B_a = tsplit(RGB_a)
    a = R_a - 12 * G_a / 11 + B_a / 11
    b = (R_a + G_a - 2 * B_a) / 9

    # Computing the *hue* angle :math:`h`.
    h = np.degrees(np.arctan2(b, a)) % 360

    # Step 5
    # Computing eccentricity factor *e_t*.
    hr = np.radians(h)

    _h = hr
    _2_h = 2 * hr
    _3_h = 3 * hr
    _4_h = 4 * hr

    e_t = (
        -0.0582 * np.cos(_h)
        - 0.0258 * np.cos(_2_h)
        - 0.1347 * np.cos(_3_h)
        + 0.0289 * np.cos(_4_h)
        - 0.1475 * np.sin(_h)
        - 0.0308 * np.sin(_2_h)
        + 0.0385 * np.sin(_3_h)
        + 0.0096 * np.sin(_4_h)
        + 1
    )

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h)
    # TODO: Compute hue composition.

    # Step 6
    # Computing achromatic responses for the stimulus.
    R_a, G_a, B_a = tsplit(RGB_a)
    A = 2 * R_a + G_a + 0.05 * B_a - 0.305

    # Step 7
    # Computing the correlate of *Lightness* :math:`J`.
    with sdiv_mode():
        J = 100 * spow(sdiv(A, A_w), surround.c * z)

    # Step 8
    # Computing the correlate of *brightness* :math:`Q`.
    with sdiv_mode():
        Q = (2 / as_float(surround.c)) * (J / 100) * A_w

    # Step 9
    # Computing the correlate of *colourfulness* :math:`M`.
    M = 43 * surround.N_c * e_t * np.sqrt(a**2 + b**2)

    # Computing the correlate of *chroma* :math:`C`.
    with sdiv_mode():
        C = 35 * sdiv(M, A_w)

    # Computing the correlate of *saturation* :math:`s`.
    with sdiv_mode():
        s = 100 * sdiv(M, Q)

    return CAM_Specification_Hellwig2022(
        as_float(from_range_100(J)),
        as_float(from_range_100(C)),
        as_float(from_range_degrees(h)),
        as_float(from_range_100(s)),
        as_float(from_range_100(Q)),
        as_float(from_range_100(M)),
        as_float(from_range_degrees(H, 400)),
        None,
    )


def Hellwig2022_to_XYZ(
    specification: CAM_Specification_Hellwig2022,
    XYZ_w: ArrayLike,
    L_A: FloatingOrArrayLike,
    Y_b: FloatingOrArrayLike,
    surround: Union[
        InductionFactors_CIECAM02, InductionFactors_Hellwig2022
    ] = VIEWING_CONDITIONS_HELLWIG2022["Average"],
    L_B: FloatingOrArrayLike = 0.01,
    discount_illuminant: Boolean = False,
) -> NDArray:
    """
    Convert from *Hellwig and Fairchild (2022)* specification to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    specification : CAM_Specification_Hellwig2022
        *Hellwig and Fairchild (2022)* colour appearance model specification.
        Correlate of *Lightness* :math:`J`, correlate of *chroma* :math:`C` or
        correlate of *colourfulness* :math:`M` and *hue* angle :math:`h` in
        degrees must be specified, e.g. :math:`JCh` or :math:`JMh`.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    Y_b
        Luminous factor of background :math:`Y_b` such as
        :math:`Y_b = 100 x L_b / L_w` where :math:`L_w` is the luminance of the
        light source and :math:`L_b` is the luminance of the background. For
        viewing images, :math:`Y_b` can be the average :math:`Y` value for the
        pixels in the entire image, or frequently, a :math:`Y` value of 20,
        approximate an :math:`L^*` of 50 is used.
    surround
        Surround viewing conditions.
    L_B
        Breaking point for the linear extension of the post-adaptation
        non-linear response compression.
    discount_illuminant
        Discount the illuminant.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If neither *C* or *M* correlates have been defined in the
        ``CAM_Specification_Hellwig2022`` argument.

    Notes
    -----
    +-------------------------------------+-----------------------+-----------\
----+
    | **Domain**                          | **Scale - Reference** | **Scale - \
1** |
    +=====================================+=======================+===========\
====+
    | ``CAM_Specification_Hellwig2022.J`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.C`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.h`` | [0, 360]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.s`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.Q`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.M`` | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``CAM_Specification_Hellwig2022.H`` | [0, 360]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+
    | ``XYZ_w``                           | [0, 100]              | [0, 1]    \
    |
    +-------------------------------------+-----------------------+-----------\
----+

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`

    Examples
    --------
    >>> specification = CAM_Specification_Hellwig2022(J=41.731207905126638,
    ...                                               C=0.025763615829912909,
    ...                                               h=217.06795976739301)
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b)
    ... # doctest: +ELLIPSIS
    array([ 19.01...,  20...  ,  21.78...])
    """

    J, C, h, _s, _Q, M, _H, _HC = astuple(specification)

    J = to_domain_100(J)
    C = to_domain_100(C)
    h = to_domain_degrees(h)
    M = to_domain_100(M)
    L_A = as_float_array(L_A)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)

    # Step 0
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB_w = vector_dot(MATRIX_16, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (
        np.clip(surround.F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92)), 0, 1)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    # Viewing conditions dependent parameters
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    n = sdiv(Y_b, Y_w)
    z = 1.48 + np.sqrt(n)

    D_RGB = (
        D[..., np.newaxis] * Y_w[..., np.newaxis] / RGB_w
        + 1
        - D[..., np.newaxis]
    )
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    F_L_RGB_w = spow(F_L[..., np.newaxis] * np.absolute(RGB_wc) / 100, 0.42)
    RGB_aw = (400 * np.sign(RGB_wc) * F_L_RGB_w) / (27.13 + F_L_RGB_w) + 0.1

    # Computing achromatic responses for the whitepoint.
    R_aw, G_aw, B_aw = tsplit(RGB_aw)
    A_w = 2 * R_aw + G_aw + 0.05 * B_aw - 0.305

    # Step 1
    if has_only_nan(M) and not has_only_nan(C):
        M = (C * A_w) / 35
    elif has_only_nan(M):
        raise ValueError(
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_Hellwig2022" argument!'
        )

    # Step 2
    # Computing eccentricity factor *e_t*.
    hr = np.radians(h)

    _h = hr
    _2_h = 2 * hr
    _3_h = 3 * hr
    _4_h = 4 * hr

    e_t = (
        -0.0582 * np.cos(_h)
        - 0.0258 * np.cos(_2_h)
        - 0.1347 * np.cos(_3_h)
        + 0.0289 * np.cos(_4_h)
        - 0.1475 * np.sin(_h)
        - 0.0308 * np.sin(_2_h)
        + 0.0385 * np.sin(_3_h)
        + 0.0096 * np.sin(_4_h)
        + 1
    )

    # Computing achromatic response :math:`A` for the stimulus.
    A = A = A_w * spow(J / 100, 1 / (surround.c * z))

    # Computing *P_p_1* to *P_p_2*.
    P_p_1 = 43 * surround.N_c * e_t
    P_p_2 = A

    # Step 3
    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    with sdiv_mode():
        gamma = M / P_p_1

    a = gamma * np.cos(hr)
    b = gamma * np.sin(hr)

    # Step 4
    # Applying post-adaptation non-linear response compression matrix.
    RGB_a = (
        vector_dot(
            [
                [460, 451, 288],
                [460, -891, -261],
                [460, -220, -6300],
            ],
            tstack([P_p_2, a, b]),
        )
        / 1403
    )

    # Step 5
    # Applying inverse post-adaptation non-linear response compression.
    RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L)
    RGB_c_l = (
        RGB_a
        - post_adaptation_non_linear_response_compression_forward(
            full(3, L_B), F_L
        )
    ) / (
        d_post_adaptation_non_linear_response_compression_forward(
            full(3, L_B), F_L
        )
    ) + L_B
    RGB_c = np.where(RGB_c < L_B, RGB_c_l, RGB_c)

    # Step 6
    RGB = RGB_c / D_RGB

    # Step 7
    XYZ = vector_dot(MATRIX_INVERSE_16, RGB)

    return from_range_100(XYZ)
