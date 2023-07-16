"""
CIECAM16 Colour Appearance Model
================================

Defines the *CIECAM16* colour appearance model objects:

-   :class:`colour.appearance.InductionFactors_CIECAM16`
-   :attr:`colour.VIEWING_CONDITIONS_CIECAM16`
-   :class:`colour.CAM_Specification_CIECAM16`
-   :func:`colour.XYZ_to_CIECAM16`
-   :func:`colour.CIECAM16_to_XYZ`

References
----------
-   :cite:`CIEDivision12022` : CIE Division 1 & CIE Division 8. (2022).
    CIE 248:2022 The CIE 2016 Colour Appearance Model for Colour Management
    Systems: CIECAM16. Commission Internationale de l'Eclairage.
    ISBN:978-3-902842-94-7
"""

from __future__ import annotations

import numpy as np
from collections import namedtuple
from dataclasses import astuple, dataclass, field

from colour.algebra import spow, vector_dot
from colour.appearance.cam16 import MATRIX_16, MATRIX_INVERSE_16
from colour.appearance.ciecam02 import (
    InductionFactors_CIECAM02,
    VIEWING_CONDITIONS_CIECAM02,
    P,
    achromatic_response_forward,
    achromatic_response_inverse,
    brightness_correlate,
    chroma_correlate,
    colourfulness_correlate,
    degree_of_adaptation,
    eccentricity_factor,
    hue_angle,
    hue_quadrature,
    lightness_correlate,
    opponent_colour_dimensions_forward,
    opponent_colour_dimensions_inverse,
    post_adaptation_non_linear_response_compression_forward,
    matrix_post_adaptation_non_linear_response_compression,
    saturation_correlate,
    temporary_magnitude_quantity_inverse,
    viewing_conditions_dependent_parameters,
)
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArithmetic,
    as_float,
    as_float_array,
    from_range_100,
    from_range_degrees,
    has_only_nan,
    ones,
    to_domain_100,
    to_domain_degrees,
    tsplit,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "InductionFactors_CIECAM16",
    "VIEWING_CONDITIONS_CIECAM16",
    "CAM_Specification_CIECAM16",
    "XYZ_to_CIECAM16",
    "CIECAM16_to_XYZ",
    "f_e_forward",
    "f_e_inverse",
    "f_q",
    "d_f_q",
]


class InductionFactors_CIECAM16(
    namedtuple("InductionFactors_CIECAM16", ("F", "c", "N_c"))
):
    """
    *CIECAM16* colour appearance model induction factors.

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
    -   The *CIECAM16* colour appearance model induction factors are the same
        as *CIECAM02* colour appearance model.

    References
    ----------
    :cite:`CIEDivision12022`
    """


VIEWING_CONDITIONS_CIECAM16: CanonicalMapping = CanonicalMapping(
    VIEWING_CONDITIONS_CIECAM02
)
VIEWING_CONDITIONS_CIECAM16.__doc__ = """
Reference *CIECAM16* colour appearance model viewing conditions.

References
----------
:cite:`CIEDivision12022`
"""


@dataclass
class CAM_Specification_CIECAM16(MixinDataclassArithmetic):
    """
    Define the *CIECAM16* colour appearance model specification.

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
    :cite:`CIEDivision12022`
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_CIECAM16(
    XYZ: ArrayLike,
    XYZ_w: ArrayLike,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: InductionFactors_CIECAM02
    | InductionFactors_CIECAM16 = VIEWING_CONDITIONS_CIECAM16["Average"],
    discount_illuminant: bool = False,
    compute_H: bool = True,
) -> CAM_Specification_CIECAM16:
    """
    Compute the *CIECAM16* colour appearance model correlates from given
    *CIE XYZ* tristimulus values.

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
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.
    compute_H
        Whether to compute *Hue* :math:`h` quadrature :math:`H`. :math:`H` is
        rarely used, and expensive to compute.

    Returns
    -------
    :class:`colour.CAM_Specification_CIECAM16`
        *CIECAM16* colour appearance model specification.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_w``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +----------------------------------+-----------------------\
+---------------+
    | **Range**                        | **Scale - Reference** \
| **Scale - 1** |
    +==================================+=======================\
+===============+
    | ``CAM_Specification_CIECAM16.J`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.C`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.h`` | [0, 360]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.s`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.Q`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.M`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.H`` | [0, 400]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+

    References
    ----------
    :cite:`CIEDivision12022`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_CIECAM16["Average"]
    >>> XYZ_to_CIECAM16(XYZ, XYZ_w, L_A, Y_b, surround)  # doctest: +ELLIPSIS
    CAM_Specification_CIECAM16(J=41.7312079..., C=0.1033557..., \
h=217.0679597..., s=2.3450150..., Q=195.3717089..., M=0.1074367..., \
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
        np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    n, F_L, N_bb, N_cb, z = viewing_conditions_dependent_parameters(
        Y_b, Y_w, L_A
    )

    D_RGB = D[..., None] * 100 / RGB_w + 1 - D[..., None]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_wc, F_L
    )

    # Computing achromatic responses for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    # Step 1
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB = vector_dot(MATRIX_16, XYZ)

    # Step 2
    RGB_c = D_RGB * RGB

    # Step 3
    # Applying forward post-adaptation non-linear response compression.
    RGB_a = f_e_forward(RGB_c, F_L) + 0.1

    # Step 4
    # Converting to preliminary cartesian coordinates.
    a, b = tsplit(opponent_colour_dimensions_forward(RGB_a))

    # Computing the *hue* angle :math:`h`.
    h = hue_angle(a, b)

    # Step 5
    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h) if compute_H else np.full(h.shape, np.nan)
    # TODO: Compute hue composition.

    # Step 6
    # Computing achromatic responses for the stimulus.
    A = achromatic_response_forward(RGB_a, N_bb)

    # Step 7
    # Computing the correlate of *Lightness* :math:`J`.
    J = lightness_correlate(A, A_w, surround.c, z)

    # Step 8
    # Computing the correlate of *brightness* :math:`Q`.
    Q = brightness_correlate(surround.c, J, A_w, F_L)

    # Step 9
    # Computing the correlate of *chroma* :math:`C`.
    C = chroma_correlate(J, n, surround.N_c, N_cb, e_t, a, b, RGB_a)

    # Computing the correlate of *colourfulness* :math:`M`.
    M = colourfulness_correlate(C, F_L)

    # Computing the correlate of *saturation* :math:`s`.
    s = saturation_correlate(M, Q)

    return CAM_Specification_CIECAM16(
        as_float(from_range_100(J)),
        as_float(from_range_100(C)),
        as_float(from_range_degrees(h)),
        as_float(from_range_100(s)),
        as_float(from_range_100(Q)),
        as_float(from_range_100(M)),
        as_float(from_range_degrees(H, 400)),
        None,
    )


def CIECAM16_to_XYZ(
    specification: CAM_Specification_CIECAM16,
    XYZ_w: ArrayLike,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: InductionFactors_CIECAM02
    | InductionFactors_CIECAM16 = VIEWING_CONDITIONS_CIECAM16["Average"],
    discount_illuminant: bool = False,
) -> NDArrayFloat:
    """
    Convert from *CIECAM16* specification to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    specification
        *CIECAM16* colour appearance model specification. Correlate of
        *Lightness* :math:`J`, correlate of *chroma* :math:`C` or correlate of
        *colourfulness* :math:`M` and *hue* angle :math:`h` in degrees must be
        specified, e.g. :math:`JCh` or :math:`JMh`.
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
    discount_illuminant
        Discount the illuminant.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If neither :math:`C` or :math:`M` correlates have been defined in the
        ``specification`` argument.

    Notes
    -----
    +----------------------------------+-----------------------\
+---------------+
    | **Domain**                       | **Scale - Reference** \
| **Scale - 1** |
    +==================================+=======================\
+===============+
    | ``CAM_Specification_CIECAM16.J`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.C`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.h`` | [0, 360]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.s`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.Q`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.M`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM16.H`` | [0, 360]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``XYZ_w``                        | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`CIEDivision12022`

    Examples
    --------
    >>> specification = CAM_Specification_CIECAM16(
    ...     J=41.731207905126638, C=0.103355738709070, h=217.067959767393010
    ... )
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> CIECAM16_to_XYZ(specification, XYZ_w, L_A, Y_b)  # doctest: +ELLIPSIS
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
        np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    n, F_L, N_bb, N_cb, z = viewing_conditions_dependent_parameters(
        Y_b, Y_w, L_A
    )

    D_RGB = D[..., None] * 100 / RGB_w + 1 - D[..., None]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_wc, F_L
    )

    # Computing achromatic responses for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    # Step 1
    if has_only_nan(C) and not has_only_nan(M):
        C = M / spow(F_L, 0.25)
    elif has_only_nan(C):
        raise ValueError(
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_CIECAM16" argument!'
        )

    # Step 2
    # Computing temporary magnitude quantity :math:`t`.
    t = temporary_magnitude_quantity_inverse(C, J, n)

    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing achromatic response :math:`A` for the stimulus.
    A = achromatic_response_inverse(A_w, J, surround.c, z)

    # Computing *P_1* to *P_3*.
    P_n = P(surround.N_c, N_cb, e_t, t, A, N_bb)
    _P_1, P_2, _P_3 = tsplit(P_n)

    # Step 3
    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    ab = opponent_colour_dimensions_inverse(P_n, h)
    a, b = tsplit(ab) * np.where(t == 0, 0, 1)

    # Step 4
    # Applying post-adaptation non-linear response compression matrix.
    RGB_a = matrix_post_adaptation_non_linear_response_compression(P_2, a, b)

    # Step 5
    # Applying inverse post-adaptation non-linear response compression.
    RGB_c = f_e_inverse(RGB_a - 0.1, F_L)

    # Step 6
    RGB = RGB_c / D_RGB

    # Step 7
    XYZ = vector_dot(MATRIX_INVERSE_16, RGB)

    return from_range_100(XYZ)


def f_e_forward(RGB_c: ArrayLike, F_L: ArrayLike) -> NDArrayFloat:
    """
    Compute the post-adaptation cone responses.

    Parameters
    ----------
    RGB_c
        *CMCCAT2000* transform sharpened :math:`RGB_c` array.
    F_L
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    :class:`numpy.ndarray`
        Compressed *CMCCAT2000* transform sharpened :math:`RGB_a` array.

    Notes
    -----
    -   This definition is different from :cite:`Li2017` and provides linear
        extensions under 0.26 and above 150. It also omits the 0.1 offset that
        is now part of the general model.

    Examples
    --------
    >>> RGB_c = np.array([19.99693975, 20.00186123, 20.01350530])
    >>> F_L = 1.16754446415
    >>> f_e_forward(RGB_c, F_L)
    ... # doctest: +ELLIPSIS
    array([ 7.8463202...,  7.8471152...,  7.8489959...])
    """

    RGB_c = as_float_array(RGB_c)
    F_L = as_float_array(F_L)
    q_L, q_U = 0.26, 150

    return np.select(
        [
            RGB_c > q_U,
            np.logical_and(q_L <= RGB_c, RGB_c <= q_U),
            RGB_c < q_L,
        ],
        [
            f_q(F_L, q_U) + d_f_q(F_L, q_U) * (RGB_c - q_U),
            f_q(F_L, RGB_c),
            f_q(F_L, q_L) * RGB_c / q_L,
        ],
    )


def f_e_inverse(RGB_a: ArrayLike, F_L: ArrayLike) -> NDArrayFloat:
    """
    Compute the modified cone-like responses.

    Parameters
    ----------
    RGB_a
        *CMCCAT2000* transform sharpened :math:`RGB_a` array.
    F_L
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    :class:`numpy.ndarray`
        Compressed *CMCCAT2000* transform sharpened :math:`RGB_c` array.

    Notes
    -----
    -   This definition is different from :cite:`Li2017` and provides linear
        extensions under 0.26 and above 150. It also omits the 0.1 offset that
        is now part of the general model.

    Examples
    --------
    >>> RGB_a = np.array([7.8463202, 7.84711528, 7.84899595])
    >>> F_L = 1.16754446415
    >>> f_e_inverse(RGB_a, F_L)
    ... # doctest: +ELLIPSIS
    array([ 19.9969397...,  20.0018612...,  20.0135052...])
    """

    RGB_a = as_float_array(RGB_a)
    F_L = as_float_array(F_L)
    q_L, q_U = 0.26, 150

    return np.select(
        [
            RGB_a > f_q(F_L, q_U),
            np.logical_and(f_q(F_L, q_L) <= RGB_a, RGB_a <= f_q(F_L, q_U)),
            RGB_a < f_q(F_L, q_L),
        ],
        [
            q_U + (RGB_a - f_q(F_L, q_U)) / d_f_q(F_L, q_U),
            100
            / F_L[..., None]
            * spow((27.13 * RGB_a) / (400 - RGB_a), 1 / 0.42),
            q_L * RGB_a / f_q(F_L, q_L),
        ],
    )


def f_q(F_L: ArrayLike, q: ArrayLike) -> NDArrayFloat:
    """
    Define the :math:`f(q)` function.

    Parameters
    ----------
    F_L
        *Luminance* level adaptation factor :math:`F_L`.
    q
        :math:`q` parameter.

    Returns
    -------
    :class:`numpy.ndarray`
        Evaluated :math:`f(q)` function.

    Examples
    --------
    >>> f_q(1.17, 0.26)  # doctest: +ELLIPSIS
    1.2886520...
    """

    F_L = as_float_array(F_L)
    q = as_float_array(q)

    F_L = F_L[..., None]

    F_L_q_100 = spow((F_L * q) / 100, 0.42)

    return (400 * F_L_q_100) / (27.13 + F_L_q_100)


def d_f_q(F_L: ArrayLike, q: ArrayLike) -> NDArrayFloat:
    """
    Define the :math:`f'(q)` function derivative.

    Parameters
    ----------
    F_L
        *Luminance* level adaptation factor :math:`F_L`.
    q
        :math:`q` parameter.

    Returns
    -------
    :class:`numpy.ndarray`
        Evaluated :math:`f'(q)` function derivative.

    Examples
    --------
    >>> d_f_q(1.17, 0.26)  # doctest: +ELLIPSIS
    array([ 2.0749623...])
    """

    F_L = as_float_array(F_L)
    q = as_float_array(q)

    F_L = F_L[..., None]

    F_L_q_100 = (F_L * q) / 100

    return (1.68 * 27.13 * F_L * spow(F_L_q_100, -0.58)) / (
        27.13 + spow(F_L_q_100, 0.42)
    ) ** 2
