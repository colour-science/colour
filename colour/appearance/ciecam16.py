"""
CIECAM16 Colour Appearance Model
================================

Define the *CIECAM16* colour appearance model for predicting perceptual colour
attributes under varying viewing conditions.

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

import typing
from dataclasses import astuple, dataclass, field

from colour.algebra import sdiv, sdiv_mode, spow, vecmul
from colour.appearance.cam16 import MATRIX_16, MATRIX_INVERSE_16
from colour.appearance.ciecam02 import (
    VIEWING_CONDITIONS_CIECAM02,
    InductionFactors_CIECAM02,
    hue_quadrature,
)
from colour.constants import EPSILON

if typing.TYPE_CHECKING:
    from colour.hints import (
        Annotated,
        ArrayLike,
        Domain100,
        NDArrayFloat,
        Range100,
    )

from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArithmetic,
    MixinDataclassIterable,
    array_namespace,
    as_float,
    from_range_100,
    from_range_degrees,
    has_only_nan,
    ones,
    to_domain_100,
    to_domain_degrees,
    tsplit,
    tstack,
    xp_as_float_array,
    xp_degrees,
    xp_radians,
    xp_select,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "InductionFactors_CIECAM16",
    "VIEWING_CONDITIONS_CIECAM16",
    "CAM_Specification_CIECAM16",
    "XYZ_to_CIECAM16",
    "CIECAM16_to_XYZ",
]


@dataclass(frozen=True)
class InductionFactors_CIECAM16(MixinDataclassIterable):
    """
    Define the *CIECAM16* colour appearance model induction factors.

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

    F: float
    c: float
    N_c: float


VIEWING_CONDITIONS_CIECAM16: CanonicalMapping = CanonicalMapping(
    VIEWING_CONDITIONS_CIECAM02
)
VIEWING_CONDITIONS_CIECAM16.__doc__ = """
Define the reference *CIECAM16* colour appearance model viewing conditions.

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
        Correlate of *lightness* :math:`J`.
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
    XYZ: Domain100,
    XYZ_w: Domain100,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: (
        InductionFactors_CIECAM02 | InductionFactors_CIECAM16
    ) = VIEWING_CONDITIONS_CIECAM16["Average"],
    discount_illuminant: bool = False,
    compute_H: bool = False,
) -> Annotated[CAM_Specification_CIECAM16, (100, 100, 360, 100, 100, 100, 400)]:
    """
    Compute the *CIECAM16* colour appearance model correlates from the
    specified *CIE XYZ* tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often
        taken to be 20% of the luminance of a white object in the scene).
    Y_b
        Luminous factor of background :math:`Y_b` such as
        :math:`Y_b = 100 \\times L_b / L_w` where :math:`L_w` is the
        luminance of the light source and :math:`L_b` is the luminance of
        the background. For viewing images, :math:`Y_b` can be the average
        :math:`Y` value for the pixels in the entire image, or frequently,
        a :math:`Y` value of 20, approximate an :math:`L^*` of 50 is used.
    surround
        Surround viewing conditions induction factors.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.
    compute_H
        When *True*, compute the *Hue Quadrature* :math:`H` correlate
        via :func:`colour.appearance.hue_quadrature`. Defaults to
        *False* because :math:`H` is rarely consumed downstream and
        skipping the bin search is a measurable cost saving.

    Returns
    -------
    :class:`colour.CAM_Specification_CIECAM16`
        *CIECAM16* colour appearance model specification.

    Notes
    -----
    +---------------------+-----------------------+---------------+
    | **Domain**          | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``XYZ``             | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``XYZ_w``           | 100                   | 1             |
    +---------------------+-----------------------+---------------+

    +---------------------+-----------------------+---------------+
    | **Range**           | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``specification.J`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.C`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.h`` | 360                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.s`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.Q`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.M`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.H`` | 400                   | 1             |
    +---------------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIEDivision12022`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_CIECAM16["Average"]
    >>> XYZ_to_CIECAM16(
    ...     XYZ, XYZ_w, L_A, Y_b, surround,
    ...     compute_H=True,
    ... )  # doctest: +ELLIPSIS
    CAM_Specification_CIECAM16(J=np.float64(41.7312079...), \
C=np.float64(0.1033557...), h=np.float64(217.0679597...), \
s=np.float64(2.3450150...), Q=np.float64(195.3717089...), \
M=np.float64(0.1074367...), H=np.float64(275.5949861...), HC=None)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)

    xp = array_namespace(XYZ, XYZ_w, L_A, Y_b)

    XYZ = xp_as_float_array(XYZ, xp=xp)
    XYZ_w = xp_as_float_array(XYZ_w, xp=xp, like=XYZ)
    L_A = xp_as_float_array(L_A, xp=xp, like=XYZ)
    Y_b = xp_as_float_array(Y_b, xp=xp, like=XYZ)

    _X_w, Y_w, _Z_w = tsplit(XYZ_w)

    # Viewing condition dependent parameters: background induction
    # factor :math:`n`, luminance level adaptation factor :math:`F_L`,
    # chromatic induction factors :math:`N_{bb}` and :math:`N_{cb}`,
    # base exponential non-linearity :math:`z`. Same formulation as
    # in *CIECAM02*.
    with sdiv_mode():
        n = sdiv(Y_b, Y_w)
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    with sdiv_mode():
        N_bb = 0.725 * spow(sdiv(1, n), 0.2)
    N_cb = N_bb
    z = 1.48 + xp.sqrt(n)

    # Converting *CIE XYZ* tristimulus values to *CAT16* sharpened *RGB*
    # values for the stimulus and the reference white, same matrix as
    # *CAM16*.
    RGB = vecmul(MATRIX_16, XYZ)
    RGB_w = vecmul(MATRIX_16, XYZ_w)

    # Computing degree of adaptation :math:`D`, same formulation as in
    # *CIECAM02*, clipped to :math:`[0, 1]` and bypassed entirely when
    # ``discount_illuminant`` is set.
    if discount_illuminant:
        D = xp_as_float_array(ones(L_A.shape), xp=xp, like=XYZ)
    else:
        F = xp_as_float_array(surround.F, xp=xp, like=XYZ)
        D = xp.clip(F * (1 - (1 / 3.6) * xp.exp((-L_A - 42) / 92)), 0, 1)

    # Computing full chromatic adaptation. *CIECAM16* uses :math:`100`
    # in place of :math:`Y_w` in the adaptation factor, unlike
    # *CIECAM02* / *CAM16*.
    D_RGB = D[..., None] * 100 / RGB_w + 1 - D[..., None]
    RGB_c = D_RGB * RGB
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression
    # via the *CIECAM16* 3-branch piecewise function with linear
    # extensions outside the :math:`[0.26, 150]` range. The :math:`+0.1`
    # offset is added back per the model definition. The whitepoint goes
    # through *CIECAM02*'s original Michaelis-Menten compression
    # :math:`(400 \\cdot |x|^{0.42}) / (27.13 + |x|^{0.42}) + 0.1` per
    # the *CIECAM16* spec.
    F_L_RGB_wc = spow(F_L[..., None] * xp.abs(RGB_wc) / 100, 0.42)
    RGB_aw = (400 * xp.sign(RGB_wc) * F_L_RGB_wc) / (27.13 + F_L_RGB_wc) + 0.1

    q_L, q_U = 0.26, 150
    F_L_q_L_p = spow((F_L[..., None] * q_L) / 100, 0.42)
    f_q_F_L_q_L = (400 * F_L_q_L_p) / (27.13 + F_L_q_L_p)
    F_L_q_U_p = spow((F_L[..., None] * q_U) / 100, 0.42)
    f_q_F_L_q_U = (400 * F_L_q_U_p) / (27.13 + F_L_q_U_p)
    F_L_q_U_lin = (F_L[..., None] * q_U) / 100
    d_f_q_F_L_q_U = (1.68 * 27.13 * F_L[..., None] * spow(F_L_q_U_lin, -0.58)) / (
        27.13 + spow(F_L_q_U_lin, 0.42)
    ) ** 2
    F_L_RGB_c_lin = (F_L[..., None] * RGB_c) / 100
    F_L_RGB_c_p = spow(F_L_RGB_c_lin, 0.42)
    f_q_F_L_RGB_c = (400 * F_L_RGB_c_p) / (27.13 + F_L_RGB_c_p)
    RGB_a = (
        xp_select(
            [
                RGB_c > q_U,
                xp.logical_and(q_L <= RGB_c, RGB_c <= q_U),
                RGB_c < q_L,
            ],
            [
                f_q_F_L_q_U + d_f_q_F_L_q_U * (RGB_c - q_U),
                f_q_F_L_RGB_c,
                f_q_F_L_q_L * RGB_c / q_L,
            ],
            xp=xp,
        )
        + 0.1
    )

    # Computing the opponent colour dimensions :math:`a` and :math:`b`,
    # same as in *CIECAM02*.
    Ra, Ga, Ba = tsplit(RGB_a)
    a = Ra - 12 * Ga / 11 + Ba / 11
    b = (Ra + Ga - 2 * Ba) / 9

    # Computing the *hue* angle :math:`h` in degrees in
    # :math:`[0, 360)`, same as in *CIECAM02*.
    h = xp_degrees(xp.atan2(b, a)) % 360

    # Computing eccentricity factor :math:`e_t`, same as in *CIECAM02*.
    e_t = 1 / 4 * (xp.cos(2 + xp_radians(h)) + 3.8)

    # Computing achromatic responses :math:`A` for the stimulus and
    # :math:`A_w` for the whitepoint, same as in *CIECAM02*.
    A = (2 * Ra + Ga + (1 / 20) * Ba - 0.305) * N_bb
    Raw, Gaw, Baw = tsplit(RGB_aw)
    A_w = (2 * Raw + Gaw + (1 / 20) * Baw - 0.305) * N_bb

    # Computing the correlate of *Lightness* :math:`J`, same form as
    # in *CIECAM02*.
    c = surround.c
    with sdiv_mode():
        J = 100 * spow(sdiv(A, A_w), c * z)

    # Computing the correlate of *brightness* :math:`Q`, same form as
    # in *CIECAM02*.
    Q = (4 / c) * xp.sqrt(J / 100) * (A_w + 4) * spow(F_L, 0.25)

    # Computing the temporary magnitude quantity :math:`t` and the
    # correlate of *chroma* :math:`C`, same forms as in *CIECAM02*.
    N_c = surround.N_c
    with sdiv_mode():
        t = ((50000 / 13) * N_c * N_cb) * sdiv(
            e_t * spow(a**2 + b**2, 0.5), Ra + Ga + 21 * Ba / 20
        )
    C = spow(t, 0.9) * spow(J / 100, 0.5) * spow(1.64 - 0.29**n, 0.73)

    # Computing the correlate of *colourfulness* :math:`M` and the
    # correlate of *saturation* :math:`s`, same forms as in *CIECAM02*.
    M = C * spow(F_L, 0.25)
    with sdiv_mode():
        s = 100 * spow(sdiv(M, Q), 0.5)

    # Computing hue :math:`h` quadrature :math:`H` only when requested
    # via ``compute_H``; the bin search is shared with *CIECAM02* and
    # delegates to :func:`hue_quadrature`.
    # TODO: Compute hue composition.
    H = hue_quadrature(h) if compute_H else xp.full_like(h, float("nan"))

    return CAM_Specification_CIECAM16(
        J=as_float(from_range_100(J)),
        C=as_float(from_range_100(C)),
        h=as_float(from_range_degrees(h)),
        s=as_float(from_range_100(s)),
        Q=as_float(from_range_100(Q)),
        M=as_float(from_range_100(M)),
        H=as_float(from_range_degrees(H, 400)),
        HC=None,
    )


def CIECAM16_to_XYZ(
    specification: Annotated[
        CAM_Specification_CIECAM16, (100, 100, 360, 100, 100, 100, 400)
    ],
    XYZ_w: Domain100,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: (
        InductionFactors_CIECAM02 | InductionFactors_CIECAM16
    ) = VIEWING_CONDITIONS_CIECAM16["Average"],
    discount_illuminant: bool = False,
) -> Range100:
    """
    Convert the *CIECAM16* colour appearance model specification to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    specification
        *CIECAM16* colour appearance model specification. Correlate of
        *lightness* :math:`J`, correlate of *chroma* :math:`C` or correlate of
        *colourfulness* :math:`M` and *hue* angle :math:`h` in degrees must be
        specified, e.g., :math:`JCh` or :math:`JMh`.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    Y_b
        Luminous factor of background :math:`Y_b` such as
        :math:`Y_b = 100 \\times L_b / L_w` where :math:`L_w` is the luminance
        of the light source and :math:`L_b` is the luminance of the background.
        For viewing images, :math:`Y_b` can be the average :math:`Y` value for
        the pixels in the entire image, or frequently, a :math:`Y` value of 20,
        approximating an :math:`L^*` of 50 is used.
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
    +---------------------+-----------------------+---------------+
    | **Domain**          | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``specification.J`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.C`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.h`` | 360                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.s`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.Q`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.M`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.H`` | 360                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``XYZ_w``           | 100                   | 1             |
    +---------------------+-----------------------+---------------+

    +---------------------+-----------------------+---------------+
    | **Range**           | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``XYZ``             | 100                   | 1             |
    +---------------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIEDivision12022`

    Examples
    --------
    >>> import numpy as np
    >>> specification = CAM_Specification_CIECAM16(
    ...     J=41.731207905126638, C=0.103355738709070, h=217.067959767393010
    ... )
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> CIECAM16_to_XYZ(specification, XYZ_w, L_A, Y_b)  # doctest: +ELLIPSIS
    array([19.01..., 20...  , 21.78...])
    """

    J, C, h, _s, _Q, M, _H, _HC = astuple(specification)

    J = to_domain_100(J)
    C = to_domain_100(C)
    h = to_domain_degrees(h)
    M = to_domain_100(M)
    XYZ_w = to_domain_100(XYZ_w)

    xp = array_namespace(J, C, h, M, XYZ_w, L_A)

    J = xp_as_float_array(J, xp=xp)
    C = xp_as_float_array(C, xp=xp, like=J)
    h = xp_as_float_array(h, xp=xp, like=J)
    M = xp_as_float_array(M, xp=xp, like=J)
    XYZ_w = xp_as_float_array(XYZ_w, xp=xp, like=J)
    L_A = xp_as_float_array(L_A, xp=xp, like=J)

    _X_w, Y_w, _Z_w = tsplit(XYZ_w)

    # Viewing condition dependent parameters: background induction
    # factor :math:`n`, luminance level adaptation factor :math:`F_L`,
    # chromatic induction factors :math:`N_{bb}` and :math:`N_{cb}`,
    # base exponential non-linearity :math:`z`. Same formulation as
    # in *CIECAM02*.
    with sdiv_mode():
        n = sdiv(Y_b, Y_w)
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    with sdiv_mode():
        N_bb = 0.725 * spow(sdiv(1, n), 0.2)
    N_cb = N_bb
    z = 1.48 + xp.sqrt(n)

    # Converting *CIE XYZ* tristimulus values to *CAT16* sharpened *RGB*
    # values for the reference white.
    RGB_w = vecmul(MATRIX_16, XYZ_w)

    # Computing degree of adaptation :math:`D`, same formulation as in
    # *CIECAM02*, clipped to :math:`[0, 1]` and bypassed entirely when
    # ``discount_illuminant`` is set.
    if discount_illuminant:
        D = xp_as_float_array(ones(L_A.shape), xp=xp, like=J)
    else:
        F = xp_as_float_array(surround.F, xp=xp, like=J)
        D = xp.clip(F * (1 - (1 / 3.6) * xp.exp((-L_A - 42) / 92)), 0, 1)

    # Computing full chromatic adaptation for the reference white.
    # *CIECAM16* uses :math:`100` in place of :math:`Y_w` in the
    # adaptation factor, unlike *CIECAM02* / *CAM16*.
    D_RGB = D[..., None] * 100 / RGB_w + 1 - D[..., None]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression
    # to the whitepoint, same form as in *CIECAM02* per the *CIECAM16*
    # spec.
    F_L_RGB_wc = spow(F_L[..., None] * xp.abs(RGB_wc) / 100, 0.42)
    RGB_aw = (400 * xp.sign(RGB_wc) * F_L_RGB_wc) / (27.13 + F_L_RGB_wc) + 0.1

    # Computing achromatic response :math:`A_w` for the whitepoint,
    # same as in *CIECAM02*.
    Raw, Gaw, Baw = tsplit(RGB_aw)
    A_w = (2 * Raw + Gaw + (1 / 20) * Baw - 0.305) * N_bb

    # Recovering the correlate of *chroma* :math:`C` from the correlate
    # of *colourfulness* :math:`M` when only :math:`M` has been
    # provided.
    if has_only_nan(C) and not has_only_nan(M):
        C = M / spow(F_L, 0.25)
    elif has_only_nan(C):
        error = (
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_CIECAM16" argument!'
        )

        raise ValueError(error)

    # Computing temporary magnitude quantity :math:`t`, same form as
    # in *CIECAM02*.
    J_prime = xp.clip(J, min=EPSILON)
    t = spow(C / (xp.sqrt(J_prime / 100) * spow(1.64 - 0.29**n, 0.73)), 1 / 0.9)

    # Computing eccentricity factor :math:`e_t`, same as in *CIECAM02*.
    e_t = 1 / 4 * (xp.cos(2 + xp_radians(h)) + 3.8)

    # Computing achromatic response :math:`A` for the stimulus, same
    # inverse form as in *CIECAM02*.
    c = surround.c
    A = A_w * spow(J / 100, 1 / (c * z))

    # Computing points :math:`P_1`, :math:`P_2`, :math:`P_3`, same
    # form as in *CIECAM02*.
    N_c = surround.N_c
    with sdiv_mode():
        P_1 = sdiv((50000 / 13) * N_c * N_cb * e_t, t)
    P_2 = A / N_bb + 0.305
    P_3 = xp.full_like(P_1, 21 / 20)

    # Computing opponent colour dimensions :math:`a` and :math:`b`
    # via the sin / cos branching protecting against the numerical
    # singularity near the hue axis. Same as in *CIECAM02*.
    hr = xp_radians(h)
    sin_hr = xp.sin(hr)
    cos_hr = xp.cos(hr)
    with sdiv_mode():
        cos_hr_sin_hr = sdiv(cos_hr, sin_hr)
        sin_hr_cos_hr = sdiv(sin_hr, cos_hr)
        P_4 = sdiv(P_1, sin_hr)
        P_5 = sdiv(P_1, cos_hr)
    n_ab = P_2 * (2 + P_3) * (460 / 1403)

    abs_sin_ge_cos = xp.abs(sin_hr) >= xp.abs(cos_hr)
    abs_sin_lt_cos = xp.abs(sin_hr) < xp.abs(cos_hr)

    a = xp.zeros_like(hr)
    b = xp.zeros_like(hr)
    b = xp.where(
        abs_sin_ge_cos,
        n_ab
        / (
            P_4
            + (2 + P_3) * (220 / 1403) * cos_hr_sin_hr
            - (27 / 1403)
            + P_3 * (6300 / 1403)
        ),
        b,
    )
    a = xp.where(abs_sin_ge_cos, b * cos_hr_sin_hr, a)
    a = xp.where(
        abs_sin_lt_cos,
        n_ab
        / (
            P_5
            + (2 + P_3) * (220 / 1403)
            - ((27 / 1403) - P_3 * (6300 / 1403)) * sin_hr_cos_hr
        ),
        a,
    )
    b = xp.where(abs_sin_lt_cos, a * sin_hr_cos_hr, b)
    t_mask = xp.where(t == 0, 0, 1)
    a = a * t_mask
    b = b * t_mask

    # Applying post-adaptation non-linear response compression matrix
    # to recover the compressed *RGB* array. Same as in *CIECAM02*.
    RGB_a = (
        vecmul(
            [
                [460, 451, 288],
                [460, -891, -261],
                [460, -220, -6300],
            ],
            tstack([P_2, a, b]),
        )
        / 1403
    )

    # Applying inverse post-adaptation non-linear response compression
    # via the *CIECAM16* 3-branch piecewise function with linear
    # extensions outside the :math:`[0.26, 150]` range. The :math:`-0.1`
    # offset removes the *CIECAM16* general model offset before the
    # inversion.
    RGB_a_p = RGB_a - 0.1
    q_L, q_U = 0.26, 150
    F_L_q_L_p = spow((F_L[..., None] * q_L) / 100, 0.42)
    f_q_F_L_q_L = (400 * F_L_q_L_p) / (27.13 + F_L_q_L_p)
    F_L_q_U_p = spow((F_L[..., None] * q_U) / 100, 0.42)
    f_q_F_L_q_U = (400 * F_L_q_U_p) / (27.13 + F_L_q_U_p)
    F_L_q_U_lin = (F_L[..., None] * q_U) / 100
    d_f_q_F_L_q_U = (1.68 * 27.13 * F_L[..., None] * spow(F_L_q_U_lin, -0.58)) / (
        27.13 + spow(F_L_q_U_lin, 0.42)
    ) ** 2
    RGB_c = xp_select(
        [
            RGB_a_p > f_q_F_L_q_U,
            xp.logical_and(f_q_F_L_q_L <= RGB_a_p, RGB_a_p <= f_q_F_L_q_U),
            RGB_a_p < f_q_F_L_q_L,
        ],
        [
            q_U + (RGB_a_p - f_q_F_L_q_U) / d_f_q_F_L_q_U,
            100 / F_L[..., None] * spow((27.13 * RGB_a_p) / (400 - RGB_a_p), 1 / 0.42),
            q_L * RGB_a_p / f_q_F_L_q_L,
        ],
        xp=xp,
    )

    # Applying inverse full chromatic adaptation.
    RGB = RGB_c / D_RGB

    # Converting *CAT16* sharpened *RGB* values back to *CIE XYZ*
    # tristimulus values.
    XYZ = vecmul(MATRIX_INVERSE_16, RGB)

    return from_range_100(XYZ)
