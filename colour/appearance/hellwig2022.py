"""
Hellwig and Fairchild (2022) Colour Appearance Model
====================================================

Define the *Hellwig and Fairchild (2022)* colour appearance model for
predicting perceptual colour attributes under varying viewing conditions.

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
-   :cite:`Hellwig2022a` : Hellwig, L., Stolitzka, D., & Fairchild, M. D.
    (2022). Extending CIECAM02 and CAM16 for the Helmholtz-Kohlrausch effect.
    Color Research & Application, col.22793. doi:10.1002/col.22793
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
    as_float_array,
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
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "InductionFactors_Hellwig2022",
    "VIEWING_CONDITIONS_HELLWIG2022",
    "CAM_Specification_Hellwig2022",
    "XYZ_to_Hellwig2022",
    "Hellwig2022_to_XYZ",
    "eccentricity_factor_Hellwig2022",
    "hue_angle_dependency_Hellwig2022",
]


@dataclass(frozen=True)
class InductionFactors_Hellwig2022(MixinDataclassIterable):
    """
    Define the *Hellwig and Fairchild (2022)* colour appearance model
    induction factors.

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

    F: float
    c: float
    N_c: float


VIEWING_CONDITIONS_HELLWIG2022: CanonicalMapping = CanonicalMapping(
    VIEWING_CONDITIONS_CIECAM02
)
VIEWING_CONDITIONS_HELLWIG2022.__doc__ = """
Define the reference *Hellwig and Fairchild (2022)* colour appearance model
viewing conditions.

References
----------
:cite:`Hellwig2022`
"""


@dataclass
class CAM_Specification_Hellwig2022(MixinDataclassArithmetic):
    """
    Define the *Hellwig and Fairchild (2022)* colour appearance model
    specification.

    Represent colour appearance attributes calculated by the
    *Hellwig and Fairchild (2022)* colour appearance model. The
    specification includes correlates for lightness, chroma, hue,
    saturation, brightness, colourfulness, and hue quadrature. This
    implementation supports the *Helmholtz-Kohlrausch* effect extension
    from :cite:`Hellwig2022a`, providing adjusted lightness and brightness
    correlates that account for the increased brightness perception of
    highly saturated colours.

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
    J_HK
        Correlate of *lightness* :math:`J_{HK}` accounting for
        *Helmholtz-Kohlrausch* effect.
    Q_HK
        Correlate of *brightness* :math:`Q_{HK}` accounting for
        *Helmholtz-Kohlrausch* effect.

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`, :cite:`Hellwig2022a`
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)
    J_HK: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q_HK: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_Hellwig2022(
    XYZ: Domain100,
    XYZ_w: Domain100,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: (
        InductionFactors_CIECAM02 | InductionFactors_Hellwig2022
    ) = VIEWING_CONDITIONS_HELLWIG2022["Average"],
    discount_illuminant: bool = False,
    compute_H: bool = False,
) -> Annotated[
    CAM_Specification_Hellwig2022, (100, 100, 360, 100, 100, 100, 400, 100, 100)
]:
    """
    Compute the *Hellwig and Fairchild (2022)* colour appearance model
    correlates from the specified *CIE XYZ* tristimulus values.

    This implementation supports the *Helmholtz-Kohlrausch* effect extension
    from :cite:`Hellwig2022a`.

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
        :math:`Y_b = 100 \\times L_b / L_w` where :math:`L_w` is the luminance
        of the light source and :math:`L_b` is the luminance of the background.
        For viewing images, :math:`Y_b` can be the average :math:`Y` value for
        the pixels in the entire image, or frequently, a :math:`Y` value of 20,
        approximating an :math:`L^*` of 50 is used.
    surround
        Surround viewing conditions induction factors.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.
    compute_H
        When *True*, compute the *Hue Quadrature* :math:`H` correlate
        via :func:`colour.appearance.ciecam02.hue_quadrature`. Defaults to
        *False* because :math:`H` is rarely consumed downstream and
        skipping the bin search is a measurable cost saving.

    Returns
    -------
    :class:`colour.CAM_Specification_Hellwig2022`
        *Hellwig and Fairchild (2022)* colour appearance model specification.

    Notes
    -----
    +------------------------+-----------------------+---------------+
    | **Domain**             | **Scale - Reference** | **Scale - 1** |
    +========================+=======================+===============+
    | ``XYZ``                | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``XYZ_w``              | 100                   | 1             |
    +------------------------+-----------------------+---------------+

    +------------------------+-----------------------+---------------+
    | **Range**              | **Scale - Reference** | **Scale - 1** |
    +========================+=======================+===============+
    | ``specification.J``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.C``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.h``    | 360                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.s``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.Q``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.M``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.H``    | 400                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.J_HK`` | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.Q_HK`` | 100                   | 1             |
    +------------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`, :cite:`Hellwig2022a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
    >>> XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)
    ... # doctest: +ELLIPSIS
    CAM_Specification_Hellwig2022(J=np.float64(41.7312079...), \
C=np.float64(0.0257636...), h=np.float64(217.0679597...), \
s=np.float64(0.0608550...), Q=np.float64(55.8523226...), \
M=np.float64(0.0339889...), H=np.float64(275.5949861...), HC=None, \
J_HK=np.float64(41.8802782...), Q_HK=np.float64(56.0518358...))
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
    # factor :math:`n`, luminance level adaptation factor :math:`F_L`
    # (same as *Hunt*) and base exponential non-linearity :math:`z`
    # (same as *CIECAM02*).
    with sdiv_mode():
        n = sdiv(Y_b, Y_w)
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    z = 1.48 + xp.sqrt(n)

    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values
    # using the *CAM16* matrix, for the stimulus and the reference white.
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

    # Computing full chromatic adaptation, applied to the stimulus and
    # the reference white via a shared adaptation factor.
    D_RGB = D[..., None] * Y_w[..., None] / RGB_w + 1 - D[..., None]
    RGB_c = D_RGB * RGB
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression,
    # same sign-preserving form as in *CIECAM02* per *Luo (2013)*.
    F_L_RGB_c = spow(F_L[..., None] * xp.abs(RGB_c) / 100, 0.42)
    RGB_a = (400 * xp.sign(RGB_c) * F_L_RGB_c) / (27.13 + F_L_RGB_c) + 0.1
    F_L_RGB_wc = spow(F_L[..., None] * xp.abs(RGB_wc) / 100, 0.42)
    RGB_aw = (400 * xp.sign(RGB_wc) * F_L_RGB_wc) / (27.13 + F_L_RGB_wc) + 0.1

    # Computing the opponent colour dimensions :math:`a` and :math:`b`,
    # same formulation as in *CIECAM02*.
    Ra, Ga, Ba = tsplit(RGB_a)
    a = Ra - 12 * Ga / 11 + Ba / 11
    b = (Ra + Ga - 2 * Ba) / 9

    # Computing the *hue* angle :math:`h` in degrees in
    # :math:`[0, 360)`, same as in *CIECAM02*.
    h = xp_degrees(xp.atan2(b, a)) % 360

    e_t = eccentricity_factor_Hellwig2022(h)

    # Computing achromatic responses :math:`A` for the stimulus and
    # :math:`A_w` for the whitepoint, using the *Hellwig 2022* weights
    # which simplify the *CIECAM02* form to :math:`2R + G + 0.05 B - 0.305`.
    A = 2 * Ra + Ga + 0.05 * Ba - 0.305
    Raw, Gaw, Baw = tsplit(RGB_aw)
    A_w = 2 * Raw + Gaw + 0.05 * Baw - 0.305

    # Computing the correlate of *Lightness* :math:`J`, same form as
    # in *CIECAM02*.
    c = surround.c
    with sdiv_mode():
        J = 100 * spow(sdiv(A, A_w), c * z)

    # Computing the correlate of *brightness* :math:`Q`. *Hellwig 2022*
    # drops the :math:`F_L^{0.25}` term that appears in *CIECAM02*'s
    # formulation.
    Q = (2 / c) * (J / 100) * A_w

    # Computing the correlate of *colourfulness* :math:`M`, *Hellwig
    # 2022* form built directly from the opponent dimensions rather
    # than the *CIECAM02* temporary magnitude quantity :math:`t`.
    N_c = surround.N_c
    M = 43.0 * N_c * e_t * xp.hypot(a, b)

    # Computing the correlate of *chroma* :math:`C` and the correlate
    # of *saturation* :math:`s`, *Hellwig 2022* simplifications of the
    # *CIECAM02* expressions.
    with sdiv_mode():
        C = 35 * sdiv(M, A_w)
        s = 100 * sdiv(M, Q)

    # *Helmholtz-Kohlrausch* effect extension: hue angle dependency
    # specific to *Hellwig 2022*.
    J_HK = J + hue_angle_dependency_Hellwig2022(h) * spow(C, 0.587)
    Q_HK = (2 / c) * (J_HK / 100) * A_w

    # Computing hue :math:`h` quadrature :math:`H` only when requested
    # via ``compute_H``; the bin search is shared with *CIECAM02* and
    # delegates to :func:`hue_quadrature`.
    # TODO: Compute hue composition.
    H = hue_quadrature(h) if compute_H else xp.full_like(h, float("nan"))

    return CAM_Specification_Hellwig2022(
        J=as_float(from_range_100(J)),
        C=as_float(from_range_100(C)),
        h=as_float(from_range_degrees(h)),
        s=as_float(from_range_100(s)),
        Q=as_float(from_range_100(Q)),
        M=as_float(from_range_100(M)),
        H=as_float(from_range_degrees(H, 400)),
        HC=None,
        J_HK=as_float(from_range_100(J_HK)),
        Q_HK=as_float(from_range_100(Q_HK)),
    )


def Hellwig2022_to_XYZ(
    specification: Annotated[
        CAM_Specification_Hellwig2022, (100, 100, 360, 100, 100, 100, 400, 100, 100)
    ],
    XYZ_w: Domain100,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: (
        InductionFactors_CIECAM02 | InductionFactors_Hellwig2022
    ) = VIEWING_CONDITIONS_HELLWIG2022["Average"],
    discount_illuminant: bool = False,
) -> Range100:
    """
    Convert the *Hellwig and Fairchild (2022)* colour appearance model
    specification to *CIE XYZ* tristimulus values.

    This implementation supports the *Helmholtz-Kohlrausch* effect extension
    from :cite:`Hellwig2022a`.

    Parameters
    ----------
    specification
        *Hellwig and Fairchild (2022)* colour appearance model specification.
        Correlate of *lightness* :math:`J`, correlate of *chroma* :math:`C`
        or correlate of *colourfulness* :math:`M` and *hue* angle :math:`h`
        in degrees must be specified, e.g., :math:`JCh` or :math:`JMh`.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often
        taken to be 20% of the luminance of a white object in the scene).
    Y_b
        Luminous factor of background :math:`Y_b` such as
        :math:`Y_b = 100 \\times L_b / L_w` where :math:`L_w` is the
        luminance of the light source and :math:`L_b` is the luminance of the
        background. For viewing images, :math:`Y_b` can be the average
        :math:`Y` value for the pixels in the entire image, or frequently, a
        :math:`Y` value of 20, approximating an :math:`L^*` of 50 is used.
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
        If neither :math:`C` or :math:`M` correlates have been defined in
        the ``specification`` argument.

    Notes
    -----
    +------------------------+-----------------------+---------------+
    | **Domain**             | **Scale - Reference** | **Scale - 1** |
    +========================+=======================+===============+
    | ``specification.J``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.C``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.h``    | 360                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.s``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.Q``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.M``    | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.H``    | 400                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.J_HK`` | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``specification.Q_HK`` | 100                   | 1             |
    +------------------------+-----------------------+---------------+
    | ``XYZ_w``              | 100                   | 1             |
    +------------------------+-----------------------+---------------+

    +------------------------+-----------------------+---------------+
    | **Range**              | **Scale - Reference** | **Scale - 1** |
    +========================+=======================+===============+
    | ``XYZ``                | 100                   | 1             |
    +------------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`, :cite:`Hellwig2022a`

    Examples
    --------
    >>> import numpy as np
    >>> specification = CAM_Specification_Hellwig2022(
    ...     J=41.731207905126638, C=0.025763615829912909, h=217.06795976739301
    ... )
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b)
    ... # doctest: +ELLIPSIS
    array([19.01..., 20...  , 21.78...])
    >>> specification = CAM_Specification_Hellwig2022(
    ...     J_HK=41.880278283880095,
    ...     C=0.025763615829912909,
    ...     h=217.06795976739301,
    ... )
    >>> Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b)
    ... # doctest: +ELLIPSIS
    array([19.01..., 20...  , 21.78...])
    """

    J, C, h, _s, _Q, M, _H, _HC, J_HK, _Q_HK = astuple(specification)

    C = to_domain_100(C)
    h = to_domain_degrees(h)
    M = to_domain_100(M)

    # *Helmholtz-Kohlrausch* effect extension, inverted: recover the
    # plain *Lightness* :math:`J` from :math:`J_{HK}` when only the
    # latter has been provided, using the *Hellwig 2022*-specific
    # 2-harmonic Fourier hue angle dependency.
    if has_only_nan(J) and not has_only_nan(J_HK):
        J_HK = to_domain_100(J_HK)

        J = J_HK - hue_angle_dependency_Hellwig2022(h) * spow(C, 0.587)
    elif has_only_nan(J):
        error = (
            'Either "J" or "J_HK" correlate must be defined in '
            'the "CAM_Specification_Hellwig2022" argument!'
        )

        raise ValueError(error)
    else:
        J = to_domain_100(J)

    L_A = as_float_array(L_A)
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
    # factor :math:`n`, luminance level adaptation factor :math:`F_L`
    # (same as *Hunt*) and base exponential non-linearity :math:`z`
    # (same as *CIECAM02*).
    with sdiv_mode():
        n = sdiv(Y_b, Y_w)
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    z = 1.48 + xp.sqrt(n)

    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values
    # using the *CAM16* matrix for the reference white.
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
    D_RGB = D[..., None] * Y_w[..., None] / RGB_w + 1 - D[..., None]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression
    # to the whitepoint, same sign-preserving form as in *CIECAM02*
    # per *Luo (2013)*.
    F_L_RGB_wc = spow(F_L[..., None] * xp.abs(RGB_wc) / 100, 0.42)
    RGB_aw = (400 * xp.sign(RGB_wc) * F_L_RGB_wc) / (27.13 + F_L_RGB_wc) + 0.1

    # Computing achromatic response :math:`A_w` for the whitepoint,
    # *Hellwig 2022* weights.
    Raw, Gaw, Baw = tsplit(RGB_aw)
    A_w = 2 * Raw + Gaw + 0.05 * Baw - 0.305

    # Recovering the correlate of *colourfulness* :math:`M` from the
    # correlate of *chroma* :math:`C` via the *Hellwig 2022* inverse
    # relation, when only :math:`C` has been provided.
    if has_only_nan(M) and not has_only_nan(C):
        M = (C * A_w) / 35
    elif has_only_nan(M):
        error = (
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_Hellwig2022" argument!'
        )

        raise ValueError(error)

    e_t = eccentricity_factor_Hellwig2022(h)

    # Computing achromatic response :math:`A` for the stimulus,
    # same inverse form as in *CIECAM02*.
    c = surround.c
    A = A_w * spow(J / 100, 1 / (c * z))

    # Computing points :math:`P'_1` and :math:`P'_2`, the *Hellwig
    # 2022* simplification of *CIECAM02*'s :math:`P_n` triple.
    N_c = surround.N_c
    P_p_1 = 43 * N_c * e_t
    P_p_2 = A

    # Computing opponent colour dimensions :math:`a` and :math:`b`
    # from :math:`P'_1`, :math:`h` and :math:`M` via the *Hellwig 2022*
    # closed-form rather than *CIECAM02*'s sin/cos-branched inverse.
    hr = xp_radians(h)
    with sdiv_mode():
        gamma = sdiv(M, P_p_1)
    a = gamma * xp.cos(hr)
    b = gamma * xp.sin(hr)

    # Applying post-adaptation non-linear response compression matrix,
    # same form as in *CIECAM02*.
    RGB_a = (
        vecmul(
            [
                [460, 451, 288],
                [460, -891, -261],
                [460, -220, -6300],
            ],
            tstack([P_p_2, a, b]),
        )
        / 1403
    )

    # Applying inverse post-adaptation non-linear response compression,
    # same form as in *CIECAM02*. The :math:`+0.1` offset compensates
    # for the *Hellwig 2022* formulation of the matrix step.
    RGB_a_p = RGB_a + 0.1
    RGB_c = (
        xp.sign(RGB_a_p - 0.1)
        * 100
        / F_L[..., None]
        * spow(
            (27.13 * xp.abs(RGB_a_p - 0.1)) / (400 - xp.abs(RGB_a_p - 0.1)),
            1 / 0.42,
        )
    )

    # Applying inverse full chromatic adaptation.
    RGB = RGB_c / D_RGB

    # Converting sharpened *RGB* values back to *CIE XYZ* tristimulus
    # values using the inverse *CAM16* matrix.
    XYZ = vecmul(MATRIX_INVERSE_16, RGB)

    return from_range_100(XYZ)


def eccentricity_factor_Hellwig2022(h: ArrayLike) -> NDArrayFloat:
    """
    Compute the eccentricity factor :math:`e_t` from the specified hue
    :math:`h` angle in degrees for the *Hellwig and Fairchild (2022)* colour
    appearance model.

    Parameters
    ----------
    h
        Hue :math:`h` angle in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Eccentricity factor :math:`e_t`.

    References
    ----------
    :cite:`Hellwig2022`

    Examples
    --------
    >>> eccentricity_factor_Hellwig2022(217.067959767393)  # doctest: +ELLIPSIS
    np.float64(0.9945215...)
    """

    h = as_float_array(h)

    xp = array_namespace(h)

    hr = xp_radians(h)

    return as_float(
        -0.0582 * xp.cos(hr)
        - 0.0258 * xp.cos(2 * hr)
        - 0.1347 * xp.cos(3 * hr)
        + 0.0289 * xp.cos(4 * hr)
        - 0.1475 * xp.sin(hr)
        - 0.0308 * xp.sin(2 * hr)
        + 0.0385 * xp.sin(3 * hr)
        + 0.0096 * xp.sin(4 * hr)
        + 1
    )


def hue_angle_dependency_Hellwig2022(h: ArrayLike) -> NDArrayFloat:
    """
    Compute the hue angle dependency of the *Helmholtz-Kohlrausch* effect for
    the *Hellwig and Fairchild (2022)* colour appearance model.

    Parameters
    ----------
    h
        Hue :math:`h` angle in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Hue angle dependency of the *Helmholtz-Kohlrausch* effect.

    References
    ----------
    :cite:`Hellwig2022a`

    Examples
    --------
    >>> hue_angle_dependency_Hellwig2022(217.06795976739301)  # doctest: +ELLIPSIS
    np.float64(1.2768219...)
    """

    h = as_float_array(h)

    xp = array_namespace(h)

    hr = xp_radians(h)

    return as_float(
        -0.160 * xp.cos(hr)
        + 0.132 * xp.cos(2 * hr)
        - 0.405 * xp.sin(hr)
        + 0.080 * xp.sin(2 * hr)
        + 0.792
    )
