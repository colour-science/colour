"""
CIECAM02 Colour Appearance Model
================================

Define the *CIECAM02* colour appearance model for predicting perceptual colour
attributes under varying viewing conditions.

-   :class:`colour.appearance.InductionFactors_CIECAM02`
-   :attr:`colour.VIEWING_CONDITIONS_CIECAM02`
-   :class:`colour.CAM_Specification_CIECAM02`
-   :func:`colour.XYZ_to_CIECAM02`
-   :func:`colour.CIECAM02_to_XYZ`

References
----------
-   :cite:`Fairchild2004c` : Fairchild, M. D. (2004). CIECAM02. In Color
    Appearance Models (2nd ed., pp. 289-301). Wiley. ISBN:978-0-470-01216-1
-   :cite:`InternationalElectrotechnicalCommission1999a` : International
    Electrotechnical Commission. (1999). IEC 61966-2-1:1999 - Multimedia
    systems and equipment - Colour measurement and management - Part 2-1:
    Colour management - Default RGB colour space - sRGB (p. 51).
    https://webstore.iec.ch/publication/6169
-   :cite:`Luo2013` : Luo, Ming Ronnier, & Li, C. (2013). CIECAM02 and Its
    Recent Developments. In C. Fernandez-Maloigne (Ed.), Advanced Color Image
    Processing and Analysis (pp. 19-58). Springer New York.
    doi:10.1007/978-1-4419-6190-7
-   :cite:`Moroneya` : Moroney, N., Fairchild, M. D., Hunt, R. W. G., Li, C.,
    Luo, M. R., & Newman, T. (2002). The CIECAM02 color appearance model. Color
    and Imaging Conference, 1, 23-27.
-   :cite:`Wikipedia2007a` : Fairchild, M. D. (2004). CIECAM02. In Color
    Appearance Models (2nd ed., pp. 289-301). Wiley. ISBN:978-0-470-01216-1
"""

from __future__ import annotations

import typing
from dataclasses import astuple, dataclass, field

import numpy as np

from colour.adaptation import CAT_CAT02
from colour.algebra import sdiv, sdiv_mode, spow, vecmul
from colour.appearance.hunt import (
    MATRIX_HPE_TO_XYZ,
    MATRIX_XYZ_TO_HPE,
)
from colour.colorimetry import CCS_ILLUMINANTS
from colour.constants import EPSILON

if typing.TYPE_CHECKING:
    from colour.hints import (
        Annotated,
        ArrayLike,
        Domain100,
        NDArrayFloat,
        Range100,
    )

from colour.models import xy_to_XYZ
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
    xp_select,
)
from colour.utilities.documentation import DocstringDict, is_documentation_building

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CAT_INVERSE_CAT02",
    "InductionFactors_CIECAM02",
    "VIEWING_CONDITIONS_CIECAM02",
    "HUE_DATA_FOR_HUE_QUADRATURE",
    "CAM_KWARGS_CIECAM02_sRGB",
    "CAM_Specification_CIECAM02",
    "XYZ_to_CIECAM02",
    "CIECAM02_to_XYZ",
    "base_exponential_non_linearity",
    "hue_quadrature",
]

CAT_INVERSE_CAT02: NDArrayFloat = np.linalg.inv(CAT_CAT02)
"""Inverse CAT02 chromatic adaptation transform."""


@dataclass(frozen=True)
class InductionFactors_CIECAM02(MixinDataclassIterable):
    """
    Define the *CIECAM02* colour appearance model induction factors.

    Parameters
    ----------
    F
        Maximum degree of adaptation :math:`F`.
    c
        Exponential non-linearity :math:`c`.
    N_c
        Chromatic induction factor :math:`N_c`.

    References
    ----------
    :cite:`Fairchild2004c`, :cite:`Luo2013`, :cite:`Moroneya`,
    :cite:`Wikipedia2007a`
    """

    F: float
    c: float
    N_c: float


VIEWING_CONDITIONS_CIECAM02: CanonicalMapping = CanonicalMapping(
    {
        "Average": InductionFactors_CIECAM02(1, 0.69, 1),
        "Dim": InductionFactors_CIECAM02(0.9, 0.59, 0.9),
        "Dark": InductionFactors_CIECAM02(0.8, 0.525, 0.8),
    }
)
VIEWING_CONDITIONS_CIECAM02.__doc__ = """
Define the reference *CIECAM02* colour appearance model viewing conditions.

References
----------
:cite:`Fairchild2004c`, :cite:`Luo2013`, :cite:`Moroneya`,
:cite:`Wikipedia2007a`
"""

HUE_DATA_FOR_HUE_QUADRATURE: dict = {
    "h_i": np.array([20.14, 90.00, 164.25, 237.53, 380.14]),
    "e_i": np.array([0.8, 0.7, 1.0, 1.2, 0.8]),
    "H_i": np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
}

CAM_KWARGS_CIECAM02_sRGB: dict = {
    "XYZ_w": xy_to_XYZ(CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    * 100,
    "L_A": 64 / np.pi * 0.2,
    "Y_b": 20,
    "surround": VIEWING_CONDITIONS_CIECAM02["Average"],
}
if is_documentation_building():  # pragma: no cover
    CAM_KWARGS_CIECAM02_sRGB = DocstringDict(CAM_KWARGS_CIECAM02_sRGB)
    CAM_KWARGS_CIECAM02_sRGB.__doc__ = """
Default parameter values for the *CIECAM02* colour appearance model usage in
the context of *sRGB*.

References
----------
:cite:`Fairchild2004c`, :cite:`InternationalElectrotechnicalCommission1999a`,
:cite:`Luo2013`, :cite:`Moroneya`, :cite:`Wikipedia2007a`
"""


@dataclass
class CAM_Specification_CIECAM02(MixinDataclassArithmetic):
    """
    Define the *CIECAM02* colour appearance model specification.

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
    :cite:`Fairchild2004c`, :cite:`Luo2013`, :cite:`Moroneya`,
    :cite:`Wikipedia2007a`
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_CIECAM02(
    XYZ: Domain100,
    XYZ_w: Domain100,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: InductionFactors_CIECAM02 = VIEWING_CONDITIONS_CIECAM02["Average"],
    discount_illuminant: bool = False,
    compute_H: bool = False,
) -> Annotated[CAM_Specification_CIECAM02, (100, 100, 360, 100, 100, 100, 400)]:
    """
    Compute the *CIECAM02* colour appearance model correlates from the
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
        Luminous factor of background :math:`Y_b` such as :math:`Y_b = 100
        \\times L_b / L_w` where :math:`L_w` is the luminance of the light
        source and :math:`L_b` is the luminance of the background. For
        viewing images, :math:`Y_b` can be the average :math:`Y` value for
        the pixels in the entire image, or frequently, a :math:`Y` value of
        20, approximate an :math:`L^*` of 50 is used.
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
    :class:`colour.CAM_Specification_CIECAM02`
        *CIECAM02* colour appearance model specification.

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
    :cite:`Fairchild2004c`, :cite:`Luo2013`, :cite:`Moroneya`,
    :cite:`Wikipedia2007a`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_CIECAM02["Average"]
    >>> XYZ_to_CIECAM02(
    ...     XYZ, XYZ_w, L_A, Y_b, surround,
    ...     compute_H=True,
    ... )  # doctest: +ELLIPSIS
    CAM_Specification_CIECAM02(J=np.float64(41.7310911...), \
C=np.float64(0.1047077...), h=np.float64(219.0484326...), \
s=np.float64(2.3603053...), Q=np.float64(195.3713259...), \
M=np.float64(0.1088421...), H=np.float64(278.0607358...), HC=None)
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
    # base exponential non-linearity :math:`z`.
    with sdiv_mode():
        n = sdiv(Y_b, Y_w)
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    with sdiv_mode():
        N_bb = 0.725 * spow(sdiv(1, n), 0.2)
    N_cb = N_bb
    z = 1.48 + xp.sqrt(n)

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB = vecmul(CAT_CAT02, XYZ)
    RGB_w = vecmul(CAT_CAT02, XYZ_w)

    # Computing degree of adaptation :math:`D`, bypassed entirely when
    # ``discount_illuminant`` is set.
    if discount_illuminant:
        D = xp_as_float_array(ones(L_A.shape), xp=xp, like=XYZ)
    else:
        F = xp_as_float_array(surround.F, xp=xp, like=XYZ)
        D = F * (1 - (1 / 3.6) * xp.exp((-L_A - 42) / 92))

    # Computing full chromatic adaptation, applied to the stimulus and
    # to the reference white, following *CIE (2004)* Equations 16.4a-16.6a
    # (the technical-report variant retaining the :math:`Y_W` factor).
    # *Fairchild (2013)* p.269 recommends the simpler Equations 16.4-16.6
    # without :math:`Y_W`; the two are equivalent when :math:`Y_W = 100` and
    # the project default scaling normalises to that.
    with sdiv_mode():
        RGB_c = (Y_w[..., None] * sdiv(D[..., None], RGB_w) + 1 - D[..., None]) * RGB
        RGB_wc = (Y_w[..., None] * sdiv(D[..., None], RGB_w) + 1 - D[..., None]) * RGB_w

    # Converting to *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
    # colourspace, applied to both stimulus and white.
    MATRIX_XYZ_HPE_x_CAT_INVERSE = xp.matmul(
        xp_as_float_array(MATRIX_XYZ_TO_HPE, xp=xp, like=XYZ),
        xp_as_float_array(CAT_INVERSE_CAT02, xp=xp, like=XYZ),
    )
    RGB_p = vecmul(MATRIX_XYZ_HPE_x_CAT_INVERSE, RGB_c)
    RGB_pw = vecmul(MATRIX_XYZ_HPE_x_CAT_INVERSE, RGB_wc)

    # Applying forward post-adaptation non-linear response compression,
    # sign-preserving for negative values per *Luo (2013)*.
    F_L_RGB_p = spow(F_L[..., None] * xp.abs(RGB_p) / 100, 0.42)
    RGB_a = (400 * xp.sign(RGB_p) * F_L_RGB_p) / (27.13 + F_L_RGB_p) + 0.1
    F_L_RGB_pw = spow(F_L[..., None] * xp.abs(RGB_pw) / 100, 0.42)
    RGB_aw = (400 * xp.sign(RGB_pw) * F_L_RGB_pw) / (27.13 + F_L_RGB_pw) + 0.1

    # Converting to preliminary cartesian coordinates :math:`a`,
    # :math:`b`.
    Ra, Ga, Ba = tsplit(RGB_a)
    a = Ra - 12 * Ga / 11 + Ba / 11
    b = (Ra + Ga - 2 * Ba) / 9

    # Computing the *hue* angle :math:`h` in degrees in
    # :math:`[0, 360)`.
    h = xp_degrees(xp.atan2(b, a)) % 360

    # Computing eccentricity factor :math:`e_t`.
    e_t = 1 / 4 * (xp.cos(2 + xp_radians(h)) + 3.8)

    # Computing achromatic responses :math:`A` for the stimulus and
    # :math:`A_w` for the whitepoint.
    A = (2 * Ra + Ga + (1 / 20) * Ba - 0.305) * N_bb
    Raw, Gaw, Baw = tsplit(RGB_aw)
    A_w = (2 * Raw + Gaw + (1 / 20) * Baw - 0.305) * N_bb

    # Computing the correlate of *Lightness* :math:`J`.
    c = surround.c
    with sdiv_mode():
        J = 100 * spow(sdiv(A, A_w), c * z)

    # Computing the correlate of *brightness* :math:`Q`.
    Q = (4 / c) * xp.sqrt(J / 100) * (A_w + 4) * spow(F_L, 0.25)

    # Computing the temporary magnitude quantity :math:`t` and the
    # correlate of *chroma* :math:`C`.
    N_c = surround.N_c
    with sdiv_mode():
        t = ((50000 / 13) * N_c * N_cb) * sdiv(
            e_t * spow(a**2 + b**2, 0.5), Ra + Ga + 21 * Ba / 20
        )
    C = spow(t, 0.9) * spow(J / 100, 0.5) * spow(1.64 - 0.29**n, 0.73)

    # Computing the correlate of *colourfulness* :math:`M`.
    M = C * spow(F_L, 0.25)

    # Computing the correlate of *saturation* :math:`s`.
    with sdiv_mode():
        s = 100 * spow(sdiv(M, Q), 0.5)

    # Computing hue :math:`h` quadrature :math:`H` only when requested
    # via ``compute_H``; the :math:`H` quadrature is rarely consumed
    # and the bin-search delegates to :func:`hue_quadrature` which is
    # kept as a public reference shared with the *ZCAM* and *sCAM*
    # paths.
    # TODO: Compute hue composition.
    H = hue_quadrature(h) if compute_H else xp.full_like(h, float("nan"))

    return CAM_Specification_CIECAM02(
        J=as_float(from_range_100(J)),
        C=as_float(from_range_100(C)),
        h=as_float(from_range_degrees(h)),
        s=as_float(from_range_100(s)),
        Q=as_float(from_range_100(Q)),
        M=as_float(from_range_100(M)),
        H=as_float(from_range_degrees(H, 400)),
        HC=None,
    )


def CIECAM02_to_XYZ(
    specification: Annotated[
        CAM_Specification_CIECAM02, (100, 100, 360, 100, 100, 100, 400)
    ],
    XYZ_w: Domain100,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: InductionFactors_CIECAM02 = VIEWING_CONDITIONS_CIECAM02["Average"],
    discount_illuminant: bool = False,
) -> Range100:
    """
    Convert the *CIECAM02* colour appearance model specification to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    specification
        *CIECAM02* colour appearance model specification. Correlate of
        *Lightness* :math:`J`, correlate of *chroma* :math:`C` or correlate of
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
        If neither :math:`C` nor :math:`M` correlates have been defined in the
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
    :cite:`Fairchild2004c`, :cite:`Luo2013`, :cite:`Moroneya`,
    :cite:`Wikipedia2007a`

    Examples
    --------
    >>> specification = CAM_Specification_CIECAM02(
    ...     J=41.731091132513917, C=0.104707757171031, h=219.048432658311780
    ... )
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b)  # doctest: +ELLIPSIS
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
    # base exponential non-linearity :math:`z`.
    with sdiv_mode():
        n = sdiv(Y_b, Y_w)
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    with sdiv_mode():
        N_bb = 0.725 * spow(sdiv(1, n), 0.2)
    N_cb = N_bb
    z = 1.48 + xp.sqrt(n)

    if has_only_nan(C) and not has_only_nan(M):
        C = M / spow(F_L, 0.25)
    elif has_only_nan(C):
        error = (
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_CIECAM02" argument!'
        )

        raise ValueError(error)

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values for the reference white.
    RGB_w = vecmul(CAT_CAT02, XYZ_w)

    # Computing degree of adaptation :math:`D`, bypassed entirely when
    # ``discount_illuminant`` is set.
    if discount_illuminant:
        D = xp_as_float_array(ones(L_A.shape), xp=xp, like=J)
    else:
        F = xp_as_float_array(surround.F, xp=xp, like=J)
        D = F * (1 - (1 / 3.6) * xp.exp((-L_A - 42) / 92))

    # Computing full chromatic adaptation for the reference white.
    with sdiv_mode():
        RGB_wc = (Y_w[..., None] * sdiv(D[..., None], RGB_w) + 1 - D[..., None]) * RGB_w

    # Converting to *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
    # colourspace.
    MATRIX_XYZ_HPE_x_CAT_INVERSE = xp.matmul(
        xp_as_float_array(MATRIX_XYZ_TO_HPE, xp=xp, like=J),
        xp_as_float_array(CAT_INVERSE_CAT02, xp=xp, like=J),
    )
    RGB_pw = vecmul(MATRIX_XYZ_HPE_x_CAT_INVERSE, RGB_wc)

    # Applying forward post-adaptation non-linear response compression
    # to the whitepoint.
    F_L_RGB_pw = spow(F_L[..., None] * xp.abs(RGB_pw) / 100, 0.42)
    RGB_aw = (400 * xp.sign(RGB_pw) * F_L_RGB_pw) / (27.13 + F_L_RGB_pw) + 0.1

    # Computing achromatic response :math:`A_w` for the whitepoint.
    Raw, Gaw, Baw = tsplit(RGB_aw)
    A_w = (2 * Raw + Gaw + (1 / 20) * Baw - 0.305) * N_bb

    # Computing the temporary magnitude quantity :math:`t`.
    J_prime = xp.clip(J, min=EPSILON)
    t = spow(C / (xp.sqrt(J_prime / 100) * spow(1.64 - 0.29**n, 0.73)), 1 / 0.9)

    # Computing eccentricity factor :math:`e_t`.
    e_t = 1 / 4 * (xp.cos(2 + xp_radians(h)) + 3.8)

    # Computing achromatic response :math:`A` for the stimulus.
    c = surround.c
    A = A_w * spow(J / 100, 1 / (c * z))

    # Computing points :math:`P_1`, :math:`P_2`, :math:`P_3`.
    N_c = surround.N_c
    with sdiv_mode():
        P_1 = sdiv((50000 / 13) * N_c * N_cb * e_t, t)
    P_2 = A / N_bb + 0.305
    P_3 = xp.full_like(P_1, 21 / 20)

    # Computing opponent colour dimensions :math:`a` and :math:`b`
    # from the points :math:`P_n` and hue :math:`h` via the sin / cos
    # branching that protects against the numerical singularity near
    # the hue axis.
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
    # to recover the compressed *RGB* array.
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

    # Applying inverse post-adaptation non-linear response compression.
    RGB_p = (
        xp.sign(RGB_a - 0.1)
        * 100
        / F_L[..., None]
        * spow(
            (27.13 * xp.abs(RGB_a - 0.1)) / (400 - xp.abs(RGB_a - 0.1)),
            1 / 0.42,
        )
    )

    # Converting from *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
    # colourspace back to adapted *RGB*.
    CAT_x_MATRIX_HPE = xp.matmul(
        xp_as_float_array(CAT_CAT02, xp=xp, like=J),
        xp_as_float_array(MATRIX_HPE_TO_XYZ, xp=xp, like=J),
    )
    RGB_c = vecmul(CAT_x_MATRIX_HPE, RGB_p)

    # Applying inverse full chromatic adaptation.
    with sdiv_mode():
        RGB = RGB_c / (Y_w[..., None] * sdiv(D[..., None], RGB_w) + 1 - D[..., None])

    # Converting *CMCCAT2000* transform sharpened *RGB* values to
    # *CIE XYZ* tristimulus values.
    XYZ = vecmul(CAT_INVERSE_CAT02, RGB)

    return from_range_100(XYZ)


def base_exponential_non_linearity(
    n: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the base exponential non-linearity :math:`n`.

    Parameters
    ----------
    n
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    :class:`numpy.ndarray`
        Base exponential non-linearity :math:`z`.

    Examples
    --------
    >>> base_exponential_non_linearity(0.2)  # doctest: +ELLIPSIS
    np.float64(1.9272135...)
    """

    n = as_float_array(n)

    xp = array_namespace(n)

    return 1.48 + xp.sqrt(n)


def hue_quadrature(h: ArrayLike) -> NDArrayFloat:
    """
    Compute hue quadrature from the specified hue :math:`h` angle in degrees.

    Parameters
    ----------
    h
        Hue :math:`h` angle in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Hue quadrature.

    Examples
    --------
    >>> hue_quadrature(219.0484326582719)  # doctest: +ELLIPSIS
    np.float64(278.0607358...)
    """

    h = as_float_array(h)

    xp = array_namespace(h)

    h = as_float_array(xp.where(xp.isnan(h), 0, h))

    # Hue quadrature bin boundaries from the *CIE 159:2004* table; the
    # intervals are unrolled (rather than gathered via ``searchsorted``)
    # so the computation stays portable across array backends.
    h_i = HUE_DATA_FOR_HUE_QUADRATURE["h_i"]
    e_i = HUE_DATA_FOR_HUE_QUADRATURE["e_i"]
    H_i = HUE_DATA_FOR_HUE_QUADRATURE["H_i"]

    def _H(
        h_k: float, e_k: float, H_k: float, h_k1: float, e_k1: float
    ) -> NDArrayFloat:
        """Compute hue quadrature for a single bin."""

        t1 = (h - h_k) / e_k
        t2 = (h_k1 - h) / e_k1
        return H_k + 100 * t1 / (t1 + t2)

    H_0 = _H(h_i[0], e_i[0], H_i[0], h_i[1], e_i[1])
    H_1 = _H(h_i[1], e_i[1], H_i[1], h_i[2], e_i[2])
    H_2 = _H(h_i[2], e_i[2], H_i[2], h_i[3], e_i[3])

    # Last interval and wrap-around use special formulas that account for
    # the circular hue boundary at 360 degrees.
    t1_3 = (h - h_i[3]) / e_i[3]
    H_3 = H_i[3] + (85.9 * t1_3) / (t1_3 + (360 - h) / 0.856)
    H_wrap = 385.9 + (14.1 * h / 0.856) / (h / 0.856 + (h_i[0] - h) / e_i[0])

    H = xp_select(
        [
            (h >= h_i[0]) & (h < h_i[1]),
            (h >= h_i[1]) & (h < h_i[2]),
            (h >= h_i[2]) & (h < h_i[3]),
            (h >= h_i[3]),
        ],
        [H_0, H_1, H_2, H_3],
        default=H_wrap,
        xp=xp,
    )

    return as_float(H)
