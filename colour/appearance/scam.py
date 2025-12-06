"""
sCAM Colour Appearance Model
============================

Define the *sCAM* colour appearance model for predicting perceptual colour
attributes under varying viewing conditions.

-   :class:`colour.appearance.InductionFactors_sCAM`
-   :attr:`colour.VIEWING_CONDITIONS_sCAM`
-   :class:`colour.CAM_Specification_sCAM`
-   :func:`colour.XYZ_to_sCAM`
-   :func:`colour.sCAM_to_XYZ`

The *sCAM* (Simple Colour Appearance Model) is based on the *sUCS* (Simple
Uniform Colour Space).

References
----------
-   :cite:`Li2024` : Li, M., & Luo, M. R. (2024). Simple color appearance model
    (sCAM) based on simple uniform color space (sUCS). Optics Express, 32(3),
    3100. doi:10.1364/OE.510196
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, field

import numpy as np

from colour.adaptation import chromatic_adaptation_Li2025
from colour.algebra import sdiv, sdiv_mode, spow
from colour.hints import (  # noqa: TC001
    Annotated,
    ArrayLike,
    Domain100,
    NDArrayFloat,
    Range100,
)
from colour.models.sucs import (
    XYZ_to_sUCS,
    sUCS_Iab_to_sUCS_ICh,
    sUCS_ICh_to_sUCS_Iab,
    sUCS_to_XYZ,
)
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArithmetic,
    MixinDataclassIterable,
    as_float,
    as_float_array,
    domain_range_scale,
    from_range_100,
    from_range_degrees,
    has_only_nan,
    to_domain_100,
    to_domain_degrees,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2024 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TVS_D65_sCAM",
    "HUE_DATA_FOR_HUE_QUADRATURE_sCAM",
    "InductionFactors_sCAM",
    "VIEWING_CONDITIONS_sCAM",
    "CAM_Specification_sCAM",
    "XYZ_to_sCAM",
    "sCAM_to_XYZ",
    "hue_quadrature",
]

TVS_D65_sCAM = np.array([0.95047, 1.00000, 1.08883])
"""*CIE XYZ* tristimulus values of *CIE Standard Illuminant D65* for *sCAM*."""

HUE_DATA_FOR_HUE_QUADRATURE_sCAM: dict = {
    "h_i": np.array([15.6, 80.3, 157.8, 219.7, 376.6]),
    "e_i": np.array([0.7, 0.6, 1.2, 0.9, 0.7]),
    "H_i": np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
}
"""Hue quadrature data for *sCAM* colour appearance model."""


@dataclass(frozen=True)
class InductionFactors_sCAM(MixinDataclassIterable):
    """
    Define the *sCAM* colour appearance model induction factors.

    Parameters
    ----------
    F
        Maximum degree of adaptation :math:`F`.
    c
        Exponential non-linearity :math:`c`.
    Fm
        Factor for colourfulness :math:`F_m`.

    References
    ----------
    :cite:`Li2024`
    """

    F: float
    c: float
    Fm: float


VIEWING_CONDITIONS_sCAM: CanonicalMapping = CanonicalMapping(
    {
        "Average": InductionFactors_sCAM(F=1.0, c=0.52, Fm=1.0),
        "Dim": InductionFactors_sCAM(F=0.9, c=0.50, Fm=0.95),
        "Dark": InductionFactors_sCAM(F=0.8, c=0.39, Fm=0.85),
    }
)
VIEWING_CONDITIONS_sCAM.__doc__ = """
Define the reference *sCAM* colour appearance model
viewing conditions.

Provide standardized surround conditions (*Average*, *Dim*, *Dark*) with
their corresponding induction factors that characterize chromatic
adaptation and perceptual non-linearities under different viewing
environments.
"""


@dataclass
class CAM_Specification_sCAM(MixinDataclassArithmetic):
    """
    Define the specification for the *sCAM* colour appearance model.

    Parameters
    ----------
    J
        Correlate of *lightness* :math:`J`.
    C
        Correlate of *chroma* :math:`C`.
    h
        *Hue* angle :math:`h` in degrees.
    Q
        Correlate of *brightness* :math:`Q`.
    M
        Correlate of *colourfulness* :math:`M`.
    H
        *Hue* :math:`h` composition :math:`H`.
    HC
        *Hue* :math:`h` composition :math:`H^C` (currently not
        implemented).
    V
        Correlate of *vividness* :math:`V`.
    K
        Correlate of *blackness* :math:`K`.
    W
        Correlate of *whiteness* :math:`W`.
    D
        Correlate of *depth* :math:`D`.

    References
    ----------
    :cite:`Li2024`
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)
    V: float | NDArrayFloat | None = field(default_factory=lambda: None)
    K: float | NDArrayFloat | None = field(default_factory=lambda: None)
    W: float | NDArrayFloat | None = field(default_factory=lambda: None)
    D: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_sCAM(
    XYZ: Domain100,
    XYZ_w: Domain100,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: InductionFactors_sCAM = VIEWING_CONDITIONS_sCAM["Average"],
    discount_illuminant: bool = False,
) -> Annotated[
    CAM_Specification_sCAM, (100, 100, 360, 100, 100, 400, 100, 100, 100, 100)
]:
    """
    Compute the *sCAM* colour appearance model correlates from the specified
    *CIE XYZ* tristimulus values.

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
        a :math:`Y` value of 20, approximating an :math:`L^*` of 50 is
        used.
    surround
        Surround viewing conditions induction factors.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    :class:`colour.CAM_Specification_sCAM`
        *sCAM* colour appearance model specification.

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
    | ``specification.Q`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.M`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.H`` | 400                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.HC``| None                  | None          |
    +---------------------+-----------------------+---------------+
    | ``specification.V`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.K`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.W`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``specification.D`` | 100                   | 1             |
    +---------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_sCAM["Average"]
    >>> XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround)  # doctest: +ELLIPSIS
    CAM_Specification_sCAM(J=49.9795668..., C=0.0140531..., h=328.2724924..., \
Q=195.23024234..., M=0.0050244..., H=363.6013437..., HC=None, V=49.9795727..., \
K=50.0204272..., W=34.9734327..., D=65.0265672...)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    Y_w = XYZ_w[..., 1] if XYZ_w.ndim > 1 else XYZ_w[1]

    with sdiv_mode():
        z = 1.48 + spow(sdiv(Y_b, Y_w), 0.5)

    F_L = 0.1710 * spow(L_A, 1 / 3) / (1 - 0.4934 * np.exp(-0.9934 * L_A))

    with sdiv_mode():
        L_A_D65 = sdiv(L_A * 100, Y_b)

    XYZ_w_D65 = TVS_D65_sCAM * L_A_D65[..., None]

    with domain_range_scale("ignore"):
        XYZ_D65 = chromatic_adaptation_Li2025(
            XYZ, XYZ_w, XYZ_w_D65, L_A, surround.F, discount_illuminant
        )

    with sdiv_mode():
        XYZ_D65 = sdiv(XYZ_D65, Y_w[..., None])

    with domain_range_scale("ignore"):
        I, C, h = tsplit(sUCS_Iab_to_sUCS_ICh(XYZ_to_sUCS(XYZ_D65)))  # noqa: E741

    I_a = 100 * spow(I / 100, surround.c * z)

    e_t = 1 + 0.06 * np.cos(np.radians(110 + h))

    with sdiv_mode():
        M = (C * spow(F_L, 0.1) * sdiv(1, spow(I_a, 0.27)) * e_t) * surround.F
        # The original paper contained two inconsistent formulas for calculating Q:
        # Equation (15) on page 6 uses an exponent of 0.1, while page 10 uses 0.46.
        # After confirmation with the author, 0.1 is the recommended value.
        Q = sdiv(2, surround.c) * I_a * spow(F_L, 0.1)

    H = hue_quadrature(h)

    V = np.sqrt(I_a**2 + 3 * C**2)

    K = 100 - V

    D = 1.3 * np.sqrt((100 - I_a) ** 2 + 1.6 * C**2)

    W = 100 - D

    return CAM_Specification_sCAM(
        J=as_float(from_range_100(I_a)),
        C=as_float(from_range_100(C)),
        h=as_float(from_range_degrees(h)),
        Q=as_float(from_range_100(Q)),
        M=as_float(from_range_100(M)),
        H=as_float(from_range_degrees(H, 400)),
        HC=None,
        V=as_float(from_range_100(V)),
        K=as_float(from_range_100(K)),
        W=as_float(from_range_100(W)),
        D=as_float(from_range_100(D)),
    )


def sCAM_to_XYZ(
    specification: Annotated[
        CAM_Specification_sCAM, (100, 100, 360, 100, 100, 400, 100, 100, 100, 100)
    ],
    XYZ_w: Domain100,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: InductionFactors_sCAM = VIEWING_CONDITIONS_sCAM["Average"],
    discount_illuminant: bool = False,
) -> Range100:
    """
    Convert the *sCAM* colour appearance model specification to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    specification
        *sCAM* colour appearance model specification.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often
        taken to be 20% of the luminance of a white object in the scene).
    Y_b
        Luminous factor of background :math:`Y_b` such as
        :math:`Y_b = 100 \\times L_b / L_w` where :math:`L_w` is the
        luminance of the light source and :math:`L_b` is the luminance of
        the background.
    surround
        Surround viewing conditions induction factors.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

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
    | ``specification.M`` | 100                   | 1             |
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
    :cite:`Li2024`

    Examples
    --------
    >>> specification = CAM_Specification_sCAM(
    ...     J=49.979566801800047, C=0.014053112120697316, h=328.2724924444729
    ... )
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20
    >>> sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b)  # doctest: +ELLIPSIS
    array([ 19.01...,  20...  ,  21.78...])
    """

    I_a, C, h, _Q, M, _H, _HC, _V, _K, _W, _D = astuple(specification)

    I_a = to_domain_100(I_a)
    C = to_domain_100(C) if not has_only_nan(C) else None
    h = to_domain_degrees(h)
    M = to_domain_100(M) if not has_only_nan(M) else None

    XYZ_w = to_domain_100(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    if has_only_nan(I_a) or has_only_nan(h):
        error = (
            '"J" and "h" correlates must be defined in '
            'the "CAM_Specification_sCAM" argument!'
        )

        raise ValueError(error)

    if has_only_nan(C) and has_only_nan(M):  # pyright: ignore
        error = (
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_sCAM" argument!'
        )

        raise ValueError(error)

    Y_w = XYZ_w[..., 1] if XYZ_w.ndim > 1 else XYZ_w[1]

    with sdiv_mode():
        z = 1.48 + spow(sdiv(Y_b, Y_w), 0.5)

    if C is None and M is not None:
        F_L = 0.1710 * spow(L_A, 1 / 3) / (1 - 0.4934 * np.exp(-0.9934 * L_A))
        e_t = 1 + 0.06 * np.cos(np.radians(110 + h))

        with sdiv_mode():
            C = sdiv(M * spow(I_a, 0.27), spow(F_L, 0.1) * e_t * surround.F)

    with sdiv_mode():
        I = 100 * spow(sdiv(I_a, 100), sdiv(1, surround.c * z))  # noqa: E741

    with domain_range_scale("ignore"):
        XYZ_D65 = sUCS_to_XYZ(sUCS_ICh_to_sUCS_Iab(tstack([I, C, h])))  # type: ignore[arg-type]

    XYZ_D65 = XYZ_D65 * Y_w[..., None]

    L_A_D65 = sdiv(L_A * 100, Y_b)
    XYZ_w_D65 = TVS_D65_sCAM * L_A_D65[..., None]

    with domain_range_scale("ignore"):
        XYZ = chromatic_adaptation_Li2025(
            XYZ_D65,
            XYZ_w_D65,
            XYZ_w,
            L_A,
            surround.F,
            discount_illuminant,
        )

    return from_range_100(XYZ)


def hue_quadrature(h: ArrayLike) -> NDArrayFloat:
    """
    Compute the *hue* quadrature :math:`H` from the specified *hue* angle
    :math:`h`.

    Parameters
    ----------
    h
        *Hue* angle :math:`h` in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        *Hue* quadrature :math:`H`.

    Notes
    -----
    +---------------------+-----------------------+---------------+
    | **Domain**          | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``h``      | 360                   | 1             |
    +---------------------+-----------------------+---------------+

    +---------------------+-----------------------+---------------+
    | **Range**           | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``H``      | 400                   | 1             |
    +---------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> h = np.array([0, 90, 180, 270])
    >>> hue_quadrature(h)  # doctest: +ELLIPSIS
    array([ 386.7962881...,  122.2477064...,  229.5474711...,  326.8471216...])
    """

    h = as_float_array(h)
    h_n = as_float_array(h % 360)

    h_i = HUE_DATA_FOR_HUE_QUADRATURE_sCAM["h_i"]
    e_i = HUE_DATA_FOR_HUE_QUADRATURE_sCAM["e_i"]
    H_i = HUE_DATA_FOR_HUE_QUADRATURE_sCAM["H_i"]

    h_n[np.asarray(np.isnan(h_n))] = 0
    h_n = np.where(h_n < h_i[0], h_n + 360, h_n)

    i = np.searchsorted(h_i, h_n, side="right") - 1
    i = np.clip(i, 0, len(h_i) - 2)

    h1 = h_i[i]
    e1 = e_i[i]
    H1 = H_i[i]

    h2_idx = (i + 1) % len(h_i)
    h2 = h_i[h2_idx]
    e2 = e_i[i + 1]

    h2 = np.where(h2 < h1, h2 + 360, h2)

    with sdiv_mode():
        term1 = sdiv(h_n - h1, e1)
        term2 = sdiv(h2 - h_n, e2)

        H = H1 + 100 * sdiv(term1, term1 + term2)

    return as_float(H)
