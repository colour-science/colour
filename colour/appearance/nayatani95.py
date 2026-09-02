"""
Nayatani (1995) Colour Appearance Model
=======================================

Define the *Nayatani (1995)* colour appearance model for predicting
perceptual colour attributes under varying viewing conditions.

-   :class:`colour.CAM_Specification_Nayatani95`
-   :func:`colour.XYZ_to_Nayatani95`

References
----------
-   :cite:`Fairchild2013ba` : Fairchild, M. D. (2013). The Nayatani et al.
    Model. In Color Appearance Models (3rd ed., pp. 4810-5085). Wiley.
    ISBN:B00DAYO8E2
-   :cite:`Nayatani1995a` : Nayatani, Y., Sobagaki, H., & Yano, K. H. T.
    (1995). Lightness dependency of chroma scales of a nonlinear
    color-appearance model and its latest formulation. Color Research &
    Application, 20(3), 156-167. doi:10.1002/col.5080200305
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from colour.adaptation.cie1994 import (
    MATRIX_XYZ_TO_RGB_CIE1994,
    beta_1,
    exponential_factors,
    intermediate_values,
)
from colour.algebra import spow, vecmul

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Domain100

from colour.hints import Annotated, NDArrayFloat, cast
from colour.models import XYZ_to_xy
from colour.utilities import (
    MixinDataclassArithmetic,
    array_namespace,
    as_float,
    as_float_array,
    from_range_degrees,
    to_domain_100,
    tsplit,
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
    "MATRIX_XYZ_TO_RGB_NAYATANI95",
    "CAM_ReferenceSpecification_Nayatani95",
    "CAM_Specification_Nayatani95",
    "XYZ_to_Nayatani95",
    "hue_quadrature",
]

MATRIX_XYZ_TO_RGB_NAYATANI95: NDArrayFloat = MATRIX_XYZ_TO_RGB_CIE1994
"""
*Nayatani (1995)* colour appearance model *CIE XYZ* tristimulus values to cone
responses matrix.
"""


@dataclass
class CAM_ReferenceSpecification_Nayatani95(MixinDataclassArithmetic):
    """
    Define the *Nayatani (1995)* colour appearance model reference
    specification.

    This specification contains field names consistent with the *Fairchild
    (2013)* reference.

    Parameters
    ----------
    L_star_P
        Correlate of *achromatic lightness* :math:`L_p^\\star`.
    C
        Correlate of *chroma* :math:`C`.
    theta
        *Hue* angle :math:`\\theta` in degrees.
    S
        Correlate of *saturation* :math:`S`.
    B_r
        Correlate of *brightness* :math:`B_r`.
    M
        Correlate of *colourfulness* :math:`M`.
    H
        *Hue* :math:`h` quadrature :math:`H`.
    H_C
        *Hue* :math:`h` composition :math:`H_C`.
    L_star_N
        Correlate of *normalised achromatic lightness* :math:`L_n^\\star`.

    References
    ----------
    :cite:`Fairchild2013ba`, :cite:`Nayatani1995a`
    """

    L_star_P: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    theta: float | NDArrayFloat | None = field(default_factory=lambda: None)
    S: float | NDArrayFloat | None = field(default_factory=lambda: None)
    B_r: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H_C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    L_star_N: float | NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class CAM_Specification_Nayatani95(MixinDataclassArithmetic):
    """
    Define the *Nayatani (1995)* colour appearance model specification.

    This specification provides a standardized interface for the
    *Nayatani (1995)* model with field names consistent across all colour
    appearance models in :mod:`colour.appearance`. While the field names differ
    from the original *Fairchild (2013)* reference notation, they map directly
    to the model's perceptual correlates.

    Parameters
    ----------
    L_star_P
        Correlate of *achromatic lightness* :math:`L_p^\\star`.
    C
        Correlate of *chroma* :math:`C`.
    h
        *Hue* angle :math:`\\theta` in degrees.
    s
        Correlate of *saturation* :math:`S`.
    Q
        Correlate of *brightness* :math:`B_r`.
    M
        Correlate of *colourfulness* :math:`M`.
    H
        *Hue* :math:`h` quadrature :math:`H`.
    HC
        *Hue* :math:`h` composition :math:`H_C`.
    L_star_N
        Correlate of *normalised achromatic lightness* :math:`L_n^\\star`.

    Notes
    -----
    -   This specification is the one used in the current model
        implementation.

    References
    ----------
    :cite:`Fairchild2013ba`, :cite:`Nayatani1995a`
    """

    L_star_P: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)
    L_star_N: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_Nayatani95(
    XYZ: Domain100,
    XYZ_n: Domain100,
    Y_o: ArrayLike,
    E_o: ArrayLike,
    E_or: ArrayLike,
    n: ArrayLike = 1,
    compute_H: bool = False,
) -> Annotated[CAM_Specification_Nayatani95, 360]:
    """
    Compute the *Nayatani (1995)* colour appearance model correlates from the
    specified *CIE XYZ* tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_n
        *CIE XYZ* tristimulus values of reference white.
    Y_o
        Luminance factor :math:`Y_o` of achromatic background as percentage
        normalised to domain [0.18, 1.0] in **'Reference'** domain-range
        scale.
    E_o
        Illuminance :math:`E_o` of the viewing field in lux.
    E_or
        Normalising illuminance :math:`E_{or}` in lux usually normalised to
        domain [1000, 3000].
    n
        Noise term used in the non-linear chromatic adaptation model.
    compute_H
        When *True*, compute the *Hue Quadrature* :math:`H` correlate
        via :func:`colour.appearance.nayatani95.hue_quadrature`. Defaults to
        *False* because :math:`H` is rarely consumed downstream and
        skipping the bin search is a measurable cost saving.

    Returns
    -------
    :class:`colour.CAM_Specification_Nayatani95`
        *Nayatani (1995)* colour appearance model specification.

    Notes
    -----
    +---------------------+-----------------------+---------------+
    | **Domain**          | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``XYZ``             | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``XYZ_n``           | 100                   | 1             |
    +---------------------+-----------------------+---------------+

    +---------------------+-----------------------+---------------+
    | **Range**           | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``specification.h`` | 360                   | 1             |
    +---------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2013ba`, :cite:`Nayatani1995a`

    Examples
    --------
    *Fairchild (2013)* Table 11.1 Case 1 (near-grey stimulus, photopic):

    >>> import numpy as np
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_n = np.array([95.05, 100.00, 108.88])
    >>> Y_o = 20.0
    >>> E_o = 5000.0
    >>> E_or = 1000.0
    >>> XYZ_to_Nayatani95(
    ...     XYZ, XYZ_n, Y_o, E_o, E_or,
    ...     compute_H=True,
    ... )  # doctest: +ELLIPSIS
    CAM_Specification_Nayatani95(L_star_P=np.float64(49.9998829...), \
C=np.float64(0.0133550...), h=np.float64(257.5232268...), \
s=np.float64(0.0133550...), Q=np.float64(62.6266734...), \
M=np.float64(0.0167262...), H=np.float64(317.7841135...), HC=None, \
L_star_N=np.float64(50.0039154...))

    *Fairchild (2013)* Table 11.1 Case 2 (chromatic stimulus, lower
    illuminance):

    >>> XYZ_to_Nayatani95(np.array([57.06, 43.06, 31.96]), XYZ_n, 20.0, \
500.0, 1000.0, compute_H=True)  # doctest: +ELLIPSIS
    CAM_Specification_Nayatani95(L_star_P=np.float64(72.9768964...), \
C=np.float64(48.3460111...), h=np.float64(21.5766539...), \
s=np.float64(37.1030727...), Q=np.float64(67.3493717...), \
M=np.float64(42.9012040...), H=np.float64(2.0564758...), HC=None, \
L_star_N=np.float64(75.8970185...))
    """

    XYZ = to_domain_100(XYZ)
    XYZ_n = to_domain_100(XYZ_n)

    xp = array_namespace(XYZ, XYZ_n, Y_o, E_o, E_or)

    Y_o = xp_as_float_array(Y_o, xp=xp, like=XYZ)
    E_o = xp_as_float_array(E_o, xp=xp, like=XYZ)
    E_or = xp_as_float_array(E_or, xp=xp, like=XYZ)
    n = xp_as_float_array(n, xp=xp, like=XYZ)

    # Computing normalising luminance :math:`L_{or}` in :math:`cd/m^2`.
    L_or = Y_o * E_or / (100 * xp.pi)

    # Computing :math:`\\xi`, :math:`\\eta`, :math:`\\zeta` values from the
    # reference white chromaticity (*CIE 1994* chromatic adaptation primitive).
    xez = intermediate_values(XYZ_to_xy(XYZ_n / 100))
    xi, eta, zeta = tsplit(xez)

    # Computing adapting field cone responses.
    RGB_o = ((Y_o[..., None] * E_o[..., None]) / (100 * xp.pi)) * xez

    # Computing stimulus cone responses.
    RGB = vecmul(MATRIX_XYZ_TO_RGB_NAYATANI95, XYZ)
    R, G, B = tsplit(RGB)

    # Computing exponential factors :math:`\\beta_1(R_o)`,
    # :math:`\\beta_1(G_o)`, :math:`\\beta_2(B_o)` and the normalising
    # :math:`\\beta_1(B_{or})` (*CIE 1994* chromatic adaptation primitives).
    # Cast back to the active backend dtype for consistent propagation.
    bRGB_o = xp_as_float_array(exponential_factors(RGB_o), xp=xp, like=XYZ)
    bR_o, bG_o, bB_o = tsplit(bRGB_o)
    bL_or = xp_as_float_array(beta_1(L_or), xp=xp, like=XYZ)

    # Computing scaling coefficients :math:`e(R)` and :math:`e(G)`: 1.758 when
    # the cone response exceeds 20 times the intermediate value, otherwise 1.
    # Both ``xp.where`` branches are bare *Python* scalars, so nothing anchors
    # the result dtype and the backend default, e.g. float32 for stock
    # *PyTorch*, would be adopted; the branches are promoted to the *Colour*
    # default float dtype first.
    one = xp_as_float_array(1, xp=xp, like=XYZ)
    eR = xp.where((20 * xi) <= R, xp_as_float_array(1.758, xp=xp, like=XYZ), one)
    eG = xp.where((20 * eta) <= G, xp_as_float_array(1.758, xp=xp, like=XYZ), one)

    # Computing the logarithmic cone-response terms shared by the achromatic,
    # tritanopic and protanopic opponent responses.
    log_R = xp.log10((R + n) / (20 * xi + n))
    log_G = xp.log10((G + n) / (20 * eta + n))
    log_B = xp.log10((B + n) / (20 * zeta + n))

    # Computing achromatic response :math:`Q` as a weighted combination of the
    # logarithmic red and green opponent terms.
    Q_response = (2 / 3) * bR_o * eR * log_R
    Q_response += (1 / 3) * bG_o * eG * log_G
    Q_response *= 41.69 / bL_or

    # Computing tritanopic response :math:`t`.
    t_response = bR_o * log_R - (12 / 11) * bG_o * log_G + (1 / 11) * bB_o * log_B

    # Computing protanopic response :math:`p`.
    p_response = (
        (1 / 9) * bR_o * log_R + (1 / 9) * bG_o * log_G - (2 / 9) * bB_o * log_B
    )

    # Computing the correlate of *brightness* :math:`B_r`.
    B_r = (50 / bL_or) * ((2 / 3) * bR_o + (1 / 3) * bG_o) + Q_response

    # Computing *brightness* :math:`B_{rw}` of ideal white.
    B_rw = (2 / 3) * bR_o * 1.758 * xp.log10((100 * xi + n) / (20 * xi + n))
    B_rw += (1 / 3) * bG_o * 1.758 * xp.log10((100 * eta + n) / (20 * eta + n))
    B_rw *= 41.69 / bL_or
    B_rw += (50 / bL_or) * (2 / 3) * bR_o
    B_rw += (50 / bL_or) * (1 / 3) * bG_o

    # Computing the correlate of achromatic *Lightness* :math:`L_p^\\star`.
    L_star_P = Q_response + 50

    # Computing the correlate of normalised achromatic *Lightness*
    # :math:`L_n^\\star`.
    L_star_N = 100 * B_r / B_rw

    # Computing the *hue* angle :math:`\\theta` in degrees from the protanopic
    # and tritanopic responses.
    theta = xp_degrees(xp.atan2(p_response, t_response)) % 360
    # Computing the *hue* :math:`h` quadrature :math:`H` only when requested
    # via ``compute_H``; the bin search delegates to :func:`hue_quadrature`,
    # a 400-step linear interpolation between unique-hue angles 20.14, 90.00,
    # 164.25, 231.00 per *Fairchild (2013)* p.202.
    H = hue_quadrature(theta) if compute_H else xp.full_like(theta, float("nan"))
    # TODO: Implement hue composition computation.

    # Computing the chromatic strength function :math:`E_s(\\theta)` used to
    # correct the saturation scale as a function of hue angle.
    theta_rad = xp_radians(theta)
    E_s = cast("NDArrayFloat", 0.9394)
    E_s += -0.2478 * xp.sin(1 * theta_rad)
    E_s += -0.0743 * xp.sin(2 * theta_rad)
    E_s += +0.0666 * xp.sin(3 * theta_rad)
    E_s += -0.0186 * xp.sin(4 * theta_rad)
    E_s += -0.0055 * xp.cos(1 * theta_rad)
    E_s += -0.0521 * xp.cos(2 * theta_rad)
    E_s += -0.0573 * xp.cos(3 * theta_rad)
    E_s += -0.0061 * xp.cos(4 * theta_rad)

    # Computing *saturation* components :math:`S_{RG}` and :math:`S_{YB}` and
    # the *saturation* correlate :math:`S`.
    S_RG = 488.93 / bL_or * E_s * t_response
    S_YB = 488.93 / bL_or * E_s * p_response
    S = xp.hypot(S_RG, S_YB)

    # Computing the correlate of *chroma* :math:`C`.
    C = spow(L_star_P / 50, 0.7) * S

    # Computing the correlate of *colourfulness* :math:`M`.
    # TODO: Investigate components usage.
    M = C * B_rw / 100

    return CAM_Specification_Nayatani95(
        L_star_P=as_float(L_star_P),
        C=as_float(C),
        h=as_float(from_range_degrees(theta)),
        s=as_float(S),
        Q=as_float(B_r),
        M=as_float(M),
        H=as_float(from_range_degrees(H, 400)),
        HC=None,
        L_star_N=as_float(L_star_N),
    )


def hue_quadrature(h: ArrayLike) -> NDArrayFloat:
    """
    Compute hue quadrature :math:`H` from the specified *Nayatani (1995)*
    hue :math:`\\theta` angle in degrees via linear interpolation between
    the four unique-hue angles.

    Parameters
    ----------
    h
        Hue :math:`\\theta` angle in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Hue quadrature :math:`H` in the 400-step *Nayatani (1995)* scale
        (R 0, Y 100, G 200, B 300, wrap to R 400).

    References
    ----------
    :cite:`Fairchild2013ba`, :cite:`Nayatani1995a`

    Examples
    --------
    >>> hue_quadrature(257.5232268)  # doctest: +ELLIPSIS
    np.float64(317.7841134...)
    """

    h = as_float_array(h)

    xp = array_namespace(h)

    h = as_float_array(xp.where(xp.isnan(h), 0, h))

    # Unique-hue angles per *Fairchild (2013)* p.202:
    #   R 20.14, Y 90.00, G 164.25, B 231.00, R 380.14 (wrap).
    # Hue quadrature is a 400-step scale obtained via linear interpolation
    # between consecutive bin boundaries; no eccentricity weighting (unlike
    # *CIECAM02*).
    H_0 = (h - 20.14) / (90.00 - 20.14) * 100
    H_1 = 100 + (h - 90.00) / (164.25 - 90.00) * 100
    H_2 = 200 + (h - 164.25) / (231.00 - 164.25) * 100
    H_3 = 300 + (h - 231.00) / (380.14 - 231.00) * 100

    # ``h < 20.14`` wraps through ``360`` into the B -> R interval.
    H_wrap = 300 + (h + 360 - 231.00) / (380.14 - 231.00) * 100

    H = xp_select(
        [
            (h >= 20.14) & (h < 90.00),
            (h >= 90.00) & (h < 164.25),
            (h >= 164.25) & (h < 231.00),
            h >= 231.00,
        ],
        [H_0, H_1, H_2, H_3],
        default=H_wrap,
        xp=xp,
    )

    return as_float(H)
