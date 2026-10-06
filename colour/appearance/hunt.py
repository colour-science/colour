"""
Hunt Colour Appearance Model
============================

Define the *Hunt* colour appearance model for predicting perceptual colour
attributes under varying viewing conditions.

-   :class:`colour.appearance.InductionFactors_Hunt`
-   :attr:`colour.VIEWING_CONDITIONS_HUNT`
-   :class:`colour.CAM_Specification_Hunt`
-   :func:`colour.XYZ_to_Hunt`

References
----------
-   :cite:`Fairchild2013u` : Fairchild, M. D. (2013). The Hunt Model. In Color
    Appearance Models (3rd ed., pp. 5094-5556). Wiley. ISBN:B00DAYO8E2
-   :cite:`Hunt2004b` : Hunt, R. W. G. (2004). The Reproduction of Colour (6th
    ed.). John Wiley & Sons, Ltd. doi:10.1002/0470024275
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np

from colour.algebra import spow, vecmul

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Domain100

from colour.hints import Annotated, NDArrayFloat, cast
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArithmetic,
    MixinDataclassIterable,
    array_namespace,
    as_float,
    as_float_array,
    from_range_degrees,
    to_domain_100,
    tsplit,
    tstack,
    usage_warning,
    xp_as_float_array,
    xp_degrees,
    xp_interp,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "InductionFactors_Hunt",
    "VIEWING_CONDITIONS_HUNT",
    "HUE_DATA_FOR_HUE_QUADRATURE",
    "MATRIX_XYZ_TO_HPE",
    "MATRIX_HPE_TO_XYZ",
    "CAM_ReferenceSpecification_Hunt",
    "CAM_Specification_Hunt",
    "XYZ_to_Hunt",
    "luminance_level_adaptation_factor",
    "XYZ_to_rgb",
]


@dataclass(frozen=True)
class InductionFactors_Hunt(MixinDataclassIterable):
    """
    Define the *Hunt* colour appearance model induction factors.

    Parameters
    ----------
    N_c
        Chromatic surround induction factor :math:`N_c`.
    N_b
        Brightness surround induction factor :math:`N_b`.
    N_cb
        Chromatic background induction factor :math:`N_{cb}`, approximated
        using tristimulus values :math:`Y_w` and :math:`Y_b` of
        respectively the reference white and the background if not specified.
    N_bb
        Brightness background induction factor :math:`N_{bb}`, approximated
        using tristimulus values :math:`Y_w` and :math:`Y_b` of
        respectively the reference white and the background if not
        specified.

    References
    ----------
    :cite:`Fairchild2013u`, :cite:`Hunt2004b`
    """

    N_c: float
    N_b: float
    N_cb: float | None = field(default_factory=lambda: None)
    N_bb: float | None = field(default_factory=lambda: None)


VIEWING_CONDITIONS_HUNT: CanonicalMapping = CanonicalMapping(
    {
        "Small Areas, Uniform Background & Surrounds": InductionFactors_Hunt(1, 300),
        "Normal Scenes": InductionFactors_Hunt(1, 75),
        "Television & CRT, Dim Surrounds": InductionFactors_Hunt(1, 25),
        "Large Transparencies On Light Boxes": InductionFactors_Hunt(0.7, 25),
        "Projected Transparencies, Dark Surrounds": InductionFactors_Hunt(0.7, 10),
    }
)
VIEWING_CONDITIONS_HUNT.__doc__ = """
Define the reference *Hunt* colour appearance model viewing conditions.

References
----------
:cite:`Fairchild2013u`, :cite:`Hunt2004b`

Aliases:

-   'small_uniform': 'Small Areas, Uniform Background & Surrounds'
-   'normal': 'Normal Scenes'
-   'tv_dim': 'Television & CRT, Dim Surrounds'
-   'light_boxes': 'Large Transparencies On Light Boxes'
-   'projected_dark': 'Projected Transparencies, Dark Surrounds'
"""
VIEWING_CONDITIONS_HUNT["small_uniform"] = VIEWING_CONDITIONS_HUNT[
    "Small Areas, Uniform Background & Surrounds"
]
VIEWING_CONDITIONS_HUNT["normal"] = VIEWING_CONDITIONS_HUNT["Normal Scenes"]
VIEWING_CONDITIONS_HUNT["tv_dim"] = VIEWING_CONDITIONS_HUNT[
    "Television & CRT, Dim Surrounds"
]
VIEWING_CONDITIONS_HUNT["light_boxes"] = VIEWING_CONDITIONS_HUNT[
    "Large Transparencies On Light Boxes"
]
VIEWING_CONDITIONS_HUNT["projected_dark"] = VIEWING_CONDITIONS_HUNT[
    "Projected Transparencies, Dark Surrounds"
]

HUE_DATA_FOR_HUE_QUADRATURE: dict = {
    "h_s": np.array([20.14, 90.00, 164.25, 237.53]),
    "e_s": np.array([0.8, 0.7, 1.0, 1.2]),
}

MATRIX_XYZ_TO_HPE: NDArrayFloat = np.array(
    [
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ]
)
"""
*Hunt* colour appearance model *CIE XYZ* tristimulus values to
*Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace matrix.
"""

MATRIX_HPE_TO_XYZ: NDArrayFloat = np.linalg.inv(MATRIX_XYZ_TO_HPE)
"""
*Hunt* colour appearance model *Hunt-Pointer-Estevez*
:math:`\\rho\\gamma\\beta` colourspace to *CIE XYZ* tristimulus values matrix.
"""


@dataclass
class CAM_ReferenceSpecification_Hunt(MixinDataclassArithmetic):
    """
    Define the *Hunt* colour appearance model reference specification.

    This specification contains field names consistent with the *Fairchild
    (2013)* reference.

    Parameters
    ----------
    J
        Correlate of *Lightness* :math:`J`.
    C_94
        Correlate of *chroma* :math:`C_{94}`.
    h_S
        *Hue* angle :math:`h_S` in degrees.
    s
        Correlate of *saturation* :math:`s`.
    Q
        Correlate of *brightness* :math:`Q`.
    M_94
        Correlate of *colourfulness* :math:`M_{94}`.
    H
        *Hue* :math:`h` quadrature :math:`H`.
    H_C
        *Hue* :math:`h` composition :math:`H_C`.

    References
    ----------
    :cite:`Fairchild2013u`, :cite:`Hunt2004b`
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C_94: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h_S: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M_94: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H_C: float | NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class CAM_Specification_Hunt(MixinDataclassArithmetic):
    """
    Define the *Hunt* colour appearance model specification.

    This specification provides a standardized interface for the *Hunt* model
    with field names consistent across all colour appearance models in
    :mod:`colour.appearance`. While the field names differ from the original
    *Fairchild (2013)* reference notation, they map directly to the model's
    perceptual correlates.

    Parameters
    ----------
    J
        Correlate of *lightness* :math:`J`.
    C
        Correlate of *chroma* :math:`C_{94}`.
    h
        *Hue* angle :math:`h_s` in degrees.
    s
        Correlate of *saturation* :math:`s`.
    Q
        Correlate of *brightness* :math:`Q`.
    M
        Correlate of *colourfulness* :math:`M_{94}`.
    H
        *Hue* :math:`h` quadrature :math:`H`.
    HC
        *Hue* :math:`h` composition :math:`H_C`.

    Notes
    -----
    -   This specification is the one used in the current model
        implementation.

    References
    ----------
    :cite:`Fairchild2013u`, :cite:`Hunt2004b`
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_Hunt(
    XYZ: Domain100,
    XYZ_w: Domain100,
    XYZ_b: Domain100,
    L_A: ArrayLike,
    surround: InductionFactors_Hunt = VIEWING_CONDITIONS_HUNT["Normal Scenes"],
    L_AS: ArrayLike | None = None,
    CCT_w: ArrayLike | None = None,
    XYZ_p: Annotated[ArrayLike | None, 100] = None,
    p: ArrayLike | None = None,
    S: ArrayLike | None = None,
    S_w: ArrayLike | None = None,
    helson_judd_effect: bool = False,
    discount_illuminant: bool = True,
) -> Annotated[CAM_Specification_Hunt, 360]:
    """
    Compute the *Hunt* colour appearance model correlates from the specified
    *CIE XYZ* tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    XYZ_b
        *CIE XYZ* tristimulus values of background.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    surround
        Surround viewing conditions induction factors.
    L_AS
        Scotopic luminance :math:`L_{AS}` of the illuminant,
        approximated if not specified.
    CCT_w
        Correlated colour temperature :math:`T_{cp}` of the illuminant,
        required to approximate :math:`L_{AS}` when not specified.
    XYZ_p
        *CIE XYZ* tristimulus values of proximal field, assumed to equal
        background if not specified.
    p
        Simultaneous contrast / assimilation factor :math:`p` with value
        normalised to domain [-1, 0] for simultaneous contrast and
        normalised to domain [0, 1] for assimilation.
    S
        Scotopic response :math:`S` to the stimulus, approximated using
        tristimulus value :math:`Y` of the stimulus if not specified.
    S_w
        Scotopic response :math:`S_w` for the reference white,
        approximated using tristimulus value :math:`Y_w` of the
        reference white if not specified.
    helson_judd_effect
        Whether to account for the *Helson-Judd* effect.
    discount_illuminant
        Whether to discount the illuminant.

    Returns
    -------
    :class:`colour.CAM_Specification_Hunt`
        *Hunt* colour appearance model specification.

    Raises
    ------
    ValueError
        If an illegal argument combination is specified.

    Notes
    -----
    +---------------------+-----------------------+---------------+
    | **Domain**          | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``XYZ``             | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``XYZ_w``           | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``XYZ_b``           | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``XYZ_p``           | 100                   | 1             |
    +---------------------+-----------------------+---------------+

    +---------------------+-----------------------+---------------+
    | **Range**           | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``specification.h`` | 360                   | 1             |
    +---------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2013u`, :cite:`Hunt2004b`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> XYZ_b = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> surround = VIEWING_CONDITIONS_HUNT["Normal Scenes"]
    >>> CCT_w = 6504
    >>> XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)
    ... # doctest: +ELLIPSIS
    CAM_Specification_Hunt(J=np.float64(30.0462678...), C=np.float64(0.1210508...), \
h=np.float64(269.2737594...), s=np.float64(0.0199093...), Q=np.float64(22.2097654...), \
M=np.float64(0.1238964...), H=None, HC=None)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    XYZ_b = to_domain_100(XYZ_b)

    xp = array_namespace(XYZ, XYZ_w, XYZ_b, L_A)

    XYZ = xp_as_float_array(XYZ, xp=xp)
    XYZ_w = xp_as_float_array(XYZ_w, xp=xp, like=XYZ)
    L_A = xp_as_float_array(L_A, xp=xp, like=XYZ)

    _X, Y, _Z = tsplit(XYZ)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    X_b, Y_b, _Z_b = tsplit(XYZ_b)

    # Arguments handling.
    if XYZ_p is not None:
        X_p, Y_p, Z_p = tsplit(to_domain_100(XYZ_p))
    else:
        X_p = X_b
        Y_p = Y_b
        Z_p = Y_b
        usage_warning(
            'Unspecified proximal field "XYZ_p" argument, using '
            'background "XYZ_b" as approximation!'
        )

    if surround.N_cb is None:
        N_cb = 0.725 * spow(Y_w / Y_b, 0.2)
        usage_warning(f'Unspecified "N_cb" argument, using approximation: "{N_cb}"')
    else:
        N_cb = surround.N_cb
    if surround.N_bb is None:
        N_bb = 0.725 * spow(Y_w / Y_b, 0.2)
        usage_warning(f'Unspecified "N_bb" argument, using approximation: "{N_bb}"')
    else:
        N_bb = surround.N_bb

    if L_AS is None and CCT_w is None:
        error = (
            'Either the scotopic luminance "L_AS" of the '
            "illuminant or its correlated colour temperature "
            '"CCT_w" must be specified!'
        )

        raise ValueError(error)
    if L_AS is None and CCT_w is not None:
        # Approximating scotopic luminance :math:`L_{AS}` from the correlated
        # colour temperature :math:`T_{cp}` per *Hunt (2004)*, "The
        # Reproduction of Colour", 6th ed., section on scotopic responses.
        L_AS = 2.26 * L_A * spow((cast("NDArrayFloat", CCT_w) / 4000) - 0.4, 1 / 3)
        usage_warning(
            f'Unspecified "L_AS" argument, using approximation from "CCT": "{L_AS}"'
        )

    if (S is None and S_w is not None) or (S is not None and S_w is None):
        error = (
            'Either both stimulus scotopic response "S" and '
            'reference white scotopic response "S_w" arguments '
            "need to be specified or none of them!"
        )

        raise ValueError(error)
    if S is None and S_w is None:
        S_p = Y
        S_w_p = Y_w
        usage_warning(
            f'Unspecified stimulus scotopic response "S" and reference white '
            f'scotopic response "S_w" arguments, using approximation: '
            f'"{S}", "{S_w}"'
        )
    else:
        # Both ``S`` and ``S_w`` are non-*None* here: the mixed-*None* case
        # raises at the ``ValueError`` guard above.
        S_p = xp_as_float_array(cast("ArrayLike", S), xp=xp, like=XYZ)
        S_w_p = xp_as_float_array(cast("ArrayLike", S_w), xp=xp, like=XYZ)

    if p is None:
        usage_warning(
            'Unspecified simultaneous contrast / assimilation "p" '
            "argument, model will not account for simultaneous chromatic "
            "contrast!"
        )

    XYZ_p = xp_as_float_array(tstack([X_p, Y_p, Z_p]), xp=xp, like=XYZ)

    # Computing luminance level adaptation factor :math:`F_L`.
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)

    # Computing chromatic adaptation common to the stimulus and reference
    # white. Only the final cone-response step is computed twice; the
    # adaptation factors :math:`h_{rgb}`, :math:`F_{rgb}`, :math:`D_{rgb}`
    # and :math:`B_{rgb}` depend only on the white and the viewing
    # conditions.
    rgb = vecmul(MATRIX_XYZ_TO_HPE, XYZ)
    rgb_w = vecmul(MATRIX_XYZ_TO_HPE, XYZ_w)
    h_rgb = 3 * rgb_w / xp.sum(rgb_w, axis=-1)[..., None]
    if not discount_illuminant:
        L_A_p = spow(L_A, 1 / 3)
        F_rgb = (1 + L_A_p + h_rgb) / (1 + L_A_p + (1 / h_rgb))
    else:
        F_rgb = xp.ones_like(h_rgb)

    def _f_n(x: NDArrayFloat) -> NDArrayFloat:
        x_p = spow(x, 0.73)

        return 40 * (x_p / (x_p + 2))

    if helson_judd_effect:
        Y_b_Y_w = Y_b / Y_w
        Y_b_Y_w_F_L = Y_b_Y_w * F_L
        D_rgb = _f_n(Y_b_Y_w_F_L * F_rgb[..., 1]) - _f_n(Y_b_Y_w_F_L * F_rgb)
    else:
        D_rgb = xp.zeros_like(F_rgb)
    B_rgb = 10**7 / (10**7 + 5 * L_A[..., None] * (rgb_w / 100))

    # Proximal-/background-adjusted reference white ``rgb_w_adapted`` used as
    # the adaptation denominator (*Fairchild (2013)* Eq. 12.23-12.28). ``rgb_w``
    # itself stays unadjusted: only the adaptation reference is adjusted, not
    # the white stimulus.
    if XYZ_p is not None and p is not None:
        p = xp_as_float_array(p, xp=xp, like=XYZ)
        rgb_p = vecmul(MATRIX_XYZ_TO_HPE, XYZ_p)
        rgb_b = vecmul(MATRIX_XYZ_TO_HPE, XYZ_b)
        p_rgb = rgb_p / rgb_b
        rgb_w_adapted = (
            rgb_w
            * (spow((1 - p) * p_rgb + (1 + p) / p_rgb, 0.5))
            / (spow((1 + p) * p_rgb + (1 - p) / p_rgb, 0.5))
        )
    else:
        rgb_w_adapted = rgb_w

    # Final cone-response step for the stimulus and the reference white,
    # ``rgb_a = 1 + B_rgb * (f_n(F_L * F_rgb * rgb / rgb_w_adapted) + D_rgb)``.
    F_L_F_rgb = F_L[..., None] * F_rgb
    rgb_n = F_L_F_rgb * rgb / rgb_w_adapted
    rgb_a = 1.0 + B_rgb * (_f_n(rgb_n) + D_rgb)
    rgb_w_n = F_L_F_rgb * rgb_w / rgb_w_adapted
    rgb_aw = 1.0 + B_rgb * (_f_n(rgb_w_n) + D_rgb)

    # Computing the achromatic post-adaptation signals :math:`A_a` and
    # :math:`A_{aw}` from the adapted cone responses.
    r_a, g_a, b_a = tsplit(rgb_a)
    A_a = 2 * r_a + g_a + (1 / 20) * b_a - 3.05 + 1
    r_aw, g_aw, b_aw = tsplit(rgb_aw)
    A_aw = 2 * r_aw + g_aw + (1 / 20) * b_aw - 3.05 + 1

    # Computing the colour difference signals :math:`C_1`, :math:`C_2`,
    # :math:`C_3` from the adapted cone responses for the stimulus and
    # the reference white.
    C_1 = r_a - g_a
    C_2 = g_a - b_a
    C_3 = b_a - r_a
    C_1_w = r_aw - g_aw
    C_2_w = g_aw - b_aw
    C_3_w = b_aw - r_aw

    # Computing the *hue* angle :math:`h` in degrees in
    # :math:`[0, 360)`.
    # TODO: Implement hue quadrature & composition computation.
    h = xp_degrees(xp.atan2(0.5 * (C_2 - C_3) / 4.5, C_1 - (C_2 / 11))) % 360

    # Computing the eccentricity factor :math:`e_s` from the hue
    # quadrature table with linear extensions outside the
    # :math:`[20.14, 237.53]` range.
    h_s = xp_as_float_array(HUE_DATA_FOR_HUE_QUADRATURE["h_s"], xp=xp, like=h)
    e_s_lut = xp_as_float_array(HUE_DATA_FOR_HUE_QUADRATURE["e_s"], xp=xp, like=h)
    e_s = xp_interp(h, h_s, e_s_lut, xp=xp)
    e_s = xp.where(h < 20.14, 0.856 - (h / 20.14) * 0.056, e_s)
    e_s = xp.where(h > 237.53, 0.856 + 0.344 * (360 - h) / (360 - 237.53), e_s)

    # Computing the low-luminance tritanopia factor :math:`F_t`.
    F_t = L_A / (L_A + 0.1)

    # Computing the yellowness-blueness :math:`M_{yb}` and
    # redness-greenness :math:`M_{rg}` responses for the stimulus and
    # the reference white.
    N_c = surround.N_c
    yb_factor = e_s * (10 / 13) * N_c * N_cb
    M_yb = 100 * (0.5 * (C_2 - C_3) / 4.5) * yb_factor * F_t
    M_rg = 100 * (C_1 - (C_2 / 11)) * yb_factor
    M_yb_w = 100 * (0.5 * (C_2_w - C_3_w) / 4.5) * yb_factor * F_t
    M_rg_w = 100 * (C_1_w - (C_2_w / 11)) * yb_factor

    # Computing the overall chromatic response :math:`M`.
    M = xp.hypot(M_yb, M_rg)
    M_w = xp.hypot(M_yb_w, M_rg_w)

    # Computing the correlate of *saturation* :math:`s`.
    s = 50 * M / xp.sum(rgb_a, axis=-1)

    # Computing achromatic signals :math:`A` and :math:`A_w` from the
    # achromatic post-adaptation signals and the scotopic response,
    # mediated by the scotopic luminance level adaptation factor
    # :math:`F_{LS}` and cone bleach factor :math:`B_S`.
    L_AS_226 = cast("NDArrayFloat", L_AS) / 2.26
    j_sc = 0.00001 / ((5 * L_AS_226) + 0.00001)
    F_LS = 3800 * (j_sc**2) * (5 * L_AS_226) + 0.2 * (spow(1 - (j_sc**2), 0.4)) * (
        spow(5 * L_AS_226, 1 / 6)
    )

    S_S_w = S_p / S_w_p
    B_S = 0.5 / (1 + 0.3 * spow((5 * L_AS_226) * S_S_w, 0.3)) + 0.5 / (
        1 + 5 * (5 * L_AS_226)
    )
    A_S = (_f_n(F_LS * S_S_w) * 3.05 * B_S) + 0.3
    A = N_bb * (A_a - 1 + A_S - 0.3 + spow(1 + (0.3**2), 0.5))

    S_S_w_w = S_w_p / S_w_p
    B_S_w = 0.5 / (1 + 0.3 * spow((5 * L_AS_226) * S_S_w_w, 0.3)) + 0.5 / (
        1 + 5 * (5 * L_AS_226)
    )
    A_S_w = (_f_n(F_LS * S_S_w_w) * 3.05 * B_S_w) + 0.3
    A_w = N_bb * (A_aw - 1 + A_S_w - 0.3 + spow(1 + (0.3**2), 0.5))

    # Computing the correlate of *brightness* :math:`Q` for the
    # stimulus and the reference white.
    # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.
    N_b = surround.N_b
    N_1 = spow(7 * A_w, 0.5) / (5.33 * spow(N_b, 0.13))
    N_2 = (7 * A_w * spow(N_b, 0.362)) / 200
    Q = spow(7 * (A + (M / 100)), 0.6) * N_1 - N_2
    brightness_w = spow(7 * (A_w + (M_w / 100)), 0.6) * N_1 - N_2

    # Computing the correlate of *Lightness* :math:`J`.
    Z = 1 + spow(Y_b / Y_w, 0.5)
    J = 100 * spow(Q / brightness_w, Z)

    # Computing the correlate of *chroma* :math:`C_{94}`.
    Y_b_Y_w_ratio = Y_b / Y_w
    C_94 = (
        2.44
        * spow(s, 0.69)
        * spow(Q / brightness_w, Y_b_Y_w_ratio)
        * (1.64 - spow(0.29, Y_b_Y_w_ratio))
    )

    # Computing the correlate of *colourfulness* :math:`M_{94}`.
    M_94 = spow(F_L, 0.15) * C_94

    return CAM_Specification_Hunt(
        J=as_float(J),
        C=as_float(C_94),
        h=as_float(from_range_degrees(h)),
        s=as_float(s),
        Q=as_float(Q),
        M=as_float(M_94),
        H=None,
        HC=None,
    )


def luminance_level_adaptation_factor(
    L_A: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the *luminance* level adaptation factor :math:`F_L`.

    Parameters
    ----------
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* level adaptation factor :math:`F_L`.

    Examples
    --------
    >>> luminance_level_adaptation_factor(318.31)  # doctest: +ELLIPSIS
    np.float64(1.1675444...)
    """

    L_A = as_float_array(L_A)

    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)

    return as_float(F_L)


def XYZ_to_rgb(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *Hunt-Pointer-Estevez*
    :math:`\\rho\\gamma\\beta` colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace values.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_to_rgb(XYZ)  # doctest: +ELLIPSIS
    array([19.4743367..., 20.3101217..., 21.78     ])
    """

    return vecmul(MATRIX_XYZ_TO_HPE, XYZ)
