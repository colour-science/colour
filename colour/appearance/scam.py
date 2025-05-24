"""
sCAM Colour Appearance Model
============================

Define the *sCAM* colour appearance model objects:

-   :class:`colour.appearance.InductionFactors_sCAM`
-   :attr:`colour.VIEWING_CONDITIONS_sCAM`
-   :class:`colour.CAM_Specification_sCAM`
-   :func:`colour.XYZ_to_sCAM`
-   :func:`colour.sCAM_to_XYZ`

The sCAM (Simple Colour Appearance Model) is based on the sUCS (Simple Uniform
Colour Space).

References
----------
-   :cite:`Li2024` : Li, M., & Luo, M. R. (2024). Simple color appearance model
    (sCAM) based on simple uniform color space (sUCS). Optics Express, 32(3),
    3100-3122. doi:10.1364/OE.510196
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

# cartesian_to_polar is not directly used here,
# _hue_angle_sCAM uses np.arctan2
from colour.algebra import (
    spow,
    vecmul,
)

# pyright: ignore, for XYZ_to_sUCS, sUCS_to_XYZ potentially seen as unused
from colour.models.sucs import XYZ_to_sUCS, sUCS_to_XYZ
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
    tstack,
    zeros,
)

if TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat


SUCS_XYZ_D65_NORMALIZED = [0.95047, 1.00000, 1.08883]

__author__ = "UltraMo114(Molin Li), Colour Developers"
__copyright__ = "Copyright 2024 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "InductionFactors_sCAM",
    "VIEWING_CONDITIONS_sCAM",
    "CAM_Specification_sCAM",
    "XYZ_to_sCAM",
    "sCAM_to_XYZ",
    "CAT16_MATRIX",
    "CAT16_INVERSE_MATRIX",
    "HUE_DATA_sCAM",
]

CAT16_MATRIX: NDArrayFloat = np.array(
    [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ]
)
"""
Chromatic Adaptation Transform CAT16 matrix (same as CAT02).
Used in sCAM for chromatic adaptation.
"""

CAT16_INVERSE_MATRIX: NDArrayFloat = np.array(
    [
        [1.86206785508723, -1.01125463053168, 0.149186775444452],
        [0.387526543236137, 0.621447441931475, -0.00897398516761252],
        [-0.0158414988493339, -0.0341229380285156, 1.04996443687785],
    ]
)
"""
Inverse of the CAT16 matrix.
"""

HUE_DATA_sCAM: dict = {
    "h_i": np.array([16.5987, 80.2763, 157.779, 219.7174, 376.5987]),
    "e_i": np.array(
        [0.7, 0.6, 1.2, 0.9, 0.7, 0.7]
    ),  # Last element repeated for wrap-around logic
    "H_i": np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
}
"""
Hue quadrature data for sCAM.
"""


@dataclass(frozen=True)
class InductionFactors_sCAM:
    """
    sCAM colour appearance model induction factors.
    """

    F: float  # Maximum degree of adaptation factor
    c: float  # Exponential non-linearity for lightness
    Fm: float  # Factor for colourfulness (M)


VIEWING_CONDITIONS_sCAM: CanonicalMapping = CanonicalMapping(
    {
        "Average": InductionFactors_sCAM(F=1.0, c=0.52, Fm=1.0),
        "Dim": InductionFactors_sCAM(F=0.9, c=0.50, Fm=0.95),
        "Dark": InductionFactors_sCAM(F=0.8, c=0.39, Fm=0.85),
    }
)
VIEWING_CONDITIONS_sCAM.__doc__ = """
Reference *sCAM* colour appearance model viewing conditions.
"""


@dataclass
class CAM_Specification_sCAM(MixinDataclassArithmetic):
    """
    Define the *sCAM* colour appearance model specification.
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)  # Lightness
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)  # Chroma
    h: float | NDArrayFloat | None = field(
        default_factory=lambda: None
    )  # Hue angle (degrees)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)  # Brightness
    M: float | NDArrayFloat | None = field(
        default_factory=lambda: None
    )  # Colourfulness
    H: float | NDArrayFloat | None = field(
        default_factory=lambda: None
    )  # Hue composition


def _chromatic_adaptation_sCAM(
    XYZ_s: ArrayLike,
    XYZ_w_s: ArrayLike,  # Whitepoint of source for XYZ_s
    XYZ_w_d: ArrayLike,  # Whitepoint of destination (target)
    L_A: ArrayLike,
    F_surround: ArrayLike,
    discount_illuminant: bool = False,
) -> NDArrayFloat:
    """
    Change a color's appearance from one lighting to another using CAT16.

    This function takes a color (`XYZ_s`) seen under an initial light
    source (`XYZ_w_s`) and calculates how it would look under a new
    light source (`XYZ_w_d`). It considers the adapting light's luminance
    (`L_A`) and the viewing surround (`F_surround`).

    Parameters
    ----------
    XYZ_s
        The XYZ values of the input color (seen under `XYZ_w_s`).
    XYZ_w_s
        The XYZ whitepoint of the original light source.
    XYZ_w_d
        The XYZ whitepoint of the new (target) light source.
    L_A
        Adapting light level (brightness in cd/m²).
    F_surround
        Factor for how the surroundings affect adaptation (e.g., 1.0 for
        average, 0.9 for dim).
    discount_illuminant
        If `True`, assumes complete adaptation to the light source (D=1).

    Returns
    -------
    NDArrayFloat
        The adapted XYZ values of the color, as it would appear under the
        new light source (`XYZ_w_d`).
    """
    XYZ_s_arr = as_float_array(XYZ_s)
    XYZ_w_s_arr = as_float_array(XYZ_w_s)
    XYZ_w_d_arr = as_float_array(XYZ_w_d)
    L_A_arr = as_float_array(L_A)
    F_surround_arr = as_float_array(F_surround)

    LMS_s = vecmul(CAT16_MATRIX, XYZ_s_arr)  # LMS of input XYZ_s

    LMS_w_s_ratio_calc = vecmul(CAT16_MATRIX, XYZ_w_s_arr)
    LMS_w_d_ratio_calc = vecmul(CAT16_MATRIX, XYZ_w_d_arr)

    if XYZ_w_s_arr.ndim > 1:
        Y_w_s_val_ratio_calc = XYZ_w_s_arr[..., 1, np.newaxis]
        Y_w_d_val_ratio_calc = XYZ_w_d_arr[..., 1, np.newaxis]
    else:
        Y_w_s_val_ratio_calc = XYZ_w_s_arr[1]
        Y_w_d_val_ratio_calc = XYZ_w_d_arr[1]

    if discount_illuminant:
        D = ones(np.shape(L_A_arr))
    else:
        D_pre = F_surround_arr * (1 - (1 / 3.6) * np.exp((-L_A_arr - 42) / 92))
        D = np.clip(D_pre, 0, 1)

    # Ensure D has compatible dimensions
    if D.ndim == 0 and LMS_s.ndim > 1 and D.shape != LMS_s.shape[-LMS_s.ndim :]:
        D_reshaped = D.reshape(
            [1] * (LMS_s.ndim - 1) + [D.size] if D.size > 1 else [1] * LMS_s.ndim
        )
    elif D.ndim < LMS_s.ndim:
        D_reshaped = D[np.newaxis, ...]
    else:
        D_reshaped = D

    LMS_w_s_safe = np.where(LMS_w_s_ratio_calc == 0, 1e-10, LMS_w_s_ratio_calc)
    Y_w_d_safe = np.where(Y_w_d_val_ratio_calc == 0, 1e-10, Y_w_d_val_ratio_calc)

    # This ratio adapts a stimulus from XYZ_w_s_arr
    # to appear as if under XYZ_w_d_arr
    adaptation_ratios = D_reshaped * (Y_w_s_val_ratio_calc / Y_w_d_safe) * (
        LMS_w_d_ratio_calc / LMS_w_s_safe
    ) + (1 - D_reshaped)
    LMS_a = adaptation_ratios * LMS_s
    # XYZ_a = vecmul(CAT16_INVERSE_MATRIX, LMS_a) # Original for RET504
    return vecmul(CAT16_INVERSE_MATRIX, LMS_a)


def _hue_angle_sCAM(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """Compute hue angle in degrees [0, 360]."""
    a_arr = as_float_array(a)
    b_arr = as_float_array(b)
    h_rad = np.arctan2(b_arr, a_arr)
    h_deg = np.degrees(h_rad)
    return as_float_array(h_deg % 360)


def _hue_composition_sCAM(h: ArrayLike) -> NDArrayFloat:
    """Compute hue composition H from hue angle h."""
    h_arr = as_float_array(h)
    h_norm = h_arr % 360

    h_i_data = HUE_DATA_sCAM["h_i"]
    e_i_data = HUE_DATA_sCAM["e_i"]
    H_i_lookup = HUE_DATA_sCAM["H_i"]

    original_shape = h_arr.shape
    h_flat = h_norm.flatten()
    H_comp_flat = zeros(h_flat.shape)

    for idx, h_val in enumerate(h_flat):
        current_h = h_val
        if current_h < h_i_data[0]:
            current_h += 360

        k = np.searchsorted(h_i_data, current_h, side="right")
        segment_idx = k - 1
        if segment_idx < 0:
            # Ensures wrap around for segment_idx, was len(h_i_data) - 1
            segment_idx = len(h_i_data) - 2  # Index for last segment start

        h1 = h_i_data[segment_idx]
        H1 = H_i_lookup[segment_idx]
        e1 = e_i_data[segment_idx]

        h2_idx = (segment_idx + 1) % len(h_i_data)
        h2 = h_i_data[h2_idx]
        e2 = e_i_data[segment_idx + 1]  # e_i_data has one extra element for this

        if h2 < h1:  # Handles case where h2 is from start of array (e.g. h_i_data[0])
            h2 += 360

        e1_safe = np.where(e1 == 0, 1e-10, e1)
        e2_safe = np.where(e2 == 0, 1e-10, e2)

        term1 = (current_h - h1) / e1_safe
        term2 = (h2 - current_h) / e2_safe

        denominator = term1 + term2
        if denominator == 0:
            if np.isclose(current_h, h1):
                H_comp_flat[idx] = H1
            elif np.isclose(current_h, h2):
                H_comp_flat[idx] = H_i_lookup[h2_idx]
            else:
                H_comp_flat[idx] = H1  # Fallback
        else:
            H_comp_flat[idx] = H1 + 100 * term1 / denominator

    return H_comp_flat.reshape(original_shape)


def XYZ_to_sCAM(
    XYZ: ArrayLike,  # Absolute XYZ of stimulus
    XYZ_w: ArrayLike,  # Absolute XYZ of source whitepoint
    L_A: ArrayLike,  # Adapting luminance (cd/m^2)
    Y_b: ArrayLike,  # Luminance factor of background (e.g., 20 for 20% grey)
    surround: InductionFactors_sCAM | str = "Average",
    discount_illuminant: bool = False,
) -> CAM_Specification_sCAM:
    """Compute sCAM correlates from XYZ."""
    # XYZ = as_float_array(XYZ) # Original commented out
    # XYZ_w = as_float_array(XYZ_w) # Original commented out
    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    if isinstance(surround, str):
        surround_factors = VIEWING_CONDITIONS_sCAM[surround]
    else:
        surround_factors = surround

    F_s_param = surround_factors.F
    c_s_param = surround_factors.c
    Fm_s_param = surround_factors.Fm

    Y_w_val = XYZ_w[..., 1] if XYZ_w.ndim > 1 else XYZ_w[1]
    Y_w_val_safe = np.where(Y_w_val == 0, 1e-10, Y_w_val)
    n = Y_b / Y_w_val_safe
    z = 1.48 + np.sqrt(np.maximum(0, n))

    Lw_d65_ref_luminance_val = L_A * 100.0 / np.where(Y_b == 0, 1e-10, Y_b)
    Lw_d65_ref_luminance = np.where(
        Lw_d65_ref_luminance_val == 0, 1e-10, Lw_d65_ref_luminance_val
    )

    XYZ_w_d65_cat_target = SUCS_XYZ_D65_NORMALIZED * Lw_d65_ref_luminance

    XYZ_adapted_abs = _chromatic_adaptation_sCAM(
        XYZ, XYZ_w, XYZ_w_d65_cat_target, L_A, F_s_param, discount_illuminant
    )

    # Normalize XYZ_adapted_abs for XYZ_to_sUCS (expects D65, Y=1)
    if Y_w_val_safe.ndim == 1:  # Logic kept as original
        XYZ_adapted_normalized = XYZ_adapted_abs / Y_w_val_safe[:, np.newaxis]
    else:
        XYZ_adapted_normalized = XYZ_adapted_abs / Y_w_val_safe

    sucs_output = XYZ_to_sUCS(XYZ_adapted_normalized)  # pyright: ignore

    I_S_ucs = sucs_output[..., 0]
    A_S_final_ucs = sucs_output[..., 1]
    B_S_final_ucs = sucs_output[..., 2]

    C_S_ucs = np.sqrt(A_S_final_ucs**2 + B_S_final_ucs**2)

    J_scam = 100 * spow(np.clip(I_S_ucs / 100.0, 0, None), c_s_param * z)
    h_scam = _hue_angle_sCAM(A_S_final_ucs, B_S_final_ucs)

    FL_num = 0.1710 * spow(np.maximum(0, L_A), 1.0 / 3.0)
    FL_den_exp_term = np.exp(-0.9934 * L_A)
    FL_den = 1.0 - 0.4934 * FL_den_exp_term
    FL = FL_num / np.where(FL_den == 0, 1e-10, FL_den)

    et = 1.0 + 0.06 * np.cos(np.radians(110.0 + h_scam))
    H_scam = _hue_composition_sCAM(h_scam)

    J_scam_safe = np.where(J_scam == 0, 1e-10, J_scam)
    M_scam = (C_S_ucs * spow(FL, 0.1) / spow(J_scam_safe, 0.27) * et) * Fm_s_param

    c_s_param_safe = np.where(c_s_param == 0, 1e-10, c_s_param)
    Q_scam = (2.0 / c_s_param_safe) * J_scam * spow(FL, 0.46)
    C_scam = C_S_ucs

    return CAM_Specification_sCAM(
        J=as_float(from_range_100(J_scam)),
        C=as_float(from_range_100(C_scam)),
        h=as_float(from_range_degrees(h_scam)),
        Q=as_float(from_range_100(Q_scam)),
        M=as_float(from_range_100(M_scam)),
        H=as_float(from_range_degrees(H_scam, 400)),
    )


def sCAM_to_XYZ(
    specification: CAM_Specification_sCAM,
    XYZ_w: ArrayLike,  # Absolute XYZ of target whitepoint
    L_A: ArrayLike,  # Adapting luminance (cd/m^2)
    Y_b: ArrayLike,  # Luminance factor of background
    surround: InductionFactors_sCAM | str = "Average",
    discount_illuminant: bool = False,
) -> NDArrayFloat:
    """Convert sCAM specification back to XYZ."""
    J_scam, C_scam_in, h_scam, _Q_scam, M_scam_in, _H_scam = astuple(specification)

    if has_only_nan(J_scam) or has_only_nan(h_scam):
        msg = "J (Lightness) and h (hue angle) must be provided."
        raise ValueError(msg)
    if has_only_nan(C_scam_in) and has_only_nan(M_scam_in):
        msg = "Either C (Chroma) or M (Colourfulness) must be provided."
        raise ValueError(msg)

    J_scam = to_domain_100(J_scam)
    h_scam = to_domain_degrees(h_scam)
    C_scam = to_domain_100(C_scam_in) if not has_only_nan(C_scam_in) else None
    M_scam = to_domain_100(M_scam_in) if not has_only_nan(M_scam_in) else None

    XYZ_w = to_domain_100(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    if isinstance(surround, str):
        surround_factors = VIEWING_CONDITIONS_sCAM[surround]
    else:
        surround_factors = surround

    F_s_param = surround_factors.F
    c_s_param = surround_factors.c
    Fm_s_param = surround_factors.Fm

    Y_w_val = XYZ_w[..., 1] if XYZ_w.ndim > 1 else XYZ_w[1]
    Y_w_val_safe = np.where(Y_w_val == 0, 1e-10, Y_w_val)
    n = Y_b / Y_w_val_safe
    z = 1.48 + np.sqrt(np.maximum(0, n))

    Lw_d65_ref_luminance_val = L_A * 100.0 / np.where(Y_b == 0, 1e-10, Y_b)
    Lw_d65_ref_luminance = np.where(
        Lw_d65_ref_luminance_val == 0, 1e-10, Lw_d65_ref_luminance_val
    )

    if C_scam is None and M_scam is not None:  # Calculate C_scam from M_scam
        FL_num = 0.1710 * spow(np.maximum(0, L_A), 1.0 / 3.0)
        FL_den_exp_term = np.exp(-0.9934 * L_A)
        FL_den = 1.0 - 0.4934 * FL_den_exp_term
        FL = FL_num / np.where(FL_den == 0, 1e-10, FL_den)

        et = 1.0 + 0.06 * np.cos(np.radians(110.0 + h_scam))
        J_scam_safe = np.where(J_scam == 0, 1e-10, J_scam)

        denominator_C_calc = spow(FL, 0.1) * et * Fm_s_param
        C_scam_calculated = np.zeros_like(M_scam)
        valid_denom = denominator_C_calc != 0
        C_scam_calculated[valid_denom] = (
            M_scam[valid_denom] * spow(J_scam_safe[valid_denom], 0.27)
        ) / denominator_C_calc[valid_denom]
        C_scam = np.maximum(0, C_scam_calculated)
    elif C_scam is None:
        msg = "Chroma (C) could not be determined."
        raise ValueError(msg)

    den_I_S_power = c_s_param * z
    I_S_ucs_power = 1.0 / np.where(den_I_S_power == 0, 1e-10, den_I_S_power)
    I_S_ucs = 100.0 * spow(np.clip(J_scam / 100.0, 0, None), I_S_ucs_power)
    I_S_ucs = np.clip(I_S_ucs, 0, None)

    A_S_ucs = C_scam * np.cos(np.radians(h_scam))
    B_S_ucs = C_scam * np.sin(np.radians(h_scam))

    sUCS_coords_for_inverse = tstack([I_S_ucs, A_S_ucs, B_S_ucs])

    XYZ_D65_normalized = sUCS_to_XYZ(sUCS_coords_for_inverse)
    XYZ_D65_abs = XYZ_D65_normalized * Lw_d65_ref_luminance

    # This is the D65 whitepoint at the reference luminance
    # used in the forward CAT
    XYZ_w_cat_source = SUCS_XYZ_D65_NORMALIZED * Lw_d65_ref_luminance

    XYZ_final = (  # Logic kept as original
        _chromatic_adaptation_sCAM(
            XYZ_D65_abs,
            XYZ_w_cat_source,  # Source white for XYZ_D65_abs
            XYZ_w,  # Target white for final output
            L_A,
            F_s_param,
            discount_illuminant,
        )
        / Lw_d65_ref_luminance
        * 100
    )

    return from_range_100(XYZ_final)
