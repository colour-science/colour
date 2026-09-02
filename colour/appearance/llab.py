"""
:math:`LLAB(l:c)` Colour Appearance Model
=========================================

Define the *:math:`LLAB(l:c)`* colour appearance model for predicting
perceptual colour attributes under varying viewing conditions.

-   :class:`colour.appearance.InductionFactors_LLAB`
-   :attr:`colour.VIEWING_CONDITIONS_LLAB`
-   :class:`colour.CAM_Specification_LLAB`
-   :func:`colour.XYZ_to_LLAB`

References
----------
-   :cite:`Fairchild2013x` : Fairchild, M. D. (2013). LLAB Model. In Color
    Appearance Models (3rd ed., pp. 6025-6178). Wiley. ISBN:B00DAYO8E2
-   :cite:`Luo1996b` : Luo, Ming Ronnier, Lo, M.-C., & Kuo, W.-G. (1996). The
    LLAB (l:c) colour model. Color Research & Application, 21(6), 412-429.
    doi:10.1002/(SICI)1520-6378(199612)21:6<412::AID-COL4>3.0.CO;2-Z
-   :cite:`Luo1996c` : Luo, Ming Ronnier, & Morovic, J. (1996). Two Unsolved
    Issues in Colour Management - Colour Appearance and Gamut Mapping.
    Conference: 5th International Conference on High Technology: Imaging
    Science and Technology - Evolution & Promise, 136-147.
    http://www.researchgate.net/publication/\
236348295_Two_Unsolved_Issues_in_Colour_Management__\
Colour_Appearance_and_Gamut_Mapping
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np

from colour.algebra import polar_to_cartesian, sdiv, sdiv_mode, spow, vecmul

if typing.TYPE_CHECKING:
    from colour.hints import Annotated, ArrayLike, Domain100, NDArrayFloat

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
    "InductionFactors_LLAB",
    "VIEWING_CONDITIONS_LLAB",
    "MATRIX_XYZ_TO_RGB_LLAB",
    "MATRIX_RGB_TO_XYZ_LLAB",
    "CAM_ReferenceSpecification_LLAB",
    "CAM_Specification_LLAB",
    "XYZ_to_LLAB",
]


@dataclass(frozen=True)
class InductionFactors_LLAB(MixinDataclassIterable):
    """
    Define the *:math:`LLAB(l:c)`* colour appearance model induction factors.

    Parameters
    ----------
    D
         *Discounting-the-Illuminant* factor :math:`D`.
    F_S
        Surround induction factor :math:`F_S`.
    F_L
        *Lightness* induction factor :math:`F_L`.
    F_C
        *Chroma* induction factor :math:`F_C`.

    References
    ----------
    :cite:`Fairchild2013x`, :cite:`Luo1996b`, :cite:`Luo1996c`
    """

    D: float
    F_S: float
    F_L: float
    F_C: float


VIEWING_CONDITIONS_LLAB: CanonicalMapping = CanonicalMapping(
    {
        "Reference Samples & Images, Average Surround, Subtending > 4": (
            InductionFactors_LLAB(1, 3, 0, 1)
        ),
        "Reference Samples & Images, Average Surround, Subtending < 4": (
            InductionFactors_LLAB(1, 3, 1, 1)
        ),
        "Television & VDU Displays, Dim Surround": (
            InductionFactors_LLAB(0.7, 3.5, 1, 1)
        ),
        "Cut Sheet Transparency, Dim Surround": (InductionFactors_LLAB(1, 5, 1, 1.1)),
        "35mm Projection Transparency, Dark Surround": (
            InductionFactors_LLAB(0.7, 4, 1, 1)
        ),
    }
)
VIEWING_CONDITIONS_LLAB.__doc__ = """
Define the reference :math:`LLAB(l:c)` colour appearance model viewing
conditions.

References
----------
:cite:`Fairchild2013x`, :cite:`Luo1996b`, :cite:`Luo1996c`

Aliases:

-   'ref_average_4_plus':
    'Reference Samples & Images, Average Surround, Subtending > 4'
-   'ref_average_4_minus':
    'Reference Samples & Images, Average Surround, Subtending < 4'
-   'tv_dim': 'Television & VDU Displays, Dim Surround'
-   'sheet_dim': 'Cut Sheet Transparency, Dim Surround'
-   'projected_dark': '35mm Projection Transparency, Dark Surround'
"""
VIEWING_CONDITIONS_LLAB["ref_average_4_plus"] = VIEWING_CONDITIONS_LLAB[
    "Reference Samples & Images, Average Surround, Subtending > 4"
]
VIEWING_CONDITIONS_LLAB["ref_average_4_minus"] = VIEWING_CONDITIONS_LLAB[
    "Reference Samples & Images, Average Surround, Subtending < 4"
]
VIEWING_CONDITIONS_LLAB["tv_dim"] = VIEWING_CONDITIONS_LLAB[
    "Television & VDU Displays, Dim Surround"
]
VIEWING_CONDITIONS_LLAB["sheet_dim"] = VIEWING_CONDITIONS_LLAB[
    "Cut Sheet Transparency, Dim Surround"
]
VIEWING_CONDITIONS_LLAB["projected_dark"] = VIEWING_CONDITIONS_LLAB[
    "35mm Projection Transparency, Dark Surround"
]

MATRIX_XYZ_TO_RGB_LLAB: NDArrayFloat = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)
"""
LLAB(l:c) colour appearance model *CIE XYZ* tristimulus values to normalised
cone responses matrix.
"""

MATRIX_RGB_TO_XYZ_LLAB: NDArrayFloat = np.linalg.inv(MATRIX_XYZ_TO_RGB_LLAB)
"""
LLAB(l:c) colour appearance model normalised cone responses to *CIE XYZ*
tristimulus values matrix.
"""


@dataclass
class CAM_ReferenceSpecification_LLAB(MixinDataclassArithmetic):
    """
    Define the *:math:`LLAB(l:c)`* colour appearance model reference
    specification.

    This specification contains field names consistent with the *Fairchild
    (2013)* reference.

    Parameters
    ----------
    L_L
        Correlate of *Lightness* :math:`L_L`.
    Ch_L
        Correlate of *chroma* :math:`Ch_L`.
    h_L
        *Hue* angle :math:`h_L` in degrees.
    s_L
        Correlate of *saturation* :math:`s_L`.
    C_L
        Correlate of *colourfulness* :math:`C_L`.
    HC
        *Hue* :math:`h` composition :math:`H^C`.
    A_L
        Opponent signal :math:`A_L`.
    B_L
        Opponent signal :math:`B_L`.

    References
    ----------
    :cite:`Fairchild2013x`, :cite:`Luo1996b`, :cite:`Luo1996c`
    """

    L_L: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Ch_L: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h_L: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s_L: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C_L: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)
    A_L: float | NDArrayFloat | None = field(default_factory=lambda: None)
    B_L: float | NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class CAM_Specification_LLAB(MixinDataclassArithmetic):
    """
    Define the *:math:`LLAB(l:c)`* colour appearance model specification.

    This specification provides a standardized interface for the *LLAB(l:c)*
    model with field names consistent across all colour appearance models in
    :mod:`colour.appearance`. While the field names differ from the original
    *Fairchild (2013)* reference notation, they map directly to the model's
    perceptual correlates.

    Parameters
    ----------
    J
        Correlate of *lightness* :math:`L_L`.
    C
        Correlate of *chroma* :math:`Ch_L`.
    h
        *Hue* angle :math:`h_L` in degrees.
    s
        Correlate of *saturation* :math:`s_L`.
    M
        Correlate of *colourfulness* :math:`C_L`.
    HC
        *Hue* :math:`h` composition :math:`H^C`.
    a
        Opponent signal :math:`A_L`.
    b
        Opponent signal :math:`B_L`.

    Notes
    -----
    -   This specification is the one used in the current model implementation.

    References
    ----------
    :cite:`Fairchild2013x`, :cite:`Luo1996b`, :cite:`Luo1996c`
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)
    a: float | NDArrayFloat | None = field(default_factory=lambda: None)
    b: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_LLAB(
    XYZ: Domain100,
    XYZ_0: Domain100,
    Y_b: ArrayLike,
    L: ArrayLike,
    surround: InductionFactors_LLAB = VIEWING_CONDITIONS_LLAB[
        "Reference Samples & Images, Average Surround, Subtending < 4"
    ],
) -> Annotated[CAM_Specification_LLAB, 360]:
    """
    Compute the *:math:`LLAB(l:c)`* colour appearance model correlates from
    the specified *CIE XYZ* tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_0
        *CIE XYZ* tristimulus values of reference white.
    Y_b
        Luminance factor of the background in :math:`cd/m^2`.
    L
        Absolute luminance :math:`L` of reference white in
        :math:`cd/m^2`.
    surround
        Surround viewing conditions induction factors.

    Returns
    -------
    :class:`colour.CAM_Specification_LLAB`
        *:math:`LLAB(l:c)`* colour appearance model specification.

    Notes
    -----
    +---------------------+-----------------------+---------------+
    | **Domain**          | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``XYZ``             | 100                   | 1             |
    +---------------------+-----------------------+---------------+
    | ``XYZ_0``           | 100                   | 1             |
    +---------------------+-----------------------+---------------+

    +---------------------+-----------------------+---------------+
    | **Range**           | **Scale - Reference** | **Scale - 1** |
    +=====================+=======================+===============+
    | ``specification.h`` | 360                   | 1             |
    +---------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2013x`, :cite:`Luo1996b`, :cite:`Luo1996c`

    Examples
    --------
    *Fairchild (2013)* Table 14.3 Case 4 (chromatic stimulus under illuminant
    A reference white, mesopic luminance):

    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_0 = np.array([109.85, 100.00, 35.58])
    >>> Y_b = 20.0
    >>> L = 31.83
    >>> surround = VIEWING_CONDITIONS_LLAB["ref_average_4_minus"]
    >>> XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)  # doctest: +ELLIPSIS
    CAM_Specification_LLAB(J=np.float64(39.81475...), C=np.float64(29.345046...), \
h=np.float64(271.852666...), s=np.float64(0.737039...), M=np.float64(54.593098...), \
HC=None, a=np.float64(1.764967...), b=np.float64(-54.564560...))
    """

    XYZ = to_domain_100(XYZ)
    XYZ_0 = to_domain_100(XYZ_0)

    xp = array_namespace(XYZ, XYZ_0, Y_b, L)

    _X, Y, _Z = tsplit(XYZ)
    Y_b = xp_as_float_array(Y_b, xp=xp, like=XYZ)
    L = xp_as_float_array(L, xp=xp, like=XYZ)
    F_S = xp_as_float_array(surround.F_S, xp=xp, like=XYZ)
    F_L = xp_as_float_array(surround.F_L, xp=xp, like=XYZ)
    F_C = xp_as_float_array(surround.F_C, xp=xp, like=XYZ)
    D = xp_as_float_array(surround.D, xp=xp, like=XYZ)

    # Computing normalised cone responses for the stimulus, the reference
    # white and the *CIE Standard Illuminant D Series* *D65* reference.
    with sdiv_mode():
        RGB = vecmul(MATRIX_XYZ_TO_RGB_LLAB, sdiv(XYZ, XYZ[..., 1, None]))
        RGB_0 = vecmul(MATRIX_XYZ_TO_RGB_LLAB, sdiv(XYZ_0, XYZ_0[..., 1, None]))
    XYZ_0r = xp_as_float_array([95.05, 100.00, 108.88], xp=xp, like=XYZ)
    RGB_0r = vecmul(MATRIX_XYZ_TO_RGB_LLAB, XYZ_0r / 100)

    # Computing chromatic adaptation: cone responses are adapted to the D65
    # reference; the blue channel uses a nonlinear power exponent.
    R, G, B = tsplit(RGB)
    R_0, G_0, B_0 = tsplit(RGB_0)
    R_0r, G_0r, B_0r = tsplit(RGB_0r)
    beta = spow(B_0 / B_0r, 0.0834)
    R_a = (D * R_0r / R_0 + 1 - D) * R
    G_a = (D * G_0r / G_0 + 1 - D) * G
    B_a = (D * B_0r / spow(B_0, beta) + 1 - D) * spow(B, beta)
    Y_stack = tstack([Y, Y, Y])
    XYZ_r = vecmul(MATRIX_RGB_TO_XYZ_LLAB, tstack([R_a, G_a, B_a]) * Y_stack)
    X_r, Y_r, Z_r = tsplit(XYZ_r)

    # Computing the nonlinear visual response :math:`f` on the three normalised
    # tristimulus components in a single call.
    one_F_s = 1 / F_S
    XYZ_n = tstack([X_r / 95.05, Y_r / 100, Z_r / 108.88])
    f_XYZ = xp.where(
        XYZ_n > 0.008856,
        spow(XYZ_n, one_F_s),
        ((spow(0.008856, one_F_s) - (16 / 116)) / 0.008856) * XYZ_n + (16 / 116),
    )
    f_X, f_Y, f_Z = tsplit(f_XYZ)

    # Computing opponent colour dimensions: modified *CIE L\\*a\\*b\\** with
    # background lightness contrast :math:`z`.
    z = 1 + F_L * spow(Y_b / 100, 0.5)
    L_L = as_float_array(116 * spow(f_Y, z) - 16)
    a = 500 * (f_X - f_Y)
    b = 200 * (f_Y - f_Z)

    # Computing the correlate of *chroma* :math:`Ch_L`.
    c = spow(a**2 + b**2, 0.5)
    Ch_L = 25 * xp.log1p(0.05 * c)

    # Computing the correlate of *colourfulness* :math:`C_L`.
    S_C = 1 + 0.47 * xp.log10(L) - 0.057 * xp.log10(L) ** 2
    S_M = 0.7 + 0.02 * L_L - 0.0002 * L_L**2
    C_L = Ch_L * S_M * S_C * F_C

    # Computing the correlate of *saturation* :math:`s_L`.
    s_L = Ch_L / L_L

    # Computing the *hue* angle :math:`h_L` in degrees.
    h_L = xp_degrees(xp.atan2(b, a)) % 360
    # TODO: Implement hue composition computation.

    # Computing the final opponent signals :math:`A_L`, :math:`B_L` from
    # polar coordinates :math:`(C_L, h_L)`.
    A_L, B_L = tsplit(polar_to_cartesian(tstack([C_L, xp_radians(h_L)])))

    return CAM_Specification_LLAB(
        J=as_float(L_L),
        C=as_float(Ch_L),
        h=as_float(from_range_degrees(h_L)),
        s=as_float(s_L),
        M=as_float(C_L),
        HC=None,
        a=as_float(A_L),
        b=as_float(B_L),
    )
