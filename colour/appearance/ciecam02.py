"""
CIECAM02 Colour Appearance Model
================================

Defines the *CIECAM02* colour appearance model objects:

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

import numpy as np
from collections import namedtuple
from dataclasses import astuple, dataclass, field

from colour.algebra import matrix_dot, sdiv, sdiv_mode, spow, vector_dot
from colour.adaptation import CAT_CAT02
from colour.appearance.hunt import (
    MATRIX_HPE_TO_XYZ,
    MATRIX_XYZ_TO_HPE,
    luminance_level_adaptation_factor,
)
from colour.colorimetry import CCS_ILLUMINANTS
from colour.constants import EPSILON
from colour.hints import ArrayLike, NDArrayFloat, Tuple, cast
from colour.models import xy_to_XYZ
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArithmetic,
    as_float,
    as_float_array,
    as_int_array,
    from_range_degrees,
    from_range_100,
    has_only_nan,
    ones,
    to_domain_100,
    to_domain_degrees,
    tsplit,
    tstack,
    zeros,
)
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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
    "chromatic_induction_factors",
    "base_exponential_non_linearity",
    "viewing_conditions_dependent_parameters",
    "degree_of_adaptation",
    "full_chromatic_adaptation_forward",
    "full_chromatic_adaptation_inverse",
    "RGB_to_rgb",
    "rgb_to_RGB",
    "post_adaptation_non_linear_response_compression_forward",
    "post_adaptation_non_linear_response_compression_inverse",
    "opponent_colour_dimensions_forward",
    "opponent_colour_dimensions_inverse",
    "hue_angle",
    "hue_quadrature",
    "eccentricity_factor",
    "achromatic_response_forward",
    "achromatic_response_inverse",
    "lightness_correlate",
    "brightness_correlate",
    "temporary_magnitude_quantity_forward",
    "temporary_magnitude_quantity_inverse",
    "chroma_correlate",
    "colourfulness_correlate",
    "saturation_correlate",
    "P",
    "matrix_post_adaptation_non_linear_response_compression",
]

CAT_INVERSE_CAT02: NDArrayFloat = np.linalg.inv(CAT_CAT02)
"""Inverse CAT02 chromatic adaptation transform."""


class InductionFactors_CIECAM02(
    namedtuple("InductionFactors_CIECAM02", ("F", "c", "N_c"))
):
    """
    *CIECAM02* colour appearance model induction factors.

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


VIEWING_CONDITIONS_CIECAM02: CanonicalMapping = CanonicalMapping(
    {
        "Average": InductionFactors_CIECAM02(1, 0.69, 1),
        "Dim": InductionFactors_CIECAM02(0.9, 0.59, 0.9),
        "Dark": InductionFactors_CIECAM02(0.8, 0.525, 0.8),
    }
)
VIEWING_CONDITIONS_CIECAM02.__doc__ = """
Reference *CIECAM02* colour appearance model viewing conditions.

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
    "XYZ_w": xy_to_XYZ(
        CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    )
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
    XYZ: ArrayLike,
    XYZ_w: ArrayLike,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: InductionFactors_CIECAM02 = VIEWING_CONDITIONS_CIECAM02[
        "Average"
    ],
    discount_illuminant: bool = False,
) -> CAM_Specification_CIECAM02:
    """
    Compute the *CIECAM02* colour appearance model correlates from given
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

    Returns
    -------
    :class:`colour.CAM_Specification_CIECAM02`
        *CIECAM02* colour appearance model specification.

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
    | ``CAM_Specification_CIECAM02.J`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.C`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.h`` | [0, 360]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.s`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.Q`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.M`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.H`` | [0, 400]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+

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
    >>> XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)  # doctest: +ELLIPSIS
    CAM_Specification_CIECAM02(J=41.7310911..., C=0.1047077..., \
h=219.0484326..., s=2.3603053..., Q=195.3713259..., M=0.1088421..., \
H=278.0607358..., HC=None)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    n, F_L, N_bb, N_cb, z = viewing_conditions_dependent_parameters(
        Y_b, Y_w, L_A
    )

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB = vector_dot(CAT_CAT02, XYZ)
    RGB_w = vector_dot(CAT_CAT02, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (
        degree_of_adaptation(surround.F, L_A)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    # Computing full chromatic adaptation.
    RGB_c = full_chromatic_adaptation_forward(RGB, RGB_w, Y_w, D)
    RGB_wc = full_chromatic_adaptation_forward(RGB_w, RGB_w, Y_w, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_p = RGB_to_rgb(RGB_c)
    RGB_pw = RGB_to_rgb(RGB_wc)

    # Applying forward post-adaptation non-linear response compression.
    RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_p, F_L)
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_pw, F_L
    )

    # Converting to preliminary cartesian coordinates.
    a, b = tsplit(opponent_colour_dimensions_forward(RGB_a))

    # Computing the *hue* angle :math:`h`.
    h = hue_angle(a, b)

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h)
    # TODO: Compute hue composition.

    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing achromatic responses for the stimulus and the whitepoint.
    A = achromatic_response_forward(RGB_a, N_bb)
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    # Computing the correlate of *Lightness* :math:`J`.
    J = lightness_correlate(A, A_w, surround.c, z)

    # Computing the correlate of *brightness* :math:`Q`.
    Q = brightness_correlate(surround.c, J, A_w, F_L)

    # Computing the correlate of *chroma* :math:`C`.
    C = chroma_correlate(J, n, surround.N_c, N_cb, e_t, a, b, RGB_a)

    # Computing the correlate of *colourfulness* :math:`M`.
    M = colourfulness_correlate(C, F_L)

    # Computing the correlate of *saturation* :math:`s`.
    s = saturation_correlate(M, Q)

    return CAM_Specification_CIECAM02(
        as_float(from_range_100(J)),
        as_float(from_range_100(C)),
        as_float(from_range_degrees(h)),
        as_float(from_range_100(s)),
        as_float(from_range_100(Q)),
        as_float(from_range_100(M)),
        as_float(from_range_degrees(H, 400)),
        None,
    )


def CIECAM02_to_XYZ(
    specification: CAM_Specification_CIECAM02,
    XYZ_w: ArrayLike,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: InductionFactors_CIECAM02 = VIEWING_CONDITIONS_CIECAM02[
        "Average"
    ],
    discount_illuminant: bool = False,
) -> NDArrayFloat:
    """
    Convert from *CIECAM02* specification to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    specification
        *CIECAM02* colour appearance model specification. Correlate of
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
    | ``CAM_Specification_CIECAM02.J`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.C`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.h`` | [0, 360]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.s`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.Q`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.M`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_CIECAM02.H`` | [0, 360]              \
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

    n, F_L, N_bb, N_cb, z = viewing_conditions_dependent_parameters(
        Y_b, Y_w, L_A
    )

    if has_only_nan(C) and not has_only_nan(M):
        C = M / spow(F_L, 0.25)
    elif has_only_nan(C):
        raise ValueError(
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_CIECAM02" argument!'
        )

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB_w = vector_dot(CAT_CAT02, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (
        degree_of_adaptation(surround.F, L_A)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    # Computing full chromatic adaptation.
    RGB_wc = full_chromatic_adaptation_forward(RGB_w, RGB_w, Y_w, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_pw = RGB_to_rgb(RGB_wc)

    # Applying post-adaptation non-linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_pw, F_L
    )

    # Computing achromatic response for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    # Computing temporary magnitude quantity :math:`t`.
    t = temporary_magnitude_quantity_inverse(C, J, n)

    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing achromatic response :math:`A` for the stimulus.
    A = achromatic_response_inverse(A_w, J, surround.c, z)

    # Computing *P_1* to *P_3*.
    P_n = P(surround.N_c, N_cb, e_t, t, A, N_bb)
    _P_1, P_2, _P_3 = tsplit(P_n)

    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    ab = opponent_colour_dimensions_inverse(P_n, h)
    a, b = tsplit(ab) * np.where(t == 0, 0, 1)

    # Applying post-adaptation non-linear response compression matrix.
    RGB_a = matrix_post_adaptation_non_linear_response_compression(P_2, a, b)

    # Applying inverse post-adaptation non-linear response compression.
    RGB_p = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_c = rgb_to_RGB(RGB_p)

    # Applying inverse full chromatic adaptation.
    RGB = full_chromatic_adaptation_inverse(RGB_c, RGB_w, Y_w, D)

    # Converting *CMCCAT2000* transform sharpened *RGB* values to *CIE XYZ*
    # tristimulus values.
    XYZ = vector_dot(CAT_INVERSE_CAT02, RGB)

    return from_range_100(XYZ)


def chromatic_induction_factors(n: ArrayLike) -> NDArrayFloat:
    """
    Return the chromatic induction factors :math:`N_{bb}` and :math:`N_{cb}`.

    Parameters
    ----------
    n
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    :class:`numpy.ndarray`
        Chromatic induction factors :math:`N_{bb}` and :math:`N_{cb}`.

    Examples
    --------
    >>> chromatic_induction_factors(0.2)  # doctest: +ELLIPSIS
    array([ 1.000304,  1.000304])
    """

    n = as_float_array(n)

    with sdiv_mode():
        N_bb = N_cb = 0.725 * spow(sdiv(1, n), 0.2)

    N_bbcb = tstack([N_bb, N_cb])

    return N_bbcb


def base_exponential_non_linearity(
    n: ArrayLike,
) -> NDArrayFloat:
    """
    Return the base exponential non-linearity :math:`n`.

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
    1.9272135...
    """

    n = as_float_array(n)

    z = 1.48 + np.sqrt(n)

    return z


def viewing_conditions_dependent_parameters(
    Y_b: ArrayLike,
    Y_w: ArrayLike,
    L_A: ArrayLike,
) -> Tuple[
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
]:
    """
    Return the viewing condition dependent parameters.

    Parameters
    ----------
    Y_b
        Adapting field *Y* tristimulus value :math:`Y_b`.
    Y_w
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    :class:`tuple`
        Viewing condition dependent parameters.

    Examples
    --------
    >>> viewing_conditions_dependent_parameters(20.0, 100.0, 318.31)
    ... # doctest: +ELLIPSIS
    (0.2000000..., 1.1675444..., 1.0003040..., 1.0003040..., 1.9272135...)
    """

    Y_b = as_float_array(Y_b)
    Y_w = as_float_array(Y_w)

    with sdiv_mode():
        n = sdiv(Y_b, Y_w)

    F_L = luminance_level_adaptation_factor(L_A)
    N_bb, N_cb = tsplit(chromatic_induction_factors(n))
    z = base_exponential_non_linearity(n)

    return n, F_L, N_bb, N_cb, z


def degree_of_adaptation(F: ArrayLike, L_A: ArrayLike) -> NDArrayFloat:
    """
    Return the degree of adaptation :math:`D` from given surround maximum
    degree of adaptation :math:`F` and adapting field *luminance* :math:`L_A`
    in :math:`cd/m^2`.

    Parameters
    ----------
    F
        Surround maximum degree of adaptation :math:`F`.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.

    Returns
    -------
    :class:`numpy.ndarray`
        Degree of adaptation :math:`D`.

    Examples
    --------
    >>> degree_of_adaptation(1.0, 318.31)  # doctest: +ELLIPSIS
    0.9944687...
    """

    F = as_float_array(F)
    L_A = as_float_array(L_A)

    D = F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92))

    return D


def full_chromatic_adaptation_forward(
    RGB: ArrayLike,
    RGB_w: ArrayLike,
    Y_w: ArrayLike,
    D: ArrayLike,
) -> NDArrayFloat:
    """
    Apply full chromatic adaptation to given *CMCCAT2000* transform sharpened
    *RGB* array using given *CMCCAT2000* transform sharpened whitepoint
    *RGB_w* array.

    Parameters
    ----------
    RGB
        *CMCCAT2000* transform sharpened *RGB* array.
    RGB_w
        *CMCCAT2000* transform sharpened whitepoint *RGB_w* array.
    Y_w
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    D
        Degree of adaptation :math:`D`.

    Returns
    -------
    :class:`numpy.ndarray`
        Adapted *RGB* array.

    Examples
    --------
    >>> RGB = np.array([18.985456, 20.707422, 21.747482])
    >>> RGB_w = np.array([94.930528, 103.536988, 108.717742])
    >>> Y_w = 100.0
    >>> D = 0.994468780088
    >>> full_chromatic_adaptation_forward(RGB, RGB_w, Y_w, D)
    ... # doctest: +ELLIPSIS
    array([ 19.9937078...,  20.0039363...,  20.0132638...])
    """

    RGB = as_float_array(RGB)
    RGB_w = as_float_array(RGB_w)
    Y_w = as_float_array(Y_w)
    D = as_float_array(D)

    with sdiv_mode():
        RGB_c = (
            Y_w[..., None] * sdiv(D[..., None], RGB_w) + 1 - D[..., None]
        ) * RGB

    return RGB_c


def full_chromatic_adaptation_inverse(
    RGB: ArrayLike,
    RGB_w: ArrayLike,
    Y_w: ArrayLike,
    D: ArrayLike,
) -> NDArrayFloat:
    """
    Revert full chromatic adaptation of given *CMCCAT2000* transform sharpened
    *RGB* array using given *CMCCAT2000* transform sharpened whitepoint
    *RGB_w* array.

    Parameters
    ----------
    RGB
        *CMCCAT2000* transform sharpened *RGB* array.
    RGB_w
        *CMCCAT2000* transform sharpened whitepoint *RGB_w* array.
    Y_w
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    D
        Degree of adaptation :math:`D`.

    Returns
    -------
    :class:`numpy.ndarray`
        Adapted *RGB* array.

    Examples
    --------
    >>> RGB = np.array([19.99370783, 20.00393634, 20.01326387])
    >>> RGB_w = np.array([94.930528, 103.536988, 108.717742])
    >>> Y_w = 100.0
    >>> D = 0.994468780088
    >>> full_chromatic_adaptation_inverse(RGB, RGB_w, Y_w, D)
    array([ 18.985456,  20.707422,  21.747482])
    """

    RGB = as_float_array(RGB)
    RGB_w = as_float_array(RGB_w)
    Y_w = as_float_array(Y_w)
    D = as_float_array(D)

    with sdiv_mode():
        RGB_c = RGB / (
            Y_w[..., None] * sdiv(D[..., None], RGB_w) + 1 - D[..., None]
        )

    return cast(NDArrayFloat, RGB_c)


def RGB_to_rgb(RGB: ArrayLike) -> NDArrayFloat:
    """
    Convert given *RGB* array to *Hunt-Pointer-Estevez*
    :math:`\\rho\\gamma\\beta` colourspace.

    Parameters
    ----------
    RGB
        *RGB* array.

    Returns
    -------
    :class:`numpy.ndarray`
        *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace array.

    Examples
    --------
    >>> RGB = np.array([19.99370783, 20.00393634, 20.01326387])
    >>> RGB_to_rgb(RGB)  # doctest: +ELLIPSIS
    array([ 19.9969397...,  20.0018612...,  20.0135053...])
    """

    rgb = vector_dot(matrix_dot(MATRIX_XYZ_TO_HPE, CAT_INVERSE_CAT02), RGB)

    return rgb


def rgb_to_RGB(rgb: ArrayLike) -> NDArrayFloat:
    """
    Convert given *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
    colourspace array to *RGB* array.

    Parameters
    ----------
    rgb
        *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* array.

    Examples
    --------
    >>> rgb = np.array([19.99693975, 20.00186123, 20.01350530])
    >>> rgb_to_RGB(rgb)  # doctest: +ELLIPSIS
    array([ 19.9937078...,  20.0039363...,  20.0132638...])
    """

    RGB = vector_dot(matrix_dot(CAT_CAT02, MATRIX_HPE_TO_XYZ), rgb)

    return RGB


def post_adaptation_non_linear_response_compression_forward(
    RGB: ArrayLike, F_L: ArrayLike
) -> NDArrayFloat:
    """
    Return given *CMCCAT2000* transform sharpened *RGB* array with post
    adaptation non-linear response compression.

    Parameters
    ----------
    RGB
        *CMCCAT2000* transform sharpened *RGB* array.
    F_L
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    :class:`numpy.ndarray`
        Compressed *CMCCAT2000* transform sharpened *RGB* array.

    Notes
    -----
    -   This definition implements negative values handling as per
        :cite:`Luo2013`.

    Examples
    --------
    >>> RGB = np.array([19.99693975, 20.00186123, 20.01350530])
    >>> F_L = 1.16754446415
    >>> post_adaptation_non_linear_response_compression_forward(RGB, F_L)
    ... # doctest: +ELLIPSIS
    array([ 7.9463202...,  7.9471152...,  7.9489959...])
    """

    RGB = as_float_array(RGB)
    F_L = as_float_array(F_L)

    F_L_RGB = spow(F_L[..., None] * np.absolute(RGB) / 100, 0.42)
    RGB_c = (400 * np.sign(RGB) * F_L_RGB) / (27.13 + F_L_RGB) + 0.1

    return RGB_c


def post_adaptation_non_linear_response_compression_inverse(
    RGB: ArrayLike, F_L: ArrayLike
) -> NDArrayFloat:
    """
    Return given *CMCCAT2000* transform sharpened *RGB* array without post
    adaptation non-linear response compression.

    Parameters
    ----------
    RGB
        *CMCCAT2000* transform sharpened *RGB* array.
    F_L
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    :class:`numpy.ndarray`
        Uncompressed *CMCCAT2000* transform sharpened *RGB* array.

    Examples
    --------
    >>> RGB = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> F_L = 1.16754446415
    >>> post_adaptation_non_linear_response_compression_inverse(RGB, F_L)
    ... # doctest: +ELLIPSIS
    array([ 19.9969397...,  20.0018612...,  20.0135052...])
    """

    RGB = as_float_array(RGB)
    F_L = as_float_array(F_L)

    RGB_p = (
        np.sign(RGB - 0.1)
        * 100
        / F_L[..., None]
        * spow(
            (27.13 * np.absolute(RGB - 0.1)) / (400 - np.absolute(RGB - 0.1)),
            1 / 0.42,
        )
    )

    return RGB_p


def opponent_colour_dimensions_forward(RGB: ArrayLike) -> NDArrayFloat:
    """
    Return opponent colour dimensions from given compressed *CMCCAT2000*
    transform sharpened *RGB* array for forward *CIECAM02* implementation.

    Parameters
    ----------
    RGB
        Compressed *CMCCAT2000* transform sharpened *RGB* array.

    Returns
    -------
    :class:`numpy.ndarray`
        Opponent colour dimensions.

    Examples
    --------
    >>> RGB = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> opponent_colour_dimensions_forward(RGB)  # doctest: +ELLIPSIS
    array([-0.0006241..., -0.0005062...])
    """

    R, G, B = tsplit(RGB)

    a = R - 12 * G / 11 + B / 11
    b = (R + G - 2 * B) / 9

    ab = tstack([a, b])

    return ab


def opponent_colour_dimensions_inverse(
    P_n: ArrayLike, h: ArrayLike
) -> NDArrayFloat:
    """
    Return opponent colour dimensions from given points :math:`P_n` and hue
    :math:`h` in degrees for inverse *CIECAM02* implementation.

    Parameters
    ----------
    P_n
        Points :math:`P_n`.
    h
        Hue :math:`h` in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Opponent colour dimensions.

    Examples
    --------
    >>> P_n = np.array([30162.89081534, 24.23720547, 1.05000000])
    >>> h = -140.95156734
    >>> opponent_colour_dimensions_inverse(P_n, h)  # doctest: +ELLIPSIS
    array([-0.0006241..., -0.0005062...])
    """

    P_1, P_2, P_3 = tsplit(P_n)
    hr = np.radians(h)

    sin_hr = np.sin(hr)
    cos_hr = np.cos(hr)

    with sdiv_mode():
        cos_hr_sin_hr = sdiv(cos_hr, sin_hr)
        sin_hr_cos_hr = sdiv(sin_hr, cos_hr)

        P_4 = sdiv(P_1, sin_hr)
        P_5 = sdiv(P_1, cos_hr)

    n = P_2 * (2 + P_3) * (460 / 1403)

    a = zeros(hr.shape)
    b = zeros(hr.shape)

    abs_sin_hr_gt_cos_hr = np.abs(sin_hr) >= np.abs(cos_hr)
    abs_sin_hr_lt_cos_hr = np.abs(sin_hr) < np.abs(cos_hr)

    b = np.where(
        abs_sin_hr_gt_cos_hr,
        n
        / (
            P_4
            + (2 + P_3) * (220 / 1403) * cos_hr_sin_hr
            - (27 / 1403)
            + P_3 * (6300 / 1403)
        ),
        b,
    )

    a = np.where(
        abs_sin_hr_gt_cos_hr,
        b * cos_hr_sin_hr,
        a,
    )

    a = np.where(
        abs_sin_hr_lt_cos_hr,
        n
        / (
            P_5
            + (2 + P_3) * (220 / 1403)
            - ((27 / 1403) - P_3 * (6300 / 1403)) * sin_hr_cos_hr
        ),
        a,
    )

    b = np.where(
        abs_sin_hr_lt_cos_hr,
        a * sin_hr_cos_hr,
        b,
    )

    ab = tstack([a, b])

    return ab


def hue_angle(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Return the *hue* angle :math:`h` in degrees.

    Parameters
    ----------
    a
        Opponent colour dimension :math:`a`.
    b
        Opponent colour dimension :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Hue* angle :math:`h` in degrees.

    Examples
    --------
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> hue_angle(a, b)  # doctest: +ELLIPSIS
    219.0484326...
    """

    a = as_float_array(a)
    b = as_float_array(b)

    h = np.degrees(np.arctan2(b, a)) % 360

    return as_float(h)


def hue_quadrature(h: ArrayLike) -> NDArrayFloat:
    """
    Return the hue quadrature from given hue :math:`h` angle in degrees.

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
    278.0607358...
    """

    h = as_float_array(h)

    h_i = HUE_DATA_FOR_HUE_QUADRATURE["h_i"]
    e_i = HUE_DATA_FOR_HUE_QUADRATURE["e_i"]
    H_i = HUE_DATA_FOR_HUE_QUADRATURE["H_i"]

    # *np.searchsorted* returns an erroneous index if a *nan* is used as input.
    h[np.asarray(np.isnan(h))] = 0
    i = as_int_array(np.searchsorted(h_i, h, side="left") - 1)

    h_ii = h_i[i]
    e_ii = e_i[i]
    H_ii = H_i[i]
    h_ii1 = h_i[i + 1]
    e_ii1 = e_i[i + 1]

    H = H_ii + (
        (100 * (h - h_ii) / e_ii) / ((h - h_ii) / e_ii + (h_ii1 - h) / e_ii1)
    )

    H = np.where(
        h < 20.14,
        385.9 + (14.1 * h / 0.856) / (h / 0.856 + (20.14 - h) / 0.8),
        H,
    )
    H = np.where(
        h >= 237.53,
        H_ii
        + (
            (85.9 * (h - h_ii) / e_ii)
            / ((h - h_ii) / e_ii + (360 - h) / 0.856)
        ),
        H,
    )
    return as_float(H)


def eccentricity_factor(h: ArrayLike) -> NDArrayFloat:
    """
    Return the eccentricity factor :math:`e_t` from given hue :math:`h` angle
    in degrees for forward *CIECAM02* implementation.

    Parameters
    ----------
    h
        Hue :math:`h` angle in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Eccentricity factor :math:`e_t`.

    Examples
    --------
    >>> eccentricity_factor(-140.951567342)  # doctest: +ELLIPSIS
    1.1740054...
    """

    h = as_float_array(h)

    e_t = 1 / 4 * (np.cos(2 + h * np.pi / 180) + 3.8)

    return e_t


def achromatic_response_forward(
    RGB: ArrayLike, N_bb: ArrayLike
) -> NDArrayFloat:
    """
    Return the achromatic response :math:`A` from given compressed
    *CMCCAT2000* transform sharpened *RGB* array and :math:`N_{bb}` chromatic
    induction factor for forward *CIECAM02* implementation.

    Parameters
    ----------
    RGB
        Compressed *CMCCAT2000* transform sharpened *RGB* array.
    N_bb
        Chromatic induction factor :math:`N_{bb}`.

    Returns
    -------
    :class:`numpy.ndarray`
        Achromatic response :math:`A`.

    Examples
    --------
    >>> RGB = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> N_bb = 1.000304004559381
    >>> achromatic_response_forward(RGB, N_bb)  # doctest: +ELLIPSIS
    23.9394809...
    """

    R, G, B = tsplit(RGB)

    A = (2 * R + G + (1 / 20) * B - 0.305) * N_bb

    return A


def achromatic_response_inverse(
    A_w: ArrayLike,
    J: ArrayLike,
    c: ArrayLike,
    z: ArrayLike,
) -> NDArrayFloat:
    """
    Return the achromatic response :math:`A` from given achromatic response
    :math:`A_w` for the whitepoint, *Lightness* correlate :math:`J`, surround
    exponential non-linearity :math:`c` and base exponential non-linearity
    :math:`z` for inverse *CIECAM02* implementation.

    Parameters
    ----------
    A_w
        Achromatic response :math:`A_w` for the whitepoint.
    J
        *Lightness* correlate :math:`J`.
    c
        Surround exponential non-linearity :math:`c`.
    z
        Base exponential non-linearity :math:`z`.

    Returns
    -------
    :class:`numpy.ndarray`
        Achromatic response :math:`A`.

    Examples
    --------
    >>> A_w = 46.1882087914
    >>> J = 41.73109113251392
    >>> c = 0.69
    >>> z = 1.927213595499958
    >>> achromatic_response_inverse(A_w, J, c, z)  # doctest: +ELLIPSIS
    23.9394809...
    """

    A_w = as_float_array(A_w)
    J = as_float_array(J)
    c = as_float_array(c)
    z = as_float_array(z)

    A = A_w * spow(J / 100, 1 / (c * z))

    return A


def lightness_correlate(
    A: ArrayLike,
    A_w: ArrayLike,
    c: ArrayLike,
    z: ArrayLike,
) -> NDArrayFloat:
    """
    Return the *Lightness* correlate :math:`J`.

    Parameters
    ----------
    A
        Achromatic response :math:`A` for the stimulus.
    A_w
        Achromatic response :math:`A_w` for the whitepoint.
    c
        Surround exponential non-linearity :math:`c`.
    z
        Base exponential non-linearity :math:`z`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Lightness* correlate :math:`J`.

    Examples
    --------
    >>> A = 23.9394809667
    >>> A_w = 46.1882087914
    >>> c = 0.69
    >>> z = 1.9272135955
    >>> lightness_correlate(A, A_w, c, z)  # doctest: +ELLIPSIS
    41.7310911...
    """

    A = as_float_array(A)
    A_w = as_float_array(A_w)
    c = as_float_array(c)
    z = as_float_array(z)

    with sdiv_mode():
        J = 100 * spow(sdiv(A, A_w), c * z)

    return J


def brightness_correlate(
    c: ArrayLike,
    J: ArrayLike,
    A_w: ArrayLike,
    F_L: ArrayLike,
) -> NDArrayFloat:
    """
    Return the *brightness* correlate :math:`Q`.

    Parameters
    ----------
    c
        Surround exponential non-linearity :math:`c`.
    J
        *Lightness* correlate :math:`J`.
    A_w
        Achromatic response :math:`A_w` for the whitepoint.
    F_L
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Brightness* correlate :math:`Q`.

    Examples
    --------
    >>> c = 0.69
    >>> J = 41.7310911325
    >>> A_w = 46.1882087914
    >>> F_L = 1.16754446415
    >>> brightness_correlate(c, J, A_w, F_L)  # doctest: +ELLIPSIS
    195.3713259...
    """

    c = as_float_array(c)
    J = as_float_array(J)
    A_w = as_float_array(A_w)
    F_L = as_float_array(F_L)

    Q = (4 / c) * np.sqrt(J / 100) * (A_w + 4) * spow(F_L, 0.25)

    return Q


def temporary_magnitude_quantity_forward(
    N_c: ArrayLike,
    N_cb: ArrayLike,
    e_t: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    RGB_a: ArrayLike,
) -> NDArrayFloat:
    """
    Return the temporary magnitude quantity :math:`t`. for forward *CIECAM02*
    implementation.

    Parameters
    ----------
    N_c
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb
        Chromatic induction factor :math:`N_{cb}`.
    e_t
        Eccentricity factor :math:`e_t`.
    a
        Opponent colour dimension :math:`a`.
    b
        Opponent colour dimension :math:`b`.
    RGB_a
        Compressed stimulus *CMCCAT2000* transform sharpened *RGB* array.

    Returns
    -------
    :class:`numpy.ndarray`
         Temporary magnitude quantity :math:`t`.

    Examples
    --------
    >>> N_c = 1.0
    >>> N_cb = 1.00030400456
    >>> e_t = 1.174005472851914
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> RGB_a = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> temporary_magnitude_quantity_forward(N_c, N_cb, e_t, a, b, RGB_a)
    ... # doctest: +ELLIPSIS
    0.1497462...
    """

    N_c = as_float_array(N_c)
    N_cb = as_float_array(N_cb)
    e_t = as_float_array(e_t)
    a = as_float_array(a)
    b = as_float_array(b)
    Ra, Ga, Ba = tsplit(RGB_a)

    with sdiv_mode():
        t = ((50000 / 13) * N_c * N_cb) * sdiv(
            e_t * spow(a**2 + b**2, 0.5), Ra + Ga + 21 * Ba / 20
        )

    return t


def temporary_magnitude_quantity_inverse(
    C: ArrayLike, J: ArrayLike, n: ArrayLike
) -> NDArrayFloat:
    """
    Return the temporary magnitude quantity :math:`t`. for inverse *CIECAM02*
    implementation.

    Parameters
    ----------
    C
        *Chroma* correlate :math:`C`.
    J
        *Lightness* correlate :math:`J`.
    n
        Function of the luminance factor of the background :math:`n`.

    Returns
    -------
    :class:`numpy.ndarray`
         Temporary magnitude quantity :math:`t`.

    Examples
    --------
    >>> C = 68.8364136888275
    >>> J = 41.749268505999
    >>> n = 0.2
    >>> temporary_magnitude_quantity_inverse(C, J, n)  # doctest: +ELLIPSIS
    202.3873619...
    """

    C = as_float_array(C)
    J = np.maximum(J, EPSILON)
    n = as_float_array(n)

    t = spow(C / (np.sqrt(J / 100) * spow(1.64 - 0.29**n, 0.73)), 1 / 0.9)

    return t


def chroma_correlate(
    J: ArrayLike,
    n: ArrayLike,
    N_c: ArrayLike,
    N_cb: ArrayLike,
    e_t: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    RGB_a: ArrayLike,
) -> NDArrayFloat:
    """
    Return the *chroma* correlate :math:`C`.

    Parameters
    ----------
    J
        *Lightness* correlate :math:`J`.
    n
        Function of the luminance factor of the background :math:`n`.
    N_c
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb
        Chromatic induction factor :math:`N_{cb}`.
    e_t
        Eccentricity factor :math:`e_t`.
    a
        Opponent colour dimension :math:`a`.
    b
        Opponent colour dimension :math:`b`.
    RGB_a
        Compressed stimulus *CMCCAT2000* transform sharpened *RGB* array.

    Returns
    -------
    :class:`numpy.ndarray`
        *Chroma* correlate :math:`C`.

    Examples
    --------
    >>> J = 41.7310911325
    >>> n = 0.2
    >>> N_c = 1.0
    >>> N_cb = 1.00030400456
    >>> e_t = 1.17400547285
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> RGB_a = np.array([7.94632020, 7.94711528, 7.94899595])
    >>> chroma_correlate(J, n, N_c, N_cb, e_t, a, b, RGB_a)
    ... # doctest: +ELLIPSIS
    0.1047077...
    """

    J = as_float_array(J)
    n = as_float_array(n)

    t = temporary_magnitude_quantity_forward(N_c, N_cb, e_t, a, b, RGB_a)
    C = spow(t, 0.9) * spow(J / 100, 0.5) * spow(1.64 - 0.29**n, 0.73)

    return C


def colourfulness_correlate(C: ArrayLike, F_L: ArrayLike) -> NDArrayFloat:
    """
    Return the *colourfulness* correlate :math:`M`.

    Parameters
    ----------
    C
        *Chroma* correlate :math:`C`.
    F_L
        *Luminance* level adaptation factor :math:`F_L`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Colourfulness* correlate :math:`M`.

    Examples
    --------
    >>> C = 0.104707757171
    >>> F_L = 1.16754446415
    >>> colourfulness_correlate(C, F_L)  # doctest: +ELLIPSIS
    0.1088421...
    """

    C = as_float_array(C)
    F_L = as_float_array(F_L)

    M = C * spow(F_L, 0.25)

    return M


def saturation_correlate(M: ArrayLike, Q: ArrayLike) -> NDArrayFloat:
    """
    Return the *saturation* correlate :math:`s`.

    Parameters
    ----------
    M
        *Colourfulness* correlate :math:`M`.
    Q
        *Brightness* correlate :math:`C`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Saturation* correlate :math:`s`.

    Examples
    --------
    >>> M = 0.108842175669
    >>> Q = 195.371325966
    >>> saturation_correlate(M, Q)  # doctest: +ELLIPSIS
    2.3603053...
    """

    M = as_float_array(M)
    Q = as_float_array(Q)

    with sdiv_mode():
        s = 100 * spow(sdiv(M, Q), 0.5)

    return s


def P(
    N_c: ArrayLike,
    N_cb: ArrayLike,
    e_t: ArrayLike,
    t: ArrayLike,
    A: ArrayLike,
    N_bb: ArrayLike,
) -> NDArrayFloat:
    """
    Return the points :math:`P_1`, :math:`P_2` and :math:`P_3`.

    Parameters
    ----------
    N_c
        Surround chromatic induction factor :math:`N_{c}`.
    N_cb
        Chromatic induction factor :math:`N_{cb}`.
    e_t
        Eccentricity factor :math:`e_t`.
    t
        Temporary magnitude quantity :math:`t`.
    A
        Achromatic response  :math:`A` for the stimulus.
    N_bb
        Chromatic induction factor :math:`N_{bb}`.

    Returns
    -------
    :class:`numpy.ndarray`
        Points :math:`P`.

    Examples
    --------
    >>> N_c = 1.0
    >>> N_cb = 1.00030400456
    >>> e_t = 1.174005472851914
    >>> t = 0.149746202921
    >>> A = 23.9394809667
    >>> N_bb = 1.00030400456
    >>> P(N_c, N_cb, e_t, t, A, N_bb)  # doctest: +ELLIPSIS
    array([  3.0162890...e+04,   2.4237205...e+01,   1.0500000...e+00])
    """

    N_c = as_float_array(N_c)
    N_cb = as_float_array(N_cb)
    e_t = as_float_array(e_t)
    t = as_float_array(t)
    A = as_float_array(A)
    N_bb = as_float_array(N_bb)

    with sdiv_mode():
        P_1 = sdiv((50000 / 13) * N_c * N_cb * e_t, t)

    P_2 = A / N_bb + 0.305
    P_3 = ones(P_1.shape) * (21 / 20)

    P_n = tstack([P_1, P_2, P_3])

    return P_n


def matrix_post_adaptation_non_linear_response_compression(
    P_2: ArrayLike, a: ArrayLike, b: ArrayLike
) -> NDArrayFloat:
    """
    Apply the post-adaptation non-linear-response compression matrix.

    Parameters
    ----------
    P_2
        Point :math:`P_2`.
    a
        Opponent colour dimension :math:`a`.
    b
        Opponent colour dimension :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        Points :math:`P`.

    Examples
    --------
    >>> P_2 = 24.2372054671
    >>> a = -0.000624112068243
    >>> b = -0.000506270106773
    >>> matrix_post_adaptation_non_linear_response_compression(P_2, a, b)
    ... # doctest: +ELLIPSIS
    array([ 7.9463202...,  7.9471152...,  7.9489959...])
    """

    P_2 = as_float_array(P_2)
    a = as_float_array(a)
    b = as_float_array(b)

    RGB_a = (
        vector_dot(
            [
                [460, 451, 288],
                [460, -891, -261],
                [460, -220, -6300],
            ],
            tstack([P_2, a, b]),
        )
        / 1403
    )

    return RGB_a
