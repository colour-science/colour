"""
Hellwig and Fairchild (2022) Colour Appearance Model
====================================================

Defines the *Hellwig and Fairchild (2022)* colour appearance model objects:

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

import numpy as np
from collections import namedtuple
from dataclasses import astuple, dataclass, field

from colour.algebra import sdiv, sdiv_mode, spow, vector_dot
from colour.appearance.cam16 import (
    MATRIX_16,
    MATRIX_INVERSE_16,
)
from colour.appearance.ciecam02 import (
    InductionFactors_CIECAM02,
    VIEWING_CONDITIONS_CIECAM02,
    achromatic_response_inverse,
    base_exponential_non_linearity,
    degree_of_adaptation,
    hue_angle,
    hue_quadrature,
    lightness_correlate,
    opponent_colour_dimensions_forward,
    post_adaptation_non_linear_response_compression_forward,
    post_adaptation_non_linear_response_compression_inverse,
    matrix_post_adaptation_non_linear_response_compression,
)
from colour.appearance.hunt import luminance_level_adaptation_factor
from colour.hints import (
    ArrayLike,
    Boolean,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
    Optional,
    Tuple,
    Union,
)
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
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "InductionFactors_Hellwig2022",
    "VIEWING_CONDITIONS_HELLWIG2022",
    "CAM_Specification_Hellwig2022",
    "XYZ_to_Hellwig2022",
    "Hellwig2022_to_XYZ",
    "viewing_conditions_dependent_parameters",
    "achromatic_response_forward",
    "opponent_colour_dimensions_inverse",
    "eccentricity_factor",
    "brightness_correlate",
    "colourfulness_correlate",
    "chroma_correlate",
    "saturation_correlate",
    "P_p",
    "hue_angle_dependency_Hellwig2022",
]


class InductionFactors_Hellwig2022(
    namedtuple("InductionFactors_Hellwig2022", ("F", "c", "N_c"))
):
    """
    *Hellwig and Fairchild (2022)* colour appearance model induction factors.

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


VIEWING_CONDITIONS_HELLWIG2022: CanonicalMapping = CanonicalMapping(
    VIEWING_CONDITIONS_CIECAM02
)
VIEWING_CONDITIONS_HELLWIG2022.__doc__ = """
Reference *Hellwig and Fairchild (2022)* colour appearance model viewing
conditions.

References
----------
:cite:`Hellwig2022`
"""


@dataclass
class CAM_Specification_Hellwig2022(MixinDataclassArithmetic):
    """
    Define the *Hellwig and Fairchild (2022)* colour appearance model
    specification.

    This specification supports the *Helmholtz-Kohlrausch* effect extension
    from :cite:`Hellwig2022a`.

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
    J_HK
        Correlate of *Lightness* :math:`J_{HK}` accounting for
        *Helmholtz-Kohlrausch* effect.
    Q_HK
        Correlate of *brightness* :math:`Q_{HK}` accounting for
        *Helmholtz-Kohlrausch* effect.

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`, :cite:`Hellwig2022a`
    """

    J: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    C: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    h: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    s: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    Q: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    M: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    H: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    HC: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    J_HK: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    Q_HK: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)


def XYZ_to_Hellwig2022(
    XYZ: ArrayLike,
    XYZ_w: ArrayLike,
    L_A: FloatingOrArrayLike,
    Y_b: FloatingOrArrayLike,
    surround: Union[
        InductionFactors_CIECAM02, InductionFactors_Hellwig2022
    ] = VIEWING_CONDITIONS_HELLWIG2022["Average"],
    discount_illuminant: Boolean = False,
) -> CAM_Specification_Hellwig2022:
    """
    Compute the *Hellwig and Fairchild (2022)* colour appearance model
    correlates from given *CIE XYZ* tristimulus values.

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
    :class:`colour.CAM_Specification_Hellwig2022`
        *Hellwig and Fairchild (2022)* colour appearance model specification.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_w``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +----------------------------------------+-----------------------\
+---------------+
    | **Range**                              | **Scale - Reference** \
| **Scale - 1** |
    +========================================+=======================\
+===============+
    | ``CAM_Specification_Hellwig2022.J``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.C``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.h``    | [0, 360]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.s``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.Q``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.M``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.H``    | [0, 400]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.J_HK`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.Q_HK`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`, :cite:`Hellwig2022a`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
    >>> XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround)
    ... # doctest: +ELLIPSIS
    CAM_Specification_Hellwig2022(J=41.7312079..., C=0.0257636..., \
h=217.0679597..., s=0.0608550..., Q=55.8523226..., M=0.0339889..., \
H=275.5949861..., HC=None, J_HK=41.8802782..., Q_HK=56.0518358...)
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

    F_L, z = viewing_conditions_dependent_parameters(Y_b, Y_w, L_A)

    D_RGB = D[..., None] * Y_w[..., None] / RGB_w + 1 - D[..., None]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_wc, F_L
    )

    # Computing achromatic responses for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw)

    # Step 1
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB = vector_dot(MATRIX_16, XYZ)

    # Step 2
    RGB_c = D_RGB * RGB

    # Step 3
    # Applying forward post-adaptation non-linear response compression.
    RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L)

    # Step 4
    # Converting to preliminary cartesian coordinates.
    a, b = tsplit(opponent_colour_dimensions_forward(RGB_a))

    # Computing the *hue* angle :math:`h`.
    h = hue_angle(a, b)

    # Step 5
    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h)
    # TODO: Compute hue composition.

    # Step 6
    # Computing achromatic responses for the stimulus.
    A = achromatic_response_forward(RGB_a)

    # Step 7
    # Computing the correlate of *Lightness* :math:`J`.
    J = lightness_correlate(A, A_w, surround.c, z)

    # Step 8
    # Computing the correlate of *brightness* :math:`Q`.
    Q = brightness_correlate(surround.c, J, A_w)

    # Step 9
    # Computing the correlate of *colourfulness* :math:`M`.
    M = colourfulness_correlate(surround.N_c, e_t, a, b)

    # Computing the correlate of *chroma* :math:`C`.
    C = chroma_correlate(M, A_w)

    # Computing the correlate of *saturation* :math:`s`.
    s = saturation_correlate(M, Q)

    # *Helmholtz-Kohlrausch* Effect Extension.
    J_HK = J + hue_angle_dependency_Hellwig2022(h) * spow(C, 0.587)
    Q_HK = (2 / surround.c) * (J_HK / 100) * A_w

    return CAM_Specification_Hellwig2022(
        as_float(from_range_100(J)),
        as_float(from_range_100(C)),
        as_float(from_range_degrees(h)),
        as_float(from_range_100(s)),
        as_float(from_range_100(Q)),
        as_float(from_range_100(M)),
        as_float(from_range_degrees(H, 400)),
        None,
        as_float(from_range_100(J_HK)),
        as_float(from_range_100(Q_HK)),
    )


def Hellwig2022_to_XYZ(
    specification: CAM_Specification_Hellwig2022,
    XYZ_w: ArrayLike,
    L_A: FloatingOrArrayLike,
    Y_b: FloatingOrArrayLike,
    surround: Union[
        InductionFactors_CIECAM02, InductionFactors_Hellwig2022
    ] = VIEWING_CONDITIONS_HELLWIG2022["Average"],
    discount_illuminant: Boolean = False,
) -> NDArray:
    """
    Convert from *Hellwig and Fairchild (2022)* specification to *CIE XYZ*
    tristimulus values.

    This implementation supports the *Helmholtz-Kohlrausch* effect extension
    from :cite:`Hellwig2022a`.

    Parameters
    ----------
    specification
        *Hellwig and Fairchild (2022)* colour appearance model specification.
        Correlate of *Lightness* :math:`J`, correlate of *chroma* :math:`C` or
        correlate of *colourfulness* :math:`M` and *hue* angle :math:`h` in
        degrees must be specified, e.g. :math:`JCh` or :math:`JMh`.
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
    +----------------------------------------+-----------------------\
+---------------+
    | **Domain**                             | **Scale - Reference** \
| **Scale - 1** |
    +========================================+=======================\
+===============+
    | ``CAM_Specification_Hellwig2022.J``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.C``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.h``    | [0, 360]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.s``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.Q``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.M``    | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.H``    | [0, 400]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.J_HK`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``CAM_Specification_Hellwig2022.Q_HK`` | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+
    | ``XYZ_w``                              | [0, 100]              \
| [0, 1]        |
    +----------------------------------------+-----------------------\
+---------------+

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`, :cite:`Hellwig2022a`

    Examples
    --------
    >>> specification = CAM_Specification_Hellwig2022(
    ...     J=41.731207905126638, C=0.025763615829912909, h=217.06795976739301
    ... )
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b)
    ... # doctest: +ELLIPSIS
    array([ 19.01...,  20...  ,  21.78...])
    >>> specification = CAM_Specification_Hellwig2022(
    ...     J_HK=41.880278283880095,
    ...     C=0.025763615829912909,
    ...     h=217.06795976739301,
    ... )
    >>> Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b)
    ... # doctest: +ELLIPSIS
    array([ 19.01...,  20...  ,  21.78...])
    """

    J, C, h, _s, _Q, M, _H, _HC, J_HK, _Q_HK = astuple(specification)

    C = to_domain_100(C)
    h = to_domain_degrees(h)
    M = to_domain_100(M)

    if has_only_nan(J) and not has_only_nan(J_HK):
        J_HK = to_domain_100(J_HK)

        J = J_HK - hue_angle_dependency_Hellwig2022(h) * spow(C, 0.587)
    elif has_only_nan(J):
        raise ValueError(
            'Either "J" or "J_HK" correlate must be defined in '
            'the "CAM_Specification_Hellwig2022" argument!'
        )
    else:
        J = to_domain_100(J)

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

    F_L, z = viewing_conditions_dependent_parameters(Y_b, Y_w, L_A)

    D_RGB = D[..., None] * Y_w[..., None] / RGB_w + 1 - D[..., None]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_wc, F_L
    )

    # Computing achromatic responses for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw)

    # Step 1
    if has_only_nan(M) and not has_only_nan(C):
        M = (C * A_w) / 35
    elif has_only_nan(M):
        raise ValueError(
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_Hellwig2022" argument!'
        )

    # Step 2
    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing achromatic response :math:`A` for the stimulus.
    A = achromatic_response_inverse(A_w, J, surround.c, z)

    # Computing *P_p_1* to *P_p_2*.
    P_p_n = P_p(surround.N_c, e_t, A)
    P_p_1, P_p_2 = tsplit(P_p_n)

    # Step 3
    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    ab = opponent_colour_dimensions_inverse(P_p_1, h, M)
    a, b = tsplit(ab)

    # Step 4
    # Applying post-adaptation non-linear response compression matrix.
    RGB_a = matrix_post_adaptation_non_linear_response_compression(P_p_2, a, b)

    # Step 5
    # Applying inverse post-adaptation non-linear response compression.
    RGB_c = post_adaptation_non_linear_response_compression_inverse(
        RGB_a + 0.1, F_L
    )

    # Step 6
    RGB = RGB_c / D_RGB

    # Step 7
    XYZ = vector_dot(MATRIX_INVERSE_16, RGB)

    return from_range_100(XYZ)


def viewing_conditions_dependent_parameters(
    Y_b: FloatingOrArrayLike,
    Y_w: FloatingOrArrayLike,
    L_A: FloatingOrArrayLike,
) -> Tuple[FloatingOrNDArray, FloatingOrNDArray]:
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
    (1.1675444..., 1.9272135...)
    """

    Y_b = as_float_array(Y_b)
    Y_w = as_float_array(Y_w)

    with sdiv_mode():
        n = sdiv(Y_b, Y_w)

    F_L = luminance_level_adaptation_factor(L_A)
    z = base_exponential_non_linearity(n)

    return F_L, z


def achromatic_response_forward(RGB: ArrayLike) -> FloatingOrNDArray:
    """
    Return the achromatic response :math:`A` from given compressed
    *CAM16* transform sharpened *RGB* array and :math:`N_{bb}` chromatic
    induction factor for forward *Hellwig and Fairchild (2022)* implementation.

    Parameters
    ----------
    RGB
        Compressed *CAM16* transform sharpened *RGB* array.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Achromatic response :math:`A`.

    Examples
    --------
    >>> RGB = np.array([7.94634384, 7.94713791, 7.9488967])
    >>> achromatic_response_forward(RGB)  # doctest: +ELLIPSIS
    23.9322704...
    """

    R, G, B = tsplit(RGB)

    A = 2 * R + G + 0.05 * B - 0.305

    return A


def opponent_colour_dimensions_inverse(
    P_p_1: FloatingOrArrayLike, h: FloatingOrArrayLike, M: FloatingOrArrayLike
) -> NDArray:
    """
    Return opponent colour dimensions from given point :math:`P'_1`, hue
    :math:`h` in degrees and correlate of *colourfulness* :math:`M` for
    inverse *Hellwig and Fairchild (2022)* implementation.

    Parameters
    ----------
    P_p_1
        Point :math:`P'_1`.
    h
        Hue :math:`h` in degrees.
    M
        Correlate of *colourfulness* :math:`M`.

    Returns
    -------
    :class:`numpy.ndarray`
        Opponent colour dimensions.

    Examples
    --------
    >>> P_p_1 = 48.7719436928
    >>> h = 217.067959767393
    >>> M = 0.0387637282462
    >>> opponent_colour_dimensions_inverse(P_p_1, h, M)  # doctest: +ELLIPSIS
    array([-0.0006341..., -0.0004790...])
    """

    P_p_1 = as_float_array(P_p_1)
    M = as_float_array(M)

    hr = np.radians(h)

    with sdiv_mode():
        gamma = M / P_p_1

    a = gamma * np.cos(hr)
    b = gamma * np.sin(hr)

    ab = tstack([a, b])

    return ab


def eccentricity_factor(h: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Return the eccentricity factor :math:`e_t` from given hue :math:`h` angle
    in degrees for forward *CIECAM02* implementation.

    Parameters
    ----------
    h
        Hue :math:`h` angle in degrees.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Eccentricity factor :math:`e_t`.

    Examples
    --------
    >>> eccentricity_factor(217.067959767393)  # doctest: +ELLIPSIS
    0.9945215...
    """

    h = as_float_array(h)

    hr = np.radians(h)

    _h = hr
    _2_h = 2 * hr
    _3_h = 3 * hr
    _4_h = 4 * hr

    e_t = (
        -0.0582 * np.cos(_h)
        - 0.0258 * np.cos(_2_h)
        - 0.1347 * np.cos(_3_h)
        + 0.0289 * np.cos(_4_h)
        - 0.1475 * np.sin(_h)
        - 0.0308 * np.sin(_2_h)
        + 0.0385 * np.sin(_3_h)
        + 0.0096 * np.sin(_4_h)
        + 1
    )

    return e_t


def brightness_correlate(
    c: FloatingOrArrayLike,
    J: FloatingOrArrayLike,
    A_w: FloatingOrArrayLike,
) -> FloatingOrNDArray:
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

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *Brightness* correlate :math:`Q`.

    Examples
    --------
    >>> c = 0.69
    >>> J = 41.7310911325
    >>> A_w = 46.1741997997
    >>> brightness_correlate(c, J, A_w)  # doctest: +ELLIPSIS
    55.8521663...
    """

    c = as_float_array(c)
    J = as_float_array(J)
    A_w = as_float_array(A_w)

    with sdiv_mode():
        Q = (2 / c) * (J / 100) * A_w

    return Q


def colourfulness_correlate(
    N_c: FloatingOrArrayLike,
    e_t: FloatingOrArrayLike,
    a: FloatingOrArrayLike,
    b: FloatingOrArrayLike,
) -> FloatingOrNDArray:
    """
    Return the *colourfulness* correlate :math:`M`.

    Parameters
    ----------
    N_c
        Surround chromatic induction factor :math:`N_{c}`.
    e_t
        Eccentricity factor :math:`e_t`.
    a
        Opponent colour dimension :math:`a`.
    b
        Opponent colour dimension :math:`b`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *Colourfulness* correlate :math:`M`.

    Examples
    --------
    >>> N_c = 1
    >>> e_t = 1.13423124867
    >>> a = -0.00063418423001
    >>> b = -0.000479072513542
    >>> colourfulness_correlate(N_c, e_t, a, b)  # doctest: +ELLIPSIS
    0.0387637...
    """

    N_c = as_float_array(N_c)
    e_t = as_float_array(e_t)
    a = as_float_array(a)
    b = as_float_array(b)

    M = 43 * N_c * e_t * np.sqrt(a**2 + b**2)

    return M


def chroma_correlate(
    M: FloatingOrArrayLike,
    A_w: FloatingOrArrayLike,
) -> FloatingOrNDArray:
    """
    Return the *chroma* correlate :math:`C`.

    Parameters
    ----------
    M
        *Colourfulness* correlate :math:`M`.
    A_w
        Achromatic response :math:`A_w` for the whitepoint.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *Chroma* correlate :math:`C`.

    Examples
    --------
    >>> M = 0.0387637282462
    >>> A_w = 46.1741997997
    >>> chroma_correlate(M, A_w)  # doctest: +ELLIPSIS
    0.0293828...
    """

    M = as_float_array(M)
    A_w = as_float_array(A_w)

    with sdiv_mode():
        C = 35 * sdiv(M, A_w)

    return C


def saturation_correlate(
    M: FloatingOrArrayLike, Q: FloatingOrArrayLike
) -> FloatingOrNDArray:
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
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *Saturation* correlate :math:`s`.

    Examples
    --------
    >>> M = 0.0387637282462
    >>> Q = 55.8523226578
    >>> saturation_correlate(M, Q)  # doctest: +ELLIPSIS
    0.0694039...
    """

    M = as_float_array(M)
    Q = as_float_array(Q)

    with sdiv_mode():
        s = 100 * sdiv(M, Q)

    return s


def P_p(
    N_c: FloatingOrArrayLike,
    e_t: FloatingOrArrayLike,
    A: FloatingOrArrayLike,
) -> NDArray:
    """
    Return the points :math:`P'_1` and :math:`P'_2`.

    Parameters
    ----------
    N_c
        Surround chromatic induction factor :math:`N_{c}`.
    e_t
        Eccentricity factor :math:`e_t`.
    A
        Achromatic response  :math:`A` for the stimulus.

    Returns
    -------
    :class:`numpy.ndarray`
        Points :math:`P'`.

    Examples
    --------
    >>> N_c = 1
    >>> e_t = 1.13423124867
    >>> A = 23.9322704261
    >>> P_p(N_c, e_t, A)  # doctest: +ELLIPSIS
    array([ 48.7719436...,  23.9322704...])
    """

    N_c = as_float_array(N_c)
    e_t = as_float_array(e_t)
    A = as_float_array(A)

    P_p_1 = 43 * N_c * e_t
    P_p_2 = A

    P_p_n = tstack([P_p_1, P_p_2])

    return P_p_n


def hue_angle_dependency_Hellwig2022(
    h: FloatingOrArrayLike,
) -> FloatingOrNDArray:
    """
    Compute the hue angle dependency of the *Helmholtz-Kohlrausch* effect.

    Parameters
    ----------
    h
        Hue :math:`h` angle in degrees.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Hue angle dependency.

    References
    ----------
    :cite:`Hellwig2022a`

    Examples
    --------
    >>> hue_angle_dependency_Hellwig2022(217.06795976739301)
    ... # doctest: +ELLIPSIS
    1.2768219...
    """

    h = as_float_array(h)

    h_r = np.radians(h)

    return as_float(
        -0.160 * np.cos(h_r)
        + 0.132 * np.cos(2 * h_r)
        - 0.405 * np.sin(h_r)
        + 0.080 * np.sin(2 * h_r)
        + 0.792
    )
