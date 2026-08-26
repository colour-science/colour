"""
Kim, Weyrich and Kautz (2009) Colour Appearance Model
=====================================================

Define the *Kim, Weyrich and Kautz (2009)* colour appearance model for
predicting perceptual colour attributes under varying viewing conditions.

This model extends *CIECAM02* to handle high dynamic range viewing conditions
by introducing media-specific parameters that modulate lightness prediction.

-   :class:`colour.appearance.InductionFactors_Kim2009`
-   :attr:`colour.VIEWING_CONDITIONS_KIM2009`
-   :class:`colour.appearance.MediaParameters_Kim2009`
-   :attr:`colour.MEDIA_PARAMETERS_KIM2009`
-   :class:`colour.CAM_Specification_Kim2009`
-   :func:`colour.XYZ_to_Kim2009`
-   :func:`colour.Kim2009_to_XYZ`

References
----------
-   :cite:`Kim2009` : Kim, M., Weyrich, T., & Kautz, J. (2009). Modeling Human
    Color Perception under Extended Luminance Levels. ACM Transactions on
    Graphics, 28(3), 27:1--27:9. doi:10.1145/1531326.1531333
"""

from __future__ import annotations

import typing
from dataclasses import astuple, dataclass, field

from colour.adaptation import CAT_CAT02
from colour.algebra import sdiv, sdiv_mode, spow, vecmul
from colour.appearance.ciecam02 import (
    CAT_INVERSE_CAT02,
    MATRIX_HPE_TO_XYZ,
    MATRIX_XYZ_TO_HPE,
    VIEWING_CONDITIONS_CIECAM02,
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
    "InductionFactors_Kim2009",
    "VIEWING_CONDITIONS_KIM2009",
    "MediaParameters_Kim2009",
    "MEDIA_PARAMETERS_KIM2009",
    "CAM_Specification_Kim2009",
    "XYZ_to_Kim2009",
    "Kim2009_to_XYZ",
]


@dataclass(frozen=True)
class InductionFactors_Kim2009(MixinDataclassIterable):
    """
    Define the *Kim, Weyrich and Kautz (2009)* colour appearance model
    surround induction factors.

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
    -   The *Kim, Weyrich and Kautz (2009)* colour appearance model induction
        factors are the same as the *CIECAM02* colour appearance model.
    -   The *Kim, Weyrich and Kautz (2009)* colour appearance model separates
        the surround modelled by the
        :class:`colour.appearance.InductionFactors_Kim2009` class instance
        from the media, modelled with the
        :class:`colour.appearance.MediaParameters_Kim2009` class instance.

    References
    ----------
    :cite:`Kim2009`
    """

    F: float
    c: float
    N_c: float


VIEWING_CONDITIONS_KIM2009: CanonicalMapping = CanonicalMapping(
    VIEWING_CONDITIONS_CIECAM02
)
VIEWING_CONDITIONS_KIM2009.__doc__ = """
Define the reference *Kim, Weyrich and Kautz (2009)* colour appearance model
viewing conditions inherited from *CIECAM02*.

References
----------
:cite:`Kim2009`
"""


@dataclass(frozen=True)
class MediaParameters_Kim2009:
    """
    Define the media parameters for the *Kim, Weyrich and Kautz (2009)* colour
    appearance model.

    Parameters
    ----------
    E
        Lightness prediction modulating parameter :math:`E`.

    References
    ----------
    :cite:`Kim2009`
    """

    E: float


MEDIA_PARAMETERS_KIM2009: CanonicalMapping = CanonicalMapping(
    {
        "High-luminance LCD Display": MediaParameters_Kim2009(1),
        "Transparent Advertising Media": MediaParameters_Kim2009(1.2175),
        "CRT Displays": MediaParameters_Kim2009(1.4572),
        "Reflective Paper": MediaParameters_Kim2009(1.7526),
    }
)
MEDIA_PARAMETERS_KIM2009.__doc__ = """
Define the reference *Kim, Weyrich and Kautz (2009)* colour appearance model
media parameters.

References
----------
:cite:`Kim2009`

Aliases:

-   'bright_lcd_display': 'High-luminance LCD Display'
-   'advertising_transparencies': 'Transparent Advertising Media'
-   'crt': 'CRT Displays'
-   'paper': 'Reflective Paper'
"""
MEDIA_PARAMETERS_KIM2009["bright_lcd_display"] = MEDIA_PARAMETERS_KIM2009[
    "High-luminance LCD Display"
]
MEDIA_PARAMETERS_KIM2009["advertising_transparencies"] = MEDIA_PARAMETERS_KIM2009[
    "Transparent Advertising Media"
]
MEDIA_PARAMETERS_KIM2009["crt"] = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
MEDIA_PARAMETERS_KIM2009["paper"] = MEDIA_PARAMETERS_KIM2009["Reflective Paper"]


@dataclass
class CAM_Specification_Kim2009(MixinDataclassArithmetic):
    """
    Represent the *Kim, Weyrich and Kautz (2009)* colour appearance model
    output specification.

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
    :cite:`Kim2009`
    """

    J: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    s: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    M: float | NDArrayFloat | None = field(default_factory=lambda: None)
    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HC: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_Kim2009(
    XYZ: Domain100,
    XYZ_w: Domain100,
    L_A: ArrayLike,
    media: MediaParameters_Kim2009 = MEDIA_PARAMETERS_KIM2009["CRT Displays"],
    surround: InductionFactors_Kim2009 = VIEWING_CONDITIONS_KIM2009["Average"],
    n_c: float = 0.57,
    discount_illuminant: bool = False,
    compute_H: bool = False,
) -> Annotated[CAM_Specification_Kim2009, (100, 100, 360, 100, 100, 100, 400)]:
    """
    Compute the *Kim, Weyrich and Kautz (2009)* colour appearance model
    correlates from the specified *CIE XYZ* tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often
        taken to be 20% of the luminance of a white object in the scene).
    media
        Media parameters.
    surround
        Surround viewing conditions induction factors.
    n_c
        Cone response sigmoidal curve modulating factor :math:`n_c`.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.
    compute_H
        When *True*, compute the *Hue Quadrature* :math:`H` correlate
        via :func:`colour.appearance.hue_quadrature`. Defaults to
        *False* because :math:`H` is rarely consumed downstream and
        skipping the bin search is a measurable cost saving.

    Returns
    -------
    :class:`colour.CAM_Specification_Kim2009`
       *Kim, Weyrich and Kautz (2009)* colour appearance model
       specification.

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
    :cite:`Kim2009`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> media = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
    >>> surround = VIEWING_CONDITIONS_KIM2009["Average"]
    >>> XYZ_to_Kim2009(XYZ, XYZ_w, L_A, media, surround, compute_H=True)
    ... # doctest: +ELLIPSIS
    CAM_Specification_Kim2009(J=np.float64(28.8619089...), C=np.float64(0.5592455...), \
h=np.float64(219.0480667...), s=np.float64(9.3837797...), Q=np.float64(52.7138883...), \
M=np.float64(0.4641738...), H=np.float64(278.0602824...), HC=None)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)

    xp = array_namespace(XYZ, XYZ_w, L_A)

    XYZ = xp_as_float_array(XYZ, xp=xp)
    XYZ_w = xp_as_float_array(XYZ_w, xp=xp, like=XYZ)
    L_A = xp_as_float_array(L_A, xp=xp, like=XYZ)

    _X_w, Y_w, _Z_w = tsplit(XYZ_w)

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values.
    RGB = vecmul(CAT_CAT02, XYZ)
    RGB_w = vecmul(CAT_CAT02, XYZ_w)

    # Computing degree of adaptation :math:`D`, same formulation as in
    # *CIECAM02*; bypassed entirely when ``discount_illuminant`` is set.
    if discount_illuminant:
        D = xp_as_float_array(ones(L_A.shape), xp=xp, like=XYZ)
    else:
        F = xp_as_float_array(surround.F, xp=xp, like=XYZ)
        D = F * (1 - (1 / 3.6) * xp.exp((-L_A - 42) / 92))

    # Computing full chromatic adaptation, same formulation as in
    # *CIECAM02*, applied to the stimulus and reference white via a
    # shared factor.
    with sdiv_mode():
        D_factor = Y_w[..., None] * sdiv(D[..., None], RGB_w) + 1 - D[..., None]
    XYZ_c = D_factor * RGB
    XYZ_wc = D_factor * RGB_w

    # Converting to *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
    # colourspace, same transform as in *CIECAM02*.
    MATRIX_XYZ_HPE_x_CAT_INVERSE = xp.matmul(
        xp_as_float_array(MATRIX_XYZ_TO_HPE, xp=xp, like=XYZ),
        xp_as_float_array(CAT_INVERSE_CAT02, xp=xp, like=XYZ),
    )
    LMS = vecmul(MATRIX_XYZ_HPE_x_CAT_INVERSE, XYZ_c)
    LMS_w = vecmul(MATRIX_XYZ_HPE_x_CAT_INVERSE, XYZ_wc)

    # Cones absolute response.
    LMS_n_c = spow(LMS, n_c)
    LMS_w_n_c = spow(LMS_w, n_c)
    L_A_n_c = spow(L_A, n_c)
    LMS_p = LMS_n_c / (LMS_n_c + L_A_n_c)
    LMS_wp = LMS_w_n_c / (LMS_w_n_c + L_A_n_c)

    # Achromatic signal :math:`A` and :math:`A_w`.
    v_A = xp_as_float_array([40, 20, 1], xp=xp, like=LMS_wp)
    A = xp.sum(v_A * LMS_p, axis=-1) / 61
    A_w = xp.sum(v_A * LMS_wp, axis=-1) / 61

    # Perceived *Lightness* :math:`J_p`.
    a_j, b_j, o_j, n_j = 0.89, 0.24, 0.65, 3.65
    A_A_w = A / A_w
    J_p = spow((-(A_A_w - b_j) * spow(o_j, n_j)) / (A_A_w - b_j - a_j), 1 / n_j)

    # Computing the media dependent *Lightness* :math:`J`.
    J = 100 * (media.E * (J_p - 1) + 1)

    # Computing the correlate of *brightness* :math:`Q`.
    n_q = 0.1308
    Q = J * spow(Y_w, n_q)

    # Opponent signals :math:`a` and :math:`b`.
    a = (1 / 11) * xp.sum(
        xp_as_float_array([11, -12, 1], xp=xp, like=LMS_p) * LMS_p, axis=-1
    )
    b = (1 / 9) * xp.sum(
        xp_as_float_array([1, 1, -2], xp=xp, like=LMS_p) * LMS_p, axis=-1
    )

    # Computing the correlate of *chroma* :math:`C`.
    a_k, n_k = 456.5, 0.62
    C = a_k * spow(xp.hypot(a, b), n_k)

    # Computing the correlate of *colourfulness* :math:`M`.
    a_m, b_m = 0.11, 0.61
    M = C * (a_m * xp.log10(Y_w) + b_m)

    # Computing the correlate of *saturation* :math:`s`.
    s = 100 * xp.sqrt(M / Q)

    # Computing the *hue* angle :math:`h`.
    h = xp_degrees(xp.atan2(b, a)) % 360

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h) if compute_H else xp.full_like(h, float("nan"))

    return CAM_Specification_Kim2009(
        J=as_float(from_range_100(J)),
        C=as_float(from_range_100(C)),
        h=as_float(from_range_degrees(h)),
        s=as_float(from_range_100(s)),
        Q=as_float(from_range_100(Q)),
        M=as_float(from_range_100(M)),
        H=as_float(from_range_degrees(H, 400)),
        HC=None,
    )


def Kim2009_to_XYZ(
    specification: Annotated[
        CAM_Specification_Kim2009, (100, 100, 360, 100, 100, 100, 400)
    ],
    XYZ_w: Domain100,
    L_A: ArrayLike,
    media: MediaParameters_Kim2009 = MEDIA_PARAMETERS_KIM2009["CRT Displays"],
    surround: InductionFactors_Kim2009 = VIEWING_CONDITIONS_KIM2009["Average"],
    n_c: float = 0.57,
    discount_illuminant: bool = False,
) -> Range100:
    """
    Convert the *Kim, Weyrich and Kautz (2009)* colour appearance model
    specification to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    specification
         *Kim, Weyrich and Kautz (2009)* colour appearance model specification.
         Correlate of *Lightness* :math:`J`, correlate of *chroma* :math:`C` or
         correlate of *colourfulness* :math:`M` and *hue* angle :math:`h` in
         degrees must be specified, e.g., :math:`JCh` or :math:`JMh`.
    XYZ_w
        *CIE XYZ* tristimulus values of reference white.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    media
        Media parameters.
    surround
        Surround viewing conditions induction factors.
    n_c
        Cone response sigmoidal curve modulating factor :math:`n_c`.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.

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
    :cite:`Kim2009`

    Examples
    --------
    >>> import numpy as np
    >>> specification = CAM_Specification_Kim2009(
    ...     J=28.861908975839647, C=0.5592455924373706, h=219.04806677662953
    ... )
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> media = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
    >>> surround = VIEWING_CONDITIONS_KIM2009["Average"]
    >>> Kim2009_to_XYZ(specification, XYZ_w, L_A, media, surround)
    ... # doctest: +ELLIPSIS
    array([19.0099995..., 19.9999999..., 21.7800000...])
    """

    J, C, h, _s, _Q, M, _H, _HC = astuple(specification)

    J = to_domain_100(J)
    C = to_domain_100(C)
    h = to_domain_degrees(h)
    M = to_domain_100(M)
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

    # Converting *CIE XYZ* tristimulus values to *CMCCAT2000* transform
    # sharpened *RGB* values for the reference white.
    RGB_w = vecmul(CAT_CAT02, XYZ_w)

    # Computing degree of adaptation :math:`D`, same formulation as in
    # *CIECAM02*; bypassed entirely when ``discount_illuminant`` is set.
    if discount_illuminant:
        D = xp_as_float_array(ones(L_A.shape), xp=xp, like=J)
    else:
        F = xp_as_float_array(surround.F, xp=xp, like=J)
        D = F * (1 - (1 / 3.6) * xp.exp((-L_A - 42) / 92))

    # Computing full chromatic adaptation for the reference white,
    # same formulation as in *CIECAM02*. The :math:`D_{factor}` value
    # is reused on the way out.
    with sdiv_mode():
        D_factor = Y_w[..., None] * sdiv(D[..., None], RGB_w) + 1 - D[..., None]
    XYZ_wc = D_factor * RGB_w

    # Converting to *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
    # colourspace, same transform as in *CIECAM02*.
    MATRIX_XYZ_HPE_x_CAT_INVERSE = xp.matmul(
        xp_as_float_array(MATRIX_XYZ_TO_HPE, xp=xp, like=J),
        xp_as_float_array(CAT_INVERSE_CAT02, xp=xp, like=J),
    )
    LMS_w = vecmul(MATRIX_XYZ_HPE_x_CAT_INVERSE, XYZ_wc)

    if has_only_nan(C) and not has_only_nan(M):
        a_m, b_m = 0.11, 0.61
        C = M / (a_m * xp.log10(Y_w) + b_m)
    elif has_only_nan(C):
        error = (
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_Kim2009" argument!'
        )

        raise ValueError(error)

    # Cones absolute response.
    LMS_w_n_c = spow(LMS_w, n_c)
    L_A_n_c = spow(L_A, n_c)
    LMS_wp = LMS_w_n_c / (LMS_w_n_c + L_A_n_c)

    # Achromatic signal :math:`A_w`
    v_A = xp_as_float_array([40, 20, 1], xp=xp, like=LMS_wp)
    A_w = xp.sum(v_A * LMS_wp, axis=-1) / 61

    # Perceived *Lightness* :math:`J_p`.
    J_p = (J / 100 - 1) / media.E + 1

    # Achromatic signal :math:`A`.
    a_j, b_j, n_j, o_j = 0.89, 0.24, 3.65, 0.65
    J_p_n_j = spow(J_p, n_j)
    A = A_w * ((a_j * J_p_n_j) / (J_p_n_j + spow(o_j, n_j)) + b_j)

    # Opponent signals :math:`a` and :math:`b`.
    a_k, n_k = 456.5, 0.62
    C_a_k_n_k = spow(C / a_k, 1 / n_k)
    hr = xp_radians(h)
    a, b = xp.cos(hr) * C_a_k_n_k, xp.sin(hr) * C_a_k_n_k

    # Cones absolute response.
    M = xp_as_float_array(
        [
            [1.0000, 0.3215, 0.2053],
            [1.0000, -0.6351, -0.1860],
            [1.0000, -0.1568, -4.4904],
        ],
        xp=xp,
        like=A,
    )
    LMS_p = vecmul(M, tstack([A, a, b]))
    LMS = spow((-spow(L_A, n_c) * LMS_p) / (LMS_p - 1), 1 / n_c)

    # Converting from *Hunt-Pointer-Estevez* :math:`\\rho\\gamma\\beta`
    # colourspace back to adapted *RGB*, same transform as in *CIECAM02*.
    CAT_x_MATRIX_HPE = xp.matmul(
        xp_as_float_array(CAT_CAT02, xp=xp, like=J),
        xp_as_float_array(MATRIX_HPE_TO_XYZ, xp=xp, like=J),
    )
    RGB_c = vecmul(CAT_x_MATRIX_HPE, LMS)

    # Applying inverse full chromatic adaptation, reusing the
    # :math:`D_{factor}` value precomputed on the forward path.
    RGB = RGB_c / D_factor

    XYZ = vecmul(CAT_INVERSE_CAT02, RGB)

    return from_range_100(XYZ)
