"""
ATD (1995) Colour Vision Model
==============================

Define the *ATD (1995)* colour vision model.

-   :class:`colour.CAM_Specification_ATD95`
-   :func:`colour.XYZ_to_ATD95`

Notes
-----
-   According to *CIE TC1-34* definition of a colour appearance model, the
    *ATD (1995)* model cannot be considered as a colour appearance model.
    It was developed with different aims and is described as a model of
    colour vision.

References
----------
-   :cite:`Fairchild2013v` : Fairchild, M. D. (2013). ATD Model. In Color
    Appearance Models (3rd ed., pp. 5852-5991). Wiley. ISBN:B00DAYO8E2
-   :cite:`Guth1995a` : Guth, S. L. (1995). Further applications of the ATD
    model for color vision. In E. Walowit (Ed.), Proc. SPIE 2414,
    Device-Independent Color Imaging II (Vol. 2414, pp. 12-26).
    doi:10.1117/12.206546
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from colour.algebra import sdiv, sdiv_mode, spow, vecmul

if typing.TYPE_CHECKING:
    from colour.hints import Annotated, ArrayLike, Domain100, NDArrayFloat

from colour.utilities import (
    MixinDataclassArithmetic,
    array_namespace,
    as_float,
    from_range_degrees,
    to_domain_100,
    tsplit,
    tstack,
    xp_as_float_array,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CAM_ReferenceSpecification_ATD95",
    "CAM_Specification_ATD95",
    "XYZ_to_ATD95",
]


@dataclass
class CAM_ReferenceSpecification_ATD95(MixinDataclassArithmetic):
    """
    Define the *ATD (1995)* colour vision model reference specification.

    This specification contains field names consistent with the *Fairchild
    (2013)* reference.

    Parameters
    ----------
    H
        *Hue* angle :math:`H` in degrees.
    C
        Correlate of *saturation* :math:`C`. *Guth (1995)* incorrectly uses
        the terms saturation and chroma interchangeably. However, :math:`C`
        represents a measure of saturation rather than chroma since it is
        calculated relative to the achromatic response for the stimulus
        rather than that of a similarly illuminated white.
    Br
        Correlate of *brightness* :math:`Br`.
    A_1
        First stage :math:`A_1` response.
    T_1
        First stage :math:`T_1` response.
    D_1
        First stage :math:`D_1` response.
    A_2
        Second stage :math:`A_2` response.
    T_2
        Second stage :math:`A_2` response.
    D_2
        Second stage :math:`D_2` response.

    References
    ----------
    :cite:`Fairchild2013v`, :cite:`Guth1995a`
    """

    H: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Br: float | NDArrayFloat | None = field(default_factory=lambda: None)
    A_1: float | NDArrayFloat | None = field(default_factory=lambda: None)
    T_1: float | NDArrayFloat | None = field(default_factory=lambda: None)
    D_1: float | NDArrayFloat | None = field(default_factory=lambda: None)
    A_2: float | NDArrayFloat | None = field(default_factory=lambda: None)
    T_2: float | NDArrayFloat | None = field(default_factory=lambda: None)
    D_2: float | NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class CAM_Specification_ATD95(MixinDataclassArithmetic):
    """
    Define the *ATD (1995)* colour vision model specification.

    This specification provides a standardized interface for the *ATD (1995)*
    model with field names consistent across all colour appearance models in
    :mod:`colour.appearance`. While the field names differ from the original
    *Fairchild (2013)* reference notation, they map directly to the model's
    perceptual correlates.

    Parameters
    ----------
    h
        *Hue* :math:`H = T_2 / D_2` per *Guth (1995)* Equation 14.24;
        the raw opponent-channel ratio, not a hue angle in degrees. A
        proper hue angle can be obtained from
        :func:`numpy.arctan2`\\ ``(T_2, D_2)`` per *Fairchild (2013)*
        p.243, which notes the raw ratio is equivocal (equal for
        complementary hues, infinite or undefined in some cases).
    C
        Correlate of *saturation* :math:`C`. *Guth (1995)* incorrectly uses
        the terms saturation and chroma interchangeably. However, :math:`C`
        represents a measure of saturation rather than chroma since it is
        measured relative to the achromatic response for the stimulus rather
        than that of a similarly illuminated white.
    Q
        Correlate of *brightness* :math:`Br`.
    A_1
        First stage :math:`A_1` response.
    T_1
        First stage :math:`T_1` response.
    D_1
        First stage :math:`D_1` response.
    A_2
        Second stage :math:`A_2` response.
    T_2
        Second stage :math:`T_2` response.
    D_2
        Second stage :math:`D_2` response.

    Notes
    -----
    -   This specification is the one used in the current model
        implementation.

    References
    ----------
    :cite:`Fairchild2013v`, :cite:`Guth1995a`
    """

    h: float | NDArrayFloat | None = field(default_factory=lambda: None)
    C: float | NDArrayFloat | None = field(default_factory=lambda: None)
    Q: float | NDArrayFloat | None = field(default_factory=lambda: None)
    A_1: float | NDArrayFloat | None = field(default_factory=lambda: None)
    T_1: float | NDArrayFloat | None = field(default_factory=lambda: None)
    D_1: float | NDArrayFloat | None = field(default_factory=lambda: None)
    A_2: float | NDArrayFloat | None = field(default_factory=lambda: None)
    T_2: float | NDArrayFloat | None = field(default_factory=lambda: None)
    D_2: float | NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_ATD95(
    XYZ: Domain100,
    XYZ_0: Domain100,
    Y_0: ArrayLike,
    k_1: ArrayLike,
    k_2: ArrayLike,
    sigma: ArrayLike = 300,
) -> Annotated[CAM_Specification_ATD95, 360]:
    """
    Compute the *ATD (1995)* colour vision model correlates from the specified
    *CIE XYZ* tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_0
        *CIE XYZ* tristimulus values of reference white.
    Y_0
        Absolute adapting field luminance in :math:`cd/m^2`.
    k_1
        Application specific weight :math:`k_1`.
    k_2
        Application specific weight :math:`k_2`.
    sigma
        Constant :math:`\\sigma` varied to predict different types of data.

    Returns
    -------
    :class:`colour.CAM_Specification_ATD95`
        *ATD (1995)* colour vision model specification.

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

    -   For unrelated colours, there is only self-adaptation and :math:`k_1`
        is set to 1.0 while :math:`k_2` is set to 0.0. For related colours
        such as typical colorimetric applications, :math:`k_1` is set to 0.0
        and :math:`k_2` is set to a value between 15 and 50 *(Guth, 1995)*.

    References
    ----------
    :cite:`Fairchild2013v`, :cite:`Guth1995a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_0 = np.array([95.05, 100.00, 108.88])
    >>> Y_0 = 318.31
    >>> k_1 = 0.0
    >>> k_2 = 50.0
    >>> XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)  # doctest: +ELLIPSIS
    CAM_Specification_ATD95(h=np.float64(1.9089869...), \
C=np.float64(1.2064060...), Q=np.float64(0.1814003...), \
A_1=np.float64(0.1787931...), T_1=np.float64(0.0286942...), \
D_1=np.float64(0.0107584...), A_2=np.float64(0.0192182...), \
T_2=np.float64(0.0205377...), D_2=np.float64(0.0107584...))
    """

    XYZ = to_domain_100(XYZ)
    XYZ_0 = to_domain_100(XYZ_0)

    xp = array_namespace(XYZ, XYZ_0, Y_0, k_1, k_2, sigma)

    Y_0 = xp_as_float_array(Y_0, xp=xp, like=XYZ)
    k_1 = xp_as_float_array(k_1, xp=xp, like=XYZ)
    k_2 = xp_as_float_array(k_2, xp=xp, like=XYZ)
    sigma = xp_as_float_array(sigma, xp=xp, like=XYZ)

    # Converting luminance in :math:`cd/m^2` to retinal illuminance in trolands
    # for the stimulus and the reference white.
    XYZ = 18 * spow(Y_0[..., None] * XYZ / 100, 0.8)
    XYZ_0 = 18 * spow(Y_0[..., None] * XYZ_0 / 100, 0.8)

    # Computing the adaptation stimulus :math:`XYZ_a` then deriving the
    # post-adaptation cone signals via the *ATD95* :math:`XYZ \\rightarrow LMS`
    # transform applied to both the stimulus and the adaptation stimulus.
    XYZ_a = k_1[..., None] * XYZ + k_2[..., None] * XYZ_0
    LMS_scales = xp_as_float_array([0.66, 1.0, 0.43], xp=xp, like=XYZ)
    LMS_offsets = xp_as_float_array([0.024, 0.036, 0.31], xp=xp, like=XYZ)
    LMS_matrix = [
        [0.2435, 0.8524, -0.0516],
        [-0.3954, 1.1642, 0.0837],
        [0.0000, 0.0400, 0.6225],
    ]
    LMS = spow(vecmul(LMS_matrix, XYZ) * LMS_scales, 0.7) + LMS_offsets
    LMS_a = spow(vecmul(LMS_matrix, XYZ_a) * LMS_scales, 0.7) + LMS_offsets

    LMS_g = LMS * (sigma[..., None] / (sigma[..., None] + LMS_a))

    # Computing opponent colour dimensions: 6 linear combinations of the
    # post-adaptation cone signals, each passed through the saturating final
    # response :math:`v / (200 + |v|)`.
    L_g, M_g, S_g = tsplit(LMS_g)
    A_1i = 3.57 * L_g + 2.64 * M_g
    T_1i = 7.18 * L_g - 6.21 * M_g
    D_1i = -0.7 * L_g + 0.085 * M_g + S_g
    A_2i = 0.09 * A_1i
    T_2i = 0.43 * T_1i + 0.76 * D_1i
    D_2i = D_1i
    stage = tstack([A_1i, T_1i, D_1i, A_2i, T_2i, D_2i])
    stage_final = stage / (200 + xp.abs(stage))
    A_1, T_1, D_1, A_2, T_2, D_2 = tsplit(stage_final)

    # Computing the correlate of *brightness* :math:`Br`.
    Br = spow(A_1**2 + T_1**2 + D_1**2, 0.5)

    # Computing the correlate of *saturation* :math:`C` and the *hue*
    # :math:`H`. Note that the reference does not take the modulus of the
    # :math:`H`, thus :math:`H` can exceed 360 degrees.
    with sdiv_mode():
        C = sdiv(spow(T_2**2 + D_2**2, 0.5), A_2)
        H = sdiv(T_2, D_2)

    return CAM_Specification_ATD95(
        h=as_float(from_range_degrees(H)),
        C=as_float(C),
        Q=as_float(Br),
        A_1=as_float(A_1),
        T_1=as_float(T_1),
        D_1=as_float(D_1),
        A_2=as_float(A_2),
        T_2=as_float(T_2),
        D_2=as_float(D_2),
    )
