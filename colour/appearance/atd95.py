# -*- coding: utf-8 -*-
"""
ATD (1995) Colour Vision Model
==============================

Defines the *ATD (1995)* colour vision model objects:

-   :class:`colour.CAM_Specification_ATD95`
-   :func:`colour.XYZ_to_ATD95`

Notes
-----
-   According to *CIE TC1-34* definition of a colour appearance model, the
    *ATD (1995)* model cannot be considered as a colour appearance model.
    It was developed with different aims and is described as a model of colour
    vision.

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

import numpy as np
from dataclasses import dataclass, field

from colour.algebra import spow, vector_dot
from colour.hints import (
    ArrayLike,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
    Optional,
)
from colour.utilities import (
    MixinDataclassArray,
    as_float,
    as_float_array,
    from_range_degrees,
    to_domain_100,
    tsplit,
    tstack,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CAM_ReferenceSpecification_ATD95',
    'CAM_Specification_ATD95',
    'XYZ_to_ATD95',
    'luminance_to_retinal_illuminance',
    'XYZ_to_LMS_ATD95',
    'opponent_colour_dimensions',
    'final_response',
]


@dataclass
class CAM_ReferenceSpecification_ATD95(MixinDataclassArray):
    """
    Defines the *ATD (1995)* colour vision model reference specification.

    This specification has field names consistent with *Fairchild (2013)*
    reference.

    Parameters
    ----------
    H
        *Hue* angle :math:`H` in degrees.
    C
        Correlate of *saturation* :math:`C`. *Guth (1995)* incorrectly uses the
        terms saturation and chroma interchangeably. However, :math:`C` is here
        a measure of saturation rather than chroma since it is measured
        relative to the achromatic response for the stimulus rather than that
        of a similarly illuminated white.
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

    H: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    C: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    Br: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    A_1: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    T_1: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    D_1: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    A_2: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    T_2: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    D_2: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)


@dataclass
class CAM_Specification_ATD95(MixinDataclassArray):
    """
    Defines the *ATD (1995)* colour vision model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    *Fairchild (2013)* reference.

    Parameters
    ----------
    h
        *Hue* angle :math:`H` in degrees.
    C
        Correlate of *saturation* :math:`C`. *Guth (1995)* incorrectly uses the
        terms saturation and chroma interchangeably. However, :math:`C` is here
        a measure of saturation rather than chroma since it is measured
        relative to the achromatic response for the stimulus rather than that
        of a similarly illuminated white.
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
        Second stage :math:`A_2` response.
    D_2
        Second stage :math:`D_2` response.

    Notes
    -----
    -   This specification is the one used in the current model implementation.

    References
    ----------
    :cite:`Fairchild2013v`, :cite:`Guth1995a`
    """

    h: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    C: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    Q: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    A_1: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    T_1: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    D_1: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    A_2: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    T_2: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    D_2: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)


def XYZ_to_ATD95(XYZ: ArrayLike,
                 XYZ_0: ArrayLike,
                 Y_0: FloatingOrArrayLike,
                 k_1: FloatingOrArrayLike,
                 k_2: FloatingOrArrayLike,
                 sigma: FloatingOrArrayLike = 300) -> CAM_Specification_ATD95:
    """
    Computes the *ATD (1995)* colour vision model correlates.

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
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_0``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +-------------------------------+-----------------------+---------------+
    | **Range**                     | **Scale - Reference** | **Scale - 1** |
    +===============================+=======================+===============+
    | ``CAM_Specification_ATD95.h`` | [0, 360]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+

    -   For unrelated colors, there is only self-adaptation and :math:`k_1` is
        set to 1.0 while :math:`k_2` is set to 0.0. For related colors such as
        typical colorimetric applications, :math:`k_1` is set to 0.0 and
        :math:`k_2` is set to a value between 15 and 50 *(Guth, 1995)*.

    References
    ----------
    :cite:`Fairchild2013v`, :cite:`Guth1995a`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_0 = np.array([95.05, 100.00, 108.88])
    >>> Y_0 = 318.31
    >>> k_1 = 0.0
    >>> k_2 = 50.0
    >>> XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)  # doctest: +ELLIPSIS
    CAM_Specification_ATD95(h=1.9089869..., C=1.2064060..., Q=0.1814003..., \
A_1=0.1787931... T_1=0.0286942..., D_1=0.0107584..., A_2=0.0192182..., \
T_2=0.0205377..., D_2=0.0107584...)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_0 = to_domain_100(XYZ_0)
    Y_0 = as_float_array(Y_0)
    k_1 = as_float_array(k_1)
    k_2 = as_float_array(k_2)
    sigma = as_float_array(sigma)

    XYZ = luminance_to_retinal_illuminance(XYZ, Y_0)
    XYZ_0 = luminance_to_retinal_illuminance(XYZ_0, Y_0)

    # Computing adaptation model.
    LMS = XYZ_to_LMS_ATD95(XYZ)
    XYZ_a = k_1[..., np.newaxis] * XYZ + k_2[..., np.newaxis] * XYZ_0
    LMS_a = XYZ_to_LMS_ATD95(XYZ_a)

    LMS_g = LMS * (sigma[..., np.newaxis] / (sigma[..., np.newaxis] + LMS_a))

    # Computing opponent colour dimensions.
    A_1, T_1, D_1, A_2, T_2, D_2 = tsplit(opponent_colour_dimensions(LMS_g))

    # Computing the correlate of *brightness* :math:`Br`.
    Br = spow(A_1 ** 2 + T_1 ** 2 + D_1 ** 2, 0.5)

    # Computing the correlate of *saturation* :math:`C`.
    C = spow(T_2 ** 2 + D_2 ** 2, 0.5) / A_2

    # Computing the *hue* :math:`H`. Note that the reference does not take the
    # modulus of the :math:`H`, thus :math:`H` can exceed 360 degrees.
    H = T_2 / D_2

    return CAM_Specification_ATD95(
        as_float(from_range_degrees(H)),
        C,
        Br,
        A_1,
        T_1,
        D_1,
        A_2,
        T_2,
        D_2,
    )


def luminance_to_retinal_illuminance(XYZ: ArrayLike,
                                     Y_c: FloatingOrArrayLike) -> NDArray:
    """
    Converts from luminance in :math:`cd/m^2` to retinal illuminance in
    trolands.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    Y_c
        Absolute adapting field luminance in :math:`cd/m^2`.

    Returns
    -------
    :class:`numpy.ndarray`
        Converted *CIE XYZ* tristimulus values in trolands.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> Y_0 = 318.31
    >>> luminance_to_retinal_illuminance(XYZ, Y_0)  # doctest: +ELLIPSIS
    array([ 479.4445924...,  499.3174313...,  534.5631673...])
    """

    XYZ = as_float_array(XYZ)
    Y_c = as_float_array(Y_c)

    return as_float_array(18 * spow(Y_c[..., np.newaxis] * XYZ / 100, 0.8))


def XYZ_to_LMS_ATD95(XYZ: ArrayLike) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *LMS* cone responses.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *LMS* cone responses.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_to_LMS_ATD95(XYZ)  # doctest: +ELLIPSIS
    array([ 6.2283272...,  7.4780666...,  3.8859772...])
    """

    LMS = vector_dot([
        [0.2435, 0.8524, -0.0516],
        [-0.3954, 1.1642, 0.0837],
        [0.0000, 0.0400, 0.6225],
    ], XYZ)
    LMS *= np.array([0.66, 1.0, 0.43])

    LMS_p = spow(LMS, 0.7)
    LMS_p += np.array([0.024, 0.036, 0.31])

    return as_float_array(LMS_p)


def opponent_colour_dimensions(LMS_g: ArrayLike) -> NDArray:
    """
    Returns opponent colour dimensions from given post adaptation cone signals.

    Parameters
    ----------
    LMS_g
        Post adaptation cone signals.

    Returns
    -------
    :class:`numpy.ndarray`
        Opponent colour dimensions.

    Examples
    --------
    >>> LMS_g = np.array([6.95457922, 7.08945043, 6.44069316])
    >>> opponent_colour_dimensions(LMS_g)  # doctest: +ELLIPSIS
    array([ 0.1787931...,  0.0286942...,  0.0107584...,  0.0192182..., ...])
    """

    L_g, M_g, S_g = tsplit(LMS_g)

    A_1i = 3.57 * L_g + 2.64 * M_g
    T_1i = 7.18 * L_g - 6.21 * M_g
    D_1i = -0.7 * L_g + 0.085 * M_g + S_g
    A_2i = 0.09 * A_1i
    T_2i = 0.43 * T_1i + 0.76 * D_1i
    D_2i = D_1i

    A_1 = final_response(A_1i)
    T_1 = final_response(T_1i)
    D_1 = final_response(D_1i)
    A_2 = final_response(A_2i)
    T_2 = final_response(T_2i)
    D_2 = final_response(D_2i)

    return tstack([A_1, T_1, D_1, A_2, T_2, D_2])


def final_response(value: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Returns the final response of given opponent colour dimension.

    Parameters
    ----------
    value
         Opponent colour dimension.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Final response of opponent colour dimension.

    Examples
    --------
    >>> final_response(43.54399695501678)  # doctest: +ELLIPSIS
    0.1787931...
    """

    value = as_float_array(value)

    return as_float(value / (200 + np.abs(value)))
