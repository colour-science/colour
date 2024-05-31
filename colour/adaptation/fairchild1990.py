"""
Fairchild (1990) Chromatic Adaptation Model
===========================================

Defines the *Fairchild (1990)* chromatic adaptation model objects:

-   :func:`colour.adaptation.chromatic_adaptation_Fairchild1990`

References
----------
-   :cite:`Fairchild1991a` : Fairchild, M. D. (1991). Formulation and testing
    of an incomplete-chromatic-adaptation model. Color Research & Application,
    16(4), 243-250. doi:10.1002/col.5080160406
-   :cite:`Fairchild2013s` : Fairchild, M. D. (2013). FAIRCHILD'S 1990 MODEL.
    In Color Appearance Models (3rd ed., pp. 4418-4495). Wiley. ISBN:B00DAYO8E2
"""

from __future__ import annotations

import numpy as np

from colour.adaptation import CAT_VON_KRIES
from colour.algebra import sdiv, sdiv_mode, spow, vecmul
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import (
    as_float_array,
    from_range_100,
    ones,
    row_as_diagonal,
    to_domain_100,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_XYZ_TO_RGB_FAIRCHILD1990",
    "MATRIX_RGB_TO_XYZ_FAIRCHILD1990",
    "chromatic_adaptation_Fairchild1990",
    "XYZ_to_RGB_Fairchild1990",
    "RGB_to_XYZ_Fairchild1990",
    "degrees_of_adaptation",
]

MATRIX_XYZ_TO_RGB_FAIRCHILD1990: NDArrayFloat = CAT_VON_KRIES
"""
*Fairchild (1990)* colour appearance model *CIE XYZ* tristimulus values to cone
responses matrix.
"""

MATRIX_RGB_TO_XYZ_FAIRCHILD1990: NDArrayFloat = np.linalg.inv(CAT_VON_KRIES)
"""
*Fairchild (1990)* colour appearance model cone responses to *CIE XYZ*
tristimulus values matrix.
"""


def chromatic_adaptation_Fairchild1990(
    XYZ_1: ArrayLike,
    XYZ_n: ArrayLike,
    XYZ_r: ArrayLike,
    Y_n: ArrayLike,
    discount_illuminant: bool = False,
) -> NDArrayFloat:
    """
    Adapt given stimulus *CIE XYZ_1* tristimulus values from test viewing
    conditions to reference viewing conditions using *Fairchild (1990)*
    chromatic adaptation model.

    Parameters
    ----------
    XYZ_1
        *CIE XYZ_1* tristimulus values of test sample / stimulus.
    XYZ_n
        Test viewing condition *CIE XYZ_n* tristimulus values of whitepoint.
    XYZ_r
        Reference viewing condition *CIE XYZ_r* tristimulus values of
        whitepoint.
    Y_n
        Luminance :math:`Y_n` of test adapting stimulus in :math:`cd/m^2`.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    :class:`numpy.ndarray`
        Adapted *CIE XYZ_2* tristimulus values of stimulus.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_1``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_n``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_r``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_2``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild1991a`, :cite:`Fairchild2013s`

    Examples
    --------
    >>> XYZ_1 = np.array([19.53, 23.07, 24.97])
    >>> XYZ_n = np.array([111.15, 100.00, 35.20])
    >>> XYZ_r = np.array([94.81, 100.00, 107.30])
    >>> Y_n = 200
    >>> chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n)
    ... # doctest: +ELLIPSIS
    array([ 23.3252634...,  23.3245581...,  76.1159375...])
    """

    XYZ_1 = to_domain_100(XYZ_1)
    XYZ_n = to_domain_100(XYZ_n)
    XYZ_r = to_domain_100(XYZ_r)
    Y_n = as_float_array(Y_n)

    LMS_1 = vecmul(MATRIX_XYZ_TO_RGB_FAIRCHILD1990, XYZ_1)
    LMS_n = vecmul(MATRIX_XYZ_TO_RGB_FAIRCHILD1990, XYZ_n)
    LMS_r = vecmul(MATRIX_XYZ_TO_RGB_FAIRCHILD1990, XYZ_r)

    p_LMS = degrees_of_adaptation(LMS_1, Y_n, discount_illuminant=discount_illuminant)

    a_LMS_1 = p_LMS / LMS_n
    a_LMS_2 = p_LMS / LMS_r

    A_1 = row_as_diagonal(a_LMS_1)
    A_2 = row_as_diagonal(a_LMS_2)

    LMSp_1 = vecmul(A_1, LMS_1)

    c = 0.219 - 0.0784 * np.log10(Y_n)
    C = row_as_diagonal(tstack([c, c, c]))

    LMS_a = vecmul(C, LMSp_1)
    LMSp_2 = vecmul(np.linalg.inv(C), LMS_a)

    LMS_c = vecmul(np.linalg.inv(A_2), LMSp_2)
    XYZ_c = vecmul(MATRIX_RGB_TO_XYZ_FAIRCHILD1990, LMS_c)

    return from_range_100(XYZ_c)


def XYZ_to_RGB_Fairchild1990(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to cone responses.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        Cone responses.

    Examples
    --------
    >>> XYZ = np.array([19.53, 23.07, 24.97])
    >>> XYZ_to_RGB_Fairchild1990(XYZ)  # doctest: +ELLIPSIS
    array([ 22.1231935...,  23.6054224...,  22.9279534...])
    """

    return vecmul(MATRIX_XYZ_TO_RGB_FAIRCHILD1990, XYZ)


def RGB_to_XYZ_Fairchild1990(RGB: ArrayLike) -> NDArrayFloat:
    """
    Convert from cone responses to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB
        Cone responses.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> RGB = np.array([22.12319350, 23.60542240, 22.92795340])
    >>> RGB_to_XYZ_Fairchild1990(RGB)  # doctest: +ELLIPSIS
    array([ 19.53,  23.07,  24.97])
    """

    return vecmul(MATRIX_RGB_TO_XYZ_FAIRCHILD1990, RGB)


def degrees_of_adaptation(
    LMS: ArrayLike,
    Y_n: ArrayLike,
    v: ArrayLike = 1 / 3,
    discount_illuminant: bool = False,
) -> NDArrayFloat:
    """
    Compute the degrees of adaptation :math:`p_L`, :math:`p_M` and
    :math:`p_S`.

    Parameters
    ----------
    LMS
        Cone responses.
    Y_n
        Luminance :math:`Y_n` of test adapting stimulus in :math:`cd/m^2`.
    v
        Exponent :math:`v`.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    :class:`numpy.ndarray`
        Degrees of adaptation :math:`p_L`, :math:`p_M` and :math:`p_S`.

    Examples
    --------
    >>> LMS = np.array([20.00052060, 19.99978300, 19.99883160])
    >>> Y_n = 31.83
    >>> degrees_of_adaptation(LMS, Y_n)  # doctest: +ELLIPSIS
    array([ 0.9799324...,  0.9960035...,  1.0233041...])
    >>> degrees_of_adaptation(LMS, Y_n, 1 / 3, True)
    array([ 1.,  1.,  1.])
    """

    LMS = as_float_array(LMS)
    if discount_illuminant:
        return ones(LMS.shape)

    Y_n = as_float_array(Y_n)
    v = as_float_array(v)

    # E illuminant.
    LMS_E = vecmul(CAT_VON_KRIES, ones(LMS.shape))

    Ye_n = spow(Y_n, v)

    def m_E(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:
        """Compute the :math:`m_E` term."""

        return (3 * x / y) / np.sum(x / y, axis=-1)[..., None]

    def P_c(x: NDArrayFloat) -> NDArrayFloat:
        """Compute the :math:`P_L`, :math:`P_M` or :math:`P_S` terms."""

        return sdiv(
            1 + Ye_n[..., None] + x,
            1 + Ye_n[..., None] + sdiv(1, x),
        )

    with sdiv_mode():
        p_LMS = P_c(m_E(LMS, LMS_E))

    return p_LMS
