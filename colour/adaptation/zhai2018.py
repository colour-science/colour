# -*- coding: utf-8 -*-
"""
Zhai and Luo (2018) Chromatic Adaptation Model
==============================================

Defines the *Zhai and Luo (2018)* chromatic adaptation model object:

-   :func:`colour.adaptation.chromatic_adaptation_Zhai2018`

References
----------
-   :cite:`Zhai2018` : Zhai, Q., & Luo, M. R. (2018). Study of chromatic
    adaptation via neutral white matches on different viewing media. Optics
    Express, 26(6), 7724. doi:10.1364/OE.26.007724
"""

import numpy as np

from colour.algebra import vector_dot
from colour.adaptation import CHROMATIC_ADAPTATION_TRANSFORMS
from colour.hints import (
    ArrayLike,
    FloatingOrArrayLike,
    Literal,
    NDArray,
    Union,
)
from colour.utilities import (
    as_float_array,
    from_range_100,
    to_domain_100,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'chromatic_adaptation_Zhai2018',
]


def chromatic_adaptation_Zhai2018(
        XYZ_b: ArrayLike,
        XYZ_wb: ArrayLike,
        XYZ_wd: ArrayLike,
        D_b: FloatingOrArrayLike = 1,
        D_d: FloatingOrArrayLike = 1,
        XYZ_wo: ArrayLike = np.array([1, 1, 1]),
        chromatic_adaptation_transform: Union[Literal['CAT02', 'CAT16'],
                                              str] = 'CAT02') -> NDArray:
    """
    Adapts given sample colour :math:`XYZ_{\\beta}` tristimulus values from
    input viewing conditions under :math:`\\beta` illuminant to output viewing
    conditions under :math:`\\delta` illuminant using *Zhai and Luo (2018)*
    chromatic adaptation model.

    According to the definition of :math:`D`, a one-step CAT such as CAT02 can
    only be used to transform colors from an incomplete adapted field into a
    complete adapted field. When CAT02 are used to transform an incomplete to
    incomplete case, :math:`D` has no baseline level to refer to.
    *Smet et al. (2017)* proposed a new concept of two-step CAT to replace the
    present CATs such as CAT02 with only one-step transform in order to define
    :math:`D` more clearly. A two-step CAT involves an illuminant representing
    the baseline states between the test and reference illuminants for the
    calculation. In the first step the test color is transformed from test
    illuminant to the baseline illuminant (:math:`BI`), and it is then
    transformed to the reference illuminant Degrees of adaptation under the
    other illuminants should be calculated relative to the adaptation under the
    :math:`BI`. When :math:`D` becomes lower towards zero, the adaptation point
    of the observer moves towards the :math:`BI`. Therefore, the chromaticity
    of the :math:`BI` should be an intrinsic property of the human vision
    system.

    Parameters
    ----------
    XYZ_b
        Sample colour :math:`XYZ_{\\beta}` under input illuminant
        :math:`\\beta`.
    XYZ_wb
        Input illuminant :math:`\\beta`.
    XYZ_wd
        Output illuminant :math:`\\delta`.
    D_b
        Degree of adaptation :math:`D_{\\beta}` of input illuminant
        :math:`\\beta`.
    D_d
        Degree of adaptation :math:`D_{\\delta}` of output illuminant
        :math:`\\delta`.
    XYZ_wo
        Baseline illuminant (:math:`BI`) :math:`o`.
    chromatic_adaptation_transform
        Chromatic adaptation transform.

    Returns
    -------
    :class:`numpy.ndarray`
        Sample corresponding colour :math:`XYZ_{\\delta}` tristimulus values
        under output illuminant :math:`D_{\\delta}`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_b``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wb`` | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wd`` | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wo`` | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_d``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Zhai2018`

    Examples
    --------
    >>> XYZ_b = np.array([48.900, 43.620, 6.250])
    >>> XYZ_wb = np.array([109.850, 100, 35.585])
    >>> XYZ_wd = np.array([95.047, 100, 108.883])
    >>> D_b = 0.9407
    >>> D_d = 0.9800
    >>> XYZ_wo = np.array([100, 100, 100])
    >>> chromatic_adaptation_Zhai2018(
    ...     XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d, XYZ_wo)  # doctest: +ELLIPSIS
    array([ 39.1856164...,  42.1546179...,  19.2367203...])
    >>> XYZ_d = np.array([39.18561644, 42.15461798, 19.23672036])
    >>> chromatic_adaptation_Zhai2018(
    ...     XYZ_d, XYZ_wd, XYZ_wb, D_d, D_b, XYZ_wo)  # doctest: +ELLIPSIS
    array([ 48.9 ,  43.62,   6.25])
    """

    XYZ_b = to_domain_100(XYZ_b)
    XYZ_wb = to_domain_100(XYZ_wb)
    XYZ_wd = to_domain_100(XYZ_wd)
    XYZ_wo = to_domain_100(XYZ_wo)
    D_b = as_float_array(D_b)
    D_d = as_float_array(D_d)

    Y_wb = XYZ_wb[..., 1][..., np.newaxis]
    Y_wd = XYZ_wd[..., 1][..., np.newaxis]
    Y_wo = XYZ_wo[..., 1][..., np.newaxis]

    chromatic_adaptation_transform = validate_method(
        chromatic_adaptation_transform, ['CAT02', 'CAT16'])
    M = CHROMATIC_ADAPTATION_TRANSFORMS[chromatic_adaptation_transform]

    RGB_b = vector_dot(M, XYZ_b)
    RGB_wb = vector_dot(M, XYZ_wb)
    RGB_wd = vector_dot(M, XYZ_wd)
    RGB_wo = vector_dot(M, XYZ_wo)

    D_RGB_b = D_b * (Y_wb / Y_wo) * (RGB_wo / RGB_wb) + 1 - D_b
    D_RGB_d = D_d * (Y_wd / Y_wo) * (RGB_wo / RGB_wd) + 1 - D_d

    D_RGB = D_RGB_b / D_RGB_d

    RGB_d = D_RGB * RGB_b

    XYZ_d = vector_dot(np.linalg.inv(M), RGB_d)

    return from_range_100(XYZ_d)
