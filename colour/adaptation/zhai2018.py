"""
Zhai and Luo (2018) Chromatic Adaptation Model
==============================================

Define the *Zhai and Luo (2018)* two-step chromatic adaptation for predicting
corresponding colours under different viewing conditions.

-   :func:`colour.adaptation.chromatic_adaptation_Zhai2018`

References
----------
-   :cite:`Zhai2018` : Zhai, Q., & Luo, M. R. (2018). Study of chromatic
    adaptation via neutral white matches on different viewing media. Optics
    Express, 26(6), 7724. doi:10.1364/OE.26.007724
"""

from __future__ import annotations

import typing

import numpy as np

from colour.adaptation import CHROMATIC_ADAPTATION_TRANSFORMS
from colour.algebra import vecmul

if typing.TYPE_CHECKING:
    from colour.hints import Literal

from colour.hints import (  # noqa: TC001
    ArrayLike,
    Domain100,
    Range100,
)
from colour.utilities import (
    as_float_array,
    from_range_100,
    get_domain_range_scale,
    optional,
    to_domain_100,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "chromatic_adaptation_Zhai2018",
]


def chromatic_adaptation_Zhai2018(
    XYZ_b: Domain100,
    XYZ_wb: Domain100,
    XYZ_wd: Domain100,
    D_b: ArrayLike = 1,
    D_d: ArrayLike = 1,
    XYZ_wo: ArrayLike | None = None,
    transform: Literal["CAT02", "CAT16"] | str = "CAT02",
) -> Range100:
    """
    Adapt the specified stimulus *CIE XYZ* tristimulus values from test
    viewing conditions to reference viewing conditions using the
    *Zhai and Luo (2018)* chromatic adaptation model.

    According to the definition of :math:`D`, a one-step chromatic adaptation
    transform (CAT) such as CAT02 can only transform colours from an
    incomplete adapted field into a complete adapted field. When CAT02 is
    used to transform from incomplete to incomplete adaptation, :math:`D` has
    no baseline level to refer to. *Smet et al. (2017)* proposed a two-step
    CAT concept to replace existing one-step transforms such as CAT02,
    providing a clearer definition of :math:`D`. A two-step CAT involves a
    baseline illuminant (BI) representing the baseline state between the test
    and reference illuminants. In the first step, the test colour is
    transformed from the test illuminant to the baseline illuminant
    (:math:`BI`), then subsequently transformed to the reference illuminant.
    Degrees of adaptation under other illuminants are calculated relative to
    the adaptation under the :math:`BI`. As :math:`D` approaches zero, the
    observer's adaptation point moves towards the :math:`BI`. Therefore, the
    chromaticity of the :math:`BI` is an intrinsic property of the human
    visual system.

    Parameters
    ----------
    XYZ_b
        Sample colour :math:`XYZ_{\\beta}` tristimulus values under input
        illuminant :math:`\\beta`.
    XYZ_wb
        Input illuminant :math:`\\beta` tristimulus values.
    XYZ_wd
        Output illuminant :math:`\\delta` tristimulus values.
    D_b
        Degree of adaptation :math:`D_{\\beta}` of input illuminant
        :math:`\\beta`.
    D_d
        Degree of adaptation :math:`D_{\\delta}` of output illuminant
        :math:`\\delta`.
    XYZ_wo
        Baseline illuminant (:math:`BI`) :math:`o` tristimulus values.
    transform
        Chromatic adaptation transform matrix.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values of the stimulus corresponding colour.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_b``  | 100                   | 1             |
    +------------+-----------------------+---------------+
    | ``XYZ_wb`` | 100                   | 1             |
    +------------+-----------------------+---------------+
    | ``XYZ_wd`` | 100                   | 1             |
    +------------+-----------------------+---------------+
    | ``XYZ_wo`` | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_d``  | 100                   | 1             |
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
    ...     XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d, XYZ_wo
    ... )  # doctest: +ELLIPSIS
    array([ 39.1856164...,  42.1546179...,  19.2367203...])
    >>> XYZ_d = np.array([39.18561644, 42.15461798, 19.23672036])
    >>> chromatic_adaptation_Zhai2018(
    ...     XYZ_d, XYZ_wd, XYZ_wb, D_d, D_b, XYZ_wo
    ... )  # doctest: +ELLIPSIS
    array([ 48.9 ,  43.62,   6.25])
    """

    XYZ_b = to_domain_100(XYZ_b)
    XYZ_wb = to_domain_100(XYZ_wb)
    XYZ_wd = to_domain_100(XYZ_wd)
    XYZ_wo = to_domain_100(
        optional(
            XYZ_wo,
            np.array([1, 1, 1])
            if get_domain_range_scale() == "reference"
            else np.array([0.01, 0.01, 0.01]),
        )
    )
    D_b = as_float_array(D_b)
    D_d = as_float_array(D_d)

    Y_wb = XYZ_wb[..., 1][..., None]
    Y_wd = XYZ_wd[..., 1][..., None]
    Y_wo = XYZ_wo[..., 1][..., None]

    transform = validate_method(transform, ("CAT02", "CAT16"))
    M = CHROMATIC_ADAPTATION_TRANSFORMS[transform]

    RGB_b = vecmul(M, XYZ_b)
    RGB_wb = vecmul(M, XYZ_wb)
    RGB_wd = vecmul(M, XYZ_wd)
    RGB_wo = vecmul(M, XYZ_wo)

    D_RGB_b = D_b * (Y_wb / Y_wo) * (RGB_wo / RGB_wb) + 1 - D_b
    D_RGB_d = D_d * (Y_wd / Y_wo) * (RGB_wo / RGB_wd) + 1 - D_d

    D_RGB = D_RGB_b / D_RGB_d

    RGB_d = D_RGB * RGB_b

    XYZ_d = vecmul(np.linalg.inv(M), RGB_d)

    return from_range_100(XYZ_d)
