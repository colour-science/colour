"""
Li (2025) Chromatic Adaptation Model
====================================

Define the *Li (2025)* chromatic adaptation model  for predicting corresponding
colours under different viewing conditions.

-   :func:`colour.adaptation.chromatic_adaptation_Li2025`

References
----------
-   :cite:`Li2025` : Li, M. (2025). One Step CAT16 Chromatic Adaptation
    Transform. https://github.com/colour-science/colour/pull/1349\
#issuecomment-3058339414
"""

from __future__ import annotations

import typing

import numpy as np

from colour.adaptation import CAT_CAT16
from colour.algebra import sdiv, sdiv_mode, vecmul

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Domain100, NDArrayFloat, Range100

from colour.utilities import (
    as_float_array,
    ones,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CAT_CAT16_INVERSE",
    "chromatic_adaptation_Li2025",
]

CAT_CAT16_INVERSE: NDArrayFloat = np.linalg.inv(CAT_CAT16)
"""Inverse adaptation matrix :math:`M^{-1}_{CAT16}` for *Li (2025)* method."""


def chromatic_adaptation_Li2025(
    XYZ_s: Domain100,
    XYZ_ws: Domain100,
    XYZ_wd: Domain100,
    L_A: ArrayLike,
    F_surround: ArrayLike,
    discount_illuminant: bool = False,
) -> Range100:
    """
    Adapt the specified stimulus *CIE XYZ* tristimulus values from test
    viewing conditions to reference viewing conditions using the
    *Li (2025)* chromatic adaptation model.

    This one-step chromatic adaptation transform is based on *CAT16* and
    includes the degree of adaptation calculation from the viewing conditions
    as specified by *CIECAM02* colour appearance model.

    Parameters
    ----------
    XYZ_s
        *CIE XYZ* tristimulus values of stimulus under source illuminant.
    XYZ_ws
        *CIE XYZ* tristimulus values of source whitepoint.
    XYZ_wd
        *CIE XYZ* tristimulus values of destination whitepoint.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    F_surround
        Maximum degree of adaptation :math:`F` from surround viewing
        conditions.
    discount_illuminant
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values of the stimulus corresponding colour.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_s``  | 100                   | 1             |
    +------------+-----------------------+---------------+
    | ``XYZ_ws`` | 100                   | 1             |
    +------------+-----------------------+---------------+
    | ``XYZ_wd`` | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_a``  | 100                   | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Li2025`

    Examples
    --------
    >>> XYZ_s = np.array([48.900, 43.620, 6.250])
    >>> XYZ_ws = np.array([109.850, 100, 35.585])
    >>> XYZ_wd = np.array([95.047, 100, 108.883])
    >>> L_A = 318.31
    >>> F_surround = 1.0
    >>> chromatic_adaptation_Li2025(  # doctest: +ELLIPSIS
    ...     XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround
    ... )
    array([ 40.0072581...,  43.7014895...,  21.3290293...])
    """

    XYZ_s = as_float_array(XYZ_s)
    XYZ_ws = as_float_array(XYZ_ws)
    XYZ_wd = as_float_array(XYZ_wd)
    L_A = as_float_array(L_A)
    F_surround = as_float_array(F_surround)

    LMS_s = vecmul(CAT_CAT16, XYZ_s)
    LMS_w_s = vecmul(CAT_CAT16, XYZ_ws)
    LMS_w_d = vecmul(CAT_CAT16, XYZ_wd)

    Y_w_s = XYZ_ws[..., 1] if XYZ_ws.ndim > 1 else XYZ_ws[1]
    Y_w_d = XYZ_wd[..., 1] if XYZ_wd.ndim > 1 else XYZ_wd[1]

    if discount_illuminant:
        D = ones(L_A.shape)
    else:
        D = F_surround * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92))
        D = np.clip(D, 0, 1)

    D = np.atleast_1d(D)[..., None] if LMS_s.ndim > 1 else D

    with sdiv_mode():
        Y_ratio = sdiv(Y_w_s, Y_w_d)
        Y_ratio = Y_ratio[..., None] if LMS_s.ndim > 1 else Y_ratio
        LMS_a = LMS_s * (D * Y_ratio * sdiv(LMS_w_d, LMS_w_s) + (1 - D))

    return vecmul(CAT_CAT16_INVERSE, LMS_a)
