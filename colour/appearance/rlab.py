# -*- coding: utf-8 -*-
"""
RLAB Colour Appearance Model
============================

Defines the *RLAB* colour appearance model objects:

-   :attr:`colour.VIEWING_CONDITIONS_RLAB`
-   :attr:`colour.D_FACTOR_RLAB`
-   :class:`colour.CAM_Specification_RLAB`
-   :func:`colour.XYZ_to_RLAB`

References
----------
-   :cite:`Fairchild1996a` : Fairchild, M. D. (1996). Refinement of the RLAB
    color space. Color Research & Application, 21(5), 338-346.
    doi:10.1002/(SICI)1520-6378(199610)21:5<338::AID-COL3>3.0.CO;2-Z
-   :cite:`Fairchild2013w` : Fairchild, M. D. (2013). The RLAB Model. In Color
    Appearance Models (3rd ed., pp. 5563-5824). Wiley. ISBN:B00DAYO8E2
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union

from colour.algebra import matrix_dot, spow, vector_dot
from colour.appearance.hunt import MATRIX_XYZ_TO_HPE, XYZ_to_rgb
from colour.utilities import (
    CaseInsensitiveMapping,
    MixinDataclassArray,
    as_float,
    as_float_array,
    from_range_degrees,
    row_as_diagonal,
    to_domain_100,
    tsplit,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_R',
    'VIEWING_CONDITIONS_RLAB',
    'D_FACTOR_RLAB',
    'CAM_ReferenceSpecification_RLAB',
    'CAM_Specification_RLAB',
    'XYZ_to_RLAB',
]

MATRIX_R = np.array([
    [1.9569, -1.1882, 0.2313],
    [0.3612, 0.6388, 0.0000],
    [0.0000, 0.0000, 1.0000],
])
"""
*RLAB* colour appearance model precomputed helper matrix.

MATRIX_R : array_like, (3, 3)
"""

VIEWING_CONDITIONS_RLAB = CaseInsensitiveMapping({
    'Average': 1 / 2.3,
    'Dim': 1 / 2.9,
    'Dark': 1 / 3.5
})
VIEWING_CONDITIONS_RLAB.__doc__ = """
Reference *RLAB* colour appearance model viewing conditions.

References
----------
:cite:`Fairchild1996a`, :cite:`Fairchild2013w`

VIEWING_CONDITIONS_RLAB : CaseInsensitiveMapping
    **{'Average', 'Dim', 'Dark'}**
"""

D_FACTOR_RLAB = CaseInsensitiveMapping({
    'Hard Copy Images': 1,
    'Soft Copy Images': 0,
    'Projected Transparencies, Dark Room': 0.5
})
D_FACTOR_RLAB.__doc__ = """
*RLAB* colour appearance model *Discounting-the-Illuminant* factor values.

References
----------
:cite:`Fairchild1996a`, :cite:`Fairchild2013w`

D_FACTOR_RLAB : CaseInsensitiveMapping
    **{'Hard Copy Images',
    'Soft Copy Images',
    'Projected Transparencies, Dark Room'}**

Aliases:

-   'hard_cp_img': 'Hard Copy Images'
-   'soft_cp_img': 'Soft Copy Images'
-   'projected_dark': 'Projected Transparencies, Dark Room'
"""
D_FACTOR_RLAB['hard_cp_img'] = D_FACTOR_RLAB['Hard Copy Images']
D_FACTOR_RLAB['soft_cp_img'] = D_FACTOR_RLAB['Soft Copy Images']
D_FACTOR_RLAB['projected_dark'] = (
    D_FACTOR_RLAB['Projected Transparencies, Dark Room'])


@dataclass
class CAM_ReferenceSpecification_RLAB(MixinDataclassArray):
    """
    Defines the *RLAB* colour appearance model reference specification.

    This specification has field names consistent with *Fairchild (2013)*
    reference.

    Parameters
    ----------
    LR : numeric or array_like
        Correlate of *Lightness* :math:`L^R`.
    CR : numeric or array_like
        Correlate of *achromatic chroma* :math:`C^R`.
    hR : numeric or array_like
        *Hue* angle :math:`h^R` in degrees.
    sR : numeric or array_like
        Correlate of *saturation* :math:`s^R`.
    HR : numeric or array_like
        *Hue* :math:`h` composition :math:`H^R`.
    aR : numeric or array_like
        Red-green chromatic response :math:`a^R`.
    bR : numeric or array_like
        Yellow-blue chromatic response :math:`b^R`.

    References
    ----------
    :cite:`Fairchild1996a`, :cite:`Fairchild2013w`
    """

    LR: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    CR: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    hR: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    sR: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    HR: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    aR: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    bR: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)


@dataclass
class CAM_Specification_RLAB(MixinDataclassArray):
    """
    Defines the *RLAB* colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    *Fairchild (2013)* reference.

    Parameters
    ----------
    J : numeric or array_like
        Correlate of *Lightness* :math:`L^R`.
    C : numeric or array_like
        Correlate of *achromatic chroma* :math:`C^R`.
    h : numeric or array_like
        *Hue* angle :math:`h^R` in degrees.
    s : numeric or array_like
        Correlate of *saturation* :math:`s^R`.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H^C`.
    a : numeric or array_like
        Red-green chromatic response :math:`a^R`.
    b : numeric or array_like
        Yellow-blue chromatic response :math:`b^R`.

    Notes
    -----
    -   This specification is the one used in the current model implementation.

    References
    ----------
    :cite:`Fairchild1996a`, :cite:`Fairchild2013w`
    """

    J: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    C: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    h: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    s: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    HC: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    a: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    b: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)


def XYZ_to_RLAB(XYZ,
                XYZ_n,
                Y_n,
                sigma=VIEWING_CONDITIONS_RLAB['Average'],
                D=D_FACTOR_RLAB['Hard Copy Images']):
    """
    Computes the *RLAB* model color appearance correlates.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_n : array_like
        *CIE XYZ* tristimulus values of reference white.
    Y_n : numeric or array_like
        Absolute adapting luminance in :math:`cd/m^2`.
    sigma : numeric or array_like, optional
        Relative luminance of the surround, see
        :attr:`colour.VIEWING_CONDITIONS_RLAB` for reference.
    D : numeric or array_like, optional
        *Discounting-the-Illuminant* factor normalised to domain [0, 1].

    Returns
    -------
    CAM_Specification_RLAB
        *RLAB* colour appearance model specification.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_n``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------------------------+-----------------------\
+---------------+
    | **Range**                    | **Scale - Reference** \
| **Scale - 1** |
    +==============================+=======================\
+===============+
    | ``CAM_Specification_RLAB.h`` | [0, 360]              \
| [0, 1]        |
    +------------------------------+-----------------------\
+---------------+

    References
    ----------
    :cite:`Fairchild1996a`, :cite:`Fairchild2013w`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_n = np.array([109.85, 100, 35.58])
    >>> Y_n = 31.83
    >>> sigma = VIEWING_CONDITIONS_RLAB['Average']
    >>> D = D_FACTOR_RLAB['Hard Copy Images']
    >>> XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)  # doctest: +ELLIPSIS
    CAM_Specification_RLAB(J=49.8347069..., C=54.8700585..., \
h=286.4860208..., s=1.1010410..., HC=None, a=15.5711021..., \
b=-52.6142956...)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_n = to_domain_100(XYZ_n)
    Y_n = as_float_array(Y_n)
    D = as_float_array(D)
    sigma = as_float_array(sigma)

    # Converting to cone responses.
    LMS_n = XYZ_to_rgb(XYZ_n)

    # Computing the :math:`A` matrix.
    LMS_l_E = (3 * LMS_n) / np.sum(LMS_n, axis=-1)[..., np.newaxis]
    LMS_p_L = ((1 + spow(Y_n[..., np.newaxis], 1 / 3) + LMS_l_E) /
               (1 + spow(Y_n[..., np.newaxis], 1 / 3) + (1 / LMS_l_E)))
    LMS_a_L = (LMS_p_L + D[..., np.newaxis] * (1 - LMS_p_L)) / LMS_n

    aR = row_as_diagonal(LMS_a_L)
    M = matrix_dot(matrix_dot(MATRIX_R, aR), MATRIX_XYZ_TO_HPE)
    XYZ_ref = vector_dot(M, XYZ)

    X_ref, Y_ref, Z_ref = tsplit(XYZ_ref)

    # Computing the correlate of *Lightness* :math:`L^R`.
    LR = 100 * spow(Y_ref, sigma)

    # Computing opponent colour dimensions :math:`a^R` and :math:`b^R`.
    aR = 430 * (spow(X_ref, sigma) - spow(Y_ref, sigma))
    bR = 170 * (spow(Y_ref, sigma) - spow(Z_ref, sigma))

    # Computing the *hue* angle :math:`h^R`.
    hR = np.degrees(np.arctan2(bR, aR)) % 360
    # TODO: Implement hue composition computation.

    # Computing the correlate of *chroma* :math:`C^R`.
    CR = np.hypot(aR, bR)

    # Computing the correlate of *saturation* :math:`s^R`.
    sR = CR / LR

    return CAM_Specification_RLAB(
        LR,
        CR,
        as_float(from_range_degrees(hR)),
        sR,
        None,
        aR,
        bR,
    )
