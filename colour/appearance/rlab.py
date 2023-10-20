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

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from colour.algebra import matrix_dot, sdiv, sdiv_mode, spow, vector_dot
from colour.appearance.hunt import MATRIX_XYZ_TO_HPE, XYZ_to_rgb
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArray,
    as_float,
    as_float_array,
    from_range_degrees,
    row_as_diagonal,
    to_domain_100,
    tsplit,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_R",
    "VIEWING_CONDITIONS_RLAB",
    "D_FACTOR_RLAB",
    "CAM_ReferenceSpecification_RLAB",
    "CAM_Specification_RLAB",
    "XYZ_to_RLAB",
]

MATRIX_R: NDArrayFloat = np.array(
    [
        [1.9569, -1.1882, 0.2313],
        [0.3612, 0.6388, 0.0000],
        [0.0000, 0.0000, 1.0000],
    ]
)
"""*RLAB* colour appearance model precomputed helper matrix."""

VIEWING_CONDITIONS_RLAB: CanonicalMapping = CanonicalMapping(
    {"Average": 1 / 2.3, "Dim": 1 / 2.9, "Dark": 1 / 3.5}
)
VIEWING_CONDITIONS_RLAB.__doc__ = """
Reference *RLAB* colour appearance model viewing conditions.

References
----------
:cite:`Fairchild1996a`, :cite:`Fairchild2013w`
"""

D_FACTOR_RLAB: CanonicalMapping = CanonicalMapping(
    {
        "Hard Copy Images": 1,
        "Soft Copy Images": 0,
        "Projected Transparencies, Dark Room": 0.5,
    }
)
D_FACTOR_RLAB.__doc__ = """
*RLAB* colour appearance model *Discounting-the-Illuminant* factor values.

References
----------
:cite:`Fairchild1996a`, :cite:`Fairchild2013w`

Aliases:

-   'hard_cp_img': 'Hard Copy Images'
-   'soft_cp_img': 'Soft Copy Images'
-   'projected_dark': 'Projected Transparencies, Dark Room'
"""
D_FACTOR_RLAB["hard_cp_img"] = D_FACTOR_RLAB["Hard Copy Images"]
D_FACTOR_RLAB["soft_cp_img"] = D_FACTOR_RLAB["Soft Copy Images"]
D_FACTOR_RLAB["projected_dark"] = D_FACTOR_RLAB[
    "Projected Transparencies, Dark Room"
]


@dataclass
class CAM_ReferenceSpecification_RLAB(MixinDataclassArray):
    """
    Define the *RLAB* colour appearance model reference specification.

    This specification has field names consistent with *Fairchild (2013)*
    reference.

    Parameters
    ----------
    LR
        Correlate of *Lightness* :math:`L^R`.
    CR
        Correlate of *achromatic chroma* :math:`C^R`.
    hR
        *Hue* angle :math:`h^R` in degrees.
    sR
        Correlate of *saturation* :math:`s^R`.
    HR
        *Hue* :math:`h` composition :math:`H^R`.
    aR
        Red-green chromatic response :math:`a^R`.
    bR
        Yellow-blue chromatic response :math:`b^R`.

    References
    ----------
    :cite:`Fairchild1996a`, :cite:`Fairchild2013w`
    """

    LR: float | NDArrayFloat | None = field(default_factory=lambda: None)
    CR: float | NDArrayFloat | None = field(default_factory=lambda: None)
    hR: float | NDArrayFloat | None = field(default_factory=lambda: None)
    sR: float | NDArrayFloat | None = field(default_factory=lambda: None)
    HR: float | NDArrayFloat | None = field(default_factory=lambda: None)
    aR: float | NDArrayFloat | None = field(default_factory=lambda: None)
    bR: float | NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class CAM_Specification_RLAB(MixinDataclassArray):
    """
    Define the *RLAB* colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    *Fairchild (2013)* reference.

    Parameters
    ----------
    J
        Correlate of *Lightness* :math:`L^R`.
    C
        Correlate of *achromatic chroma* :math:`C^R`.
    h
        *Hue* angle :math:`h^R` in degrees.
    s
        Correlate of *saturation* :math:`s^R`.
    HC
        *Hue* :math:`h` composition :math:`H^C`.
    a
        Red-green chromatic response :math:`a^R`.
    b
        Yellow-blue chromatic response :math:`b^R`.

    Notes
    -----
    -   This specification is the one used in the current model implementation.

    References
    ----------
    :cite:`Fairchild1996a`, :cite:`Fairchild2013w`
    """

    J: NDArrayFloat | None = field(default_factory=lambda: None)
    C: NDArrayFloat | None = field(default_factory=lambda: None)
    h: NDArrayFloat | None = field(default_factory=lambda: None)
    s: NDArrayFloat | None = field(default_factory=lambda: None)
    HC: NDArrayFloat | None = field(default_factory=lambda: None)
    a: NDArrayFloat | None = field(default_factory=lambda: None)
    b: NDArrayFloat | None = field(default_factory=lambda: None)


def XYZ_to_RLAB(
    XYZ: ArrayLike,
    XYZ_n: ArrayLike,
    Y_n: ArrayLike,
    sigma: ArrayLike = VIEWING_CONDITIONS_RLAB["Average"],
    D: ArrayLike = D_FACTOR_RLAB["Hard Copy Images"],
) -> CAM_Specification_RLAB:
    """
    Compute the *RLAB* model color appearance correlates.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_n
        *CIE XYZ* tristimulus values of reference white.
    Y_n
        Absolute adapting luminance in :math:`cd/m^2`.
    sigma
        Relative luminance of the surround, see
        :attr:`colour.VIEWING_CONDITIONS_RLAB` for reference.
    D
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
    >>> sigma = VIEWING_CONDITIONS_RLAB["Average"]
    >>> D = D_FACTOR_RLAB["Hard Copy Images"]
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
    LMS_l_E = 3 * LMS_n / np.sum(LMS_n, axis=-1)[..., None]
    LMS_p_L = (1 + spow(Y_n[..., None], 1 / 3) + LMS_l_E) / (
        1 + spow(Y_n[..., None], 1 / 3) + 1 / LMS_l_E
    )

    LMS_a_L = (LMS_p_L + D[..., None] * (1 - LMS_p_L)) / LMS_n

    M = matrix_dot(
        matrix_dot(MATRIX_R, row_as_diagonal(LMS_a_L)), MATRIX_XYZ_TO_HPE
    )
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
    with sdiv_mode():
        sR = sdiv(CR, LR)

    return CAM_Specification_RLAB(
        LR,
        CR,
        as_float(from_range_degrees(hR)),
        sR,
        None,
        as_float(aR),
        as_float(bR),
    )
