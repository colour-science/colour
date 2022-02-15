"""
ProLab Colourspace
==================

Defines the *ProLab* colourspace transformations:

-   :func:`colour.XYZ_to_ProLab`
-   :func:`colour.ProLab_to_XYZ`

References
----------
-   :cite:`Ivan2021` : Ivan A. Konovalenko, Anna A. Smagina, Dmitry P.
    Nikolaev, Petr P. Nikolaev. ProLab: perceptually uniform projective colour
    coordinate system. doi:10.1109/ACCESS.2017
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import ArrayLike, NDArray
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.utilities import as_float_array, from_range_1, ones, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_Q",
    "MATRIX_INVERSE_Q",
    "ProLab_to_XYZ",
    "XYZ_to_ProLab",
]

MATRIX_Q: NDArray = np.array(
    [
        [75.54, 486.66, 167.39, 0],
        [617.72, -595.45, -22.27, 0],
        [48.34, 194.94, -243.28, 0],
        [0.7554, 3.8666, 1.6739, 1],
    ]
)
"""Normalised cone responses to *CIE XYZ* tristimulus values matrix."""

MATRIX_INVERSE_Q: NDArray = np.linalg.inv(MATRIX_Q)
"""Normalised cone responses to *ProLab* colourspace matrix."""


def projective_transformation(a: ArrayLike, Q: ArrayLike) -> NDArray:
    """
    Transform given array :math:`a` with the projective transformation matrix
    :math:`Q`.

    Parameters
    ----------
    a
        Array :math:`a` to apply the projective transformation matrix onto.
    Q
        Projective transformation matrix :math:`Q`.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed array :math:`a`.
    """

    a = as_float_array(a)
    Q = as_float_array(Q)

    shape = list(a.shape)
    shape[-1] = shape[-1] + 1

    M = ones(tuple(shape))
    M[..., :-1] = a

    homography = np.dot(M, np.transpose(Q))
    homography[..., 0:-1] /= homography[..., -1][..., np.newaxis]

    return homography[..., 0:-1]


def XYZ_to_ProLab(
    XYZ: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
) -> NDArray:
    """
    Convert from *CIE XYZ* tristimulus values to *ProLab* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *ProLab* colourspace array.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``ProLab`` | ``L`` : [0, 1]        | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-1, 1]       | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-1, 1]       | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Ivan2021`

    Examples
    --------
    >>> Lab = np.array([0.51634019, 0.15469500, 0.06289579])
    >>> XYZ_to_ProLab(Lab) # doctest: +ELLIPSIS
    array([  59.846628... ,  115.039635... ,   20.1251035...])
    """

    XYZ = to_domain_1(XYZ)
    XYZ_n = xyY_to_XYZ(xy_to_xyY(illuminant))

    ProLab = projective_transformation(XYZ / XYZ_n, MATRIX_Q)

    return from_range_1(ProLab)


def ProLab_to_XYZ(
    ProLab: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
) -> NDArray:
    """
    Convert from *ProLab* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    ProLab
        *ProLab* colourspace array.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Lab``    | ``L`` : [0, 1]        | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-1, 1]       | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-1, 1]       | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Ivan2021`

    Examples
    --------
    >>> ProLab = np.array([59.8466286, 115.0396354, 20.12510352])
    >>> ProLab_to_XYZ(ProLab) # doctest: +ELLIPSIS
    array([ 0.5163401...,  0.154695 ...,  0.0628957...])
    """

    ProLab = to_domain_1(ProLab)
    XYZ_n = xyY_to_XYZ(xy_to_xyY(illuminant))

    XYZ = projective_transformation(ProLab, MATRIX_INVERSE_Q)

    XYZ *= XYZ_n

    return from_range_1(XYZ)
