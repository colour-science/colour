# -*- coding: utf-8 -*-
"""
:math:`I_GP_GT_G` Colourspace
=============================

Defines the :math:`I_GP_GT_G` colourspace transformations:

-   :func:`colour.XYZ_to_IgPgTg`
-   :func:`colour.IgPgTg_to_XYZ`

References
----------
-   :cite:`Hellwig2020` : Hellwig, L., & Fairchild, M. D. (2020). Using
    Gaussian Spectra to Derive a Hue-linear Color Space. Journal of Perceptual
    Imaging. doi:10.2352/J.Percept.Imaging.2020.3.2.020401
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow, vector_dot
from colour.hints import ArrayLike, NDArray
from colour.utilities import from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_IGPGTG_XYZ_TO_LMS',
    'MATRIX_IGPGTG_LMS_TO_XYZ',
    'MATRIX_IGPGTG_LMS_P_TO_IGPGTG',
    'MATRIX_IGPGTG_IGPGTG_TO_LMS_P',
    'XYZ_to_IgPgTg',
    'IgPgTg_to_XYZ',
]

MATRIX_IGPGTG_XYZ_TO_LMS: NDArray = np.array([
    [2.968, 2.741, -0.649],
    [1.237, 5.969, -0.173],
    [-0.318, 0.387, 2.311],
])
"""
*CIE XYZ* tristimulus values to normalised cone responses matrix.
"""

MATRIX_IGPGTG_LMS_TO_XYZ: NDArray = np.linalg.inv(MATRIX_IGPGTG_XYZ_TO_LMS)
"""
Normalised cone responses to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_IGPGTG_LMS_P_TO_IGPGTG: NDArray = np.array([
    [0.117, 1.464, 0.130],
    [8.285, -8.361, 21.400],
    [-1.208, 2.412, -36.530],
])
"""
Normalised non-linear cone responses to :math:`I_GP_GT_G` colourspace matrix.
"""

MATRIX_IGPGTG_IGPGTG_TO_LMS_P: NDArray = np.linalg.inv(
    MATRIX_IGPGTG_LMS_P_TO_IGPGTG)
"""
:math:`I_GP_GT_G` colourspace to normalised non-linear cone responses matrix.
"""


def XYZ_to_IgPgTg(XYZ: ArrayLike) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to :math:`I_GP_GT_G`
    colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`I_GP_GT_G` colourspace array.

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
    | ``IgPgTg`` | ``IG`` : [0, 1]       | ``IG`` : [0, 1] |
    |            |                       |                 |
    |            | ``PG`` : [-1, 1]      | ``PG`` : [-1, 1]|
    |            |                       |                 |
    |            | ``TG`` : [-1, 1]      | ``TG`` : [-1, 1]|
    +------------+-----------------------+-----------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    :cite:`Hellwig2020`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_IgPgTg(XYZ)  # doctest: +ELLIPSIS
    array([ 0.4242125...,  0.1863249...,  0.1068922...])
    """

    XYZ = to_domain_1(XYZ)

    LMS = vector_dot(MATRIX_IGPGTG_XYZ_TO_LMS, XYZ)
    LMS_prime = spow(LMS / np.array([18.36, 21.46, 19435]), 0.427)
    IgPgTg = vector_dot(MATRIX_IGPGTG_LMS_P_TO_IGPGTG, LMS_prime)

    return from_range_1(IgPgTg)


def IgPgTg_to_XYZ(IgPgTg: ArrayLike) -> NDArray:
    """
    Converts from :math:`I_GP_GT_G` colourspace to *CIE XYZ* tristimulus
    values.

    Parameters
    ----------
    IgPgTg
        :math:`I_GP_GT_G` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``IgPgTg`` | ``IG`` : [0, 1]       | ``IG`` : [0, 1] |
    |            |                       |                 |
    |            | ``PG`` : [-1, 1]      | ``PG`` : [-1, 1]|
    |            |                       |                 |
    |            | ``TG`` : [-1, 1]      | ``TG`` : [-1, 1]|
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Hellwig2020`

    Examples
    --------
    >>> IgPgTg = np.array([0.42421258, 0.18632491, 0.10689223])
    >>> IgPgTg_to_XYZ(IgPgTg)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    IgPgTg = to_domain_1(IgPgTg)

    LMS = vector_dot(MATRIX_IGPGTG_IGPGTG_TO_LMS_P, IgPgTg)
    LMS_prime = spow(LMS, 1 / 0.427) * np.array([18.36, 21.46, 19435])
    XYZ = vector_dot(MATRIX_IGPGTG_LMS_TO_XYZ, LMS_prime)

    return from_range_1(XYZ)
