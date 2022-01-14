# -*- coding: utf-8 -*-
"""
Oklab Colourspace
=================

Defines the *Oklab* colourspace transformations:

-   :func:`colour.XYZ_to_Oklab`
-   :func:`colour.Oklab_to_XYZ`

References
----------
-   :cite:`Ottosson2020` : Ottosson, B. (2020). A perceptual color space for
    image processing. Retrieved December 24, 2020, from
    https://bottosson.github.io/posts/oklab/
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
    'MATRIX_1_XYZ_TO_LMS',
    'MATRIX_1_LMS_TO_XYZ',
    'MATRIX_2_LMS_TO_LAB',
    'MATRIX_2_LAB_TO_LMS',
    'XYZ_to_Oklab',
    'Oklab_to_XYZ',
]

MATRIX_1_XYZ_TO_LMS: NDArray = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
])
"""
*CIE XYZ* tristimulus values to normalised cone responses matrix.
"""

MATRIX_1_LMS_TO_XYZ: NDArray = np.linalg.inv(MATRIX_1_XYZ_TO_LMS)
"""
Normalised cone responses to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_2_LMS_TO_LAB: NDArray = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
"""
Normalised cone responses to *Oklab* colourspace matrix.
"""

MATRIX_2_LAB_TO_LMS: NDArray = np.linalg.inv(MATRIX_2_LMS_TO_LAB)
"""
*Oklab* colourspace to normalised cone responses matrix.
"""


def XYZ_to_Oklab(XYZ: ArrayLike) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *Oklab* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *Oklab* colourspace array.

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
    | ``Lab``    | ``L`` : [0, 1]        | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-1, 1]       | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-1, 1]       | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    :cite:`Ottosson2020`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_Oklab(XYZ)  # doctest: +ELLIPSIS
    array([ 0.5163401...,  0.154695 ...,  0.0628957...])
    """

    XYZ = to_domain_1(XYZ)

    LMS = vector_dot(MATRIX_1_XYZ_TO_LMS, XYZ)
    LMS_prime = spow(LMS, 1 / 3)
    Lab = vector_dot(MATRIX_2_LMS_TO_LAB, LMS_prime)

    return from_range_1(Lab)


def Oklab_to_XYZ(Lab: ArrayLike) -> NDArray:
    """
    Converts from *Oklab* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Lab
        *Oklab* colourspace array.

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
    :cite:`Ottosson2020`

    Examples
    --------
    >>> Lab = np.array([0.51634019, 0.15469500, 0.06289579])
    >>> Oklab_to_XYZ(Lab)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    Lab = to_domain_1(Lab)

    LMS = vector_dot(MATRIX_2_LAB_TO_LMS, Lab)
    LMS_prime = spow(LMS, 3)
    XYZ = vector_dot(MATRIX_1_LMS_TO_XYZ, LMS_prime)

    return from_range_1(XYZ)
