# -*- coding: utf-8 -*-
"""
IPT Colourspace
===============

Defines the *IPT* colourspace transformations:

-   :func:`colour.XYZ_to_IPT`
-   :func:`colour.IPT_to_XYZ`

And computation of correlates:

-   :func:`colour.IPT_hue_angle`

References
----------
-   :cite:`Fairchild2013y` : Fairchild, M. D. (2013). IPT Colourspace. In
    Color Appearance Models (3rd ed., pp. 6197-6223). Wiley. ISBN:B00DAYO8E2
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow, vector_dot
from colour.hints import ArrayLike, FloatingOrNDArray, NDArray
from colour.utilities import (
    as_float,
    from_range_1,
    from_range_degrees,
    to_domain_1,
    tsplit,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_IPT_XYZ_TO_LMS',
    'MATRIX_IPT_LMS_TO_XYZ',
    'MATRIX_IPT_LMS_P_TO_IPT',
    'MATRIX_IPT_IPT_TO_LMS_P',
    'XYZ_to_IPT',
    'IPT_to_XYZ',
    'IPT_hue_angle',
]

MATRIX_IPT_XYZ_TO_LMS: NDArray = np.array([
    [0.4002, 0.7075, -0.0807],
    [-0.2280, 1.1500, 0.0612],
    [0.0000, 0.0000, 0.9184],
])
"""
*CIE XYZ* tristimulus values to normalised cone responses matrix.
"""

MATRIX_IPT_LMS_TO_XYZ: NDArray = np.linalg.inv(MATRIX_IPT_XYZ_TO_LMS)
"""
Normalised cone responses to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_IPT_LMS_P_TO_IPT: NDArray = np.array([
    [0.4000, 0.4000, 0.2000],
    [4.4550, -4.8510, 0.3960],
    [0.8056, 0.3572, -1.1628],
])
"""
Normalised non-linear cone responses to *IPT* colourspace matrix.
"""

MATRIX_IPT_IPT_TO_LMS_P: NDArray = np.linalg.inv(MATRIX_IPT_LMS_P_TO_IPT)
"""
*IPT* colourspace to normalised non-linear cone responses matrix.
"""


def XYZ_to_IPT(XYZ: ArrayLike) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *IPT* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *IPT* colourspace array.

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
    | ``IPT``    | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``P`` : [-1, 1]       | ``P`` : [-1, 1] |
    |            |                       |                 |
    |            | ``T`` : [-1, 1]       | ``T`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    :cite:`Fairchild2013y`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_IPT(XYZ)  # doctest: +ELLIPSIS
    array([ 0.3842619...,  0.3848730...,  0.1888683...])
    """

    XYZ = to_domain_1(XYZ)

    LMS = vector_dot(MATRIX_IPT_XYZ_TO_LMS, XYZ)
    LMS_prime = spow(LMS, 0.43)
    IPT = vector_dot(MATRIX_IPT_LMS_P_TO_IPT, LMS_prime)

    return from_range_1(IPT)


def IPT_to_XYZ(IPT: ArrayLike) -> NDArray:
    """
    Converts from *IPT* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    IPT
        *IPT* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``IPT``    | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``P`` : [-1, 1]       | ``P`` : [-1, 1] |
    |            |                       |                 |
    |            | ``T`` : [-1, 1]       | ``T`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Fairchild2013y`

    Examples
    --------
    >>> IPT = np.array([0.38426191, 0.38487306, 0.18886838])
    >>> IPT_to_XYZ(IPT)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    IPT = to_domain_1(IPT)

    LMS = vector_dot(MATRIX_IPT_IPT_TO_LMS_P, IPT)
    LMS_prime = spow(LMS, 1 / 0.43)
    XYZ = vector_dot(MATRIX_IPT_LMS_TO_XYZ, LMS_prime)

    return from_range_1(XYZ)


def IPT_hue_angle(IPT: ArrayLike) -> FloatingOrNDArray:
    """
    Computes the hue angle in degrees from *IPT* colourspace.

    Parameters
    ----------
    IPT
        *IPT* colourspace array.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Hue angle in degrees.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``IPT``    | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``P`` : [-1, 1]       | ``P`` : [-1, 1] |
    |            |                       |                 |
    |            | ``T`` : [-1, 1]       | ``T`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``hue``    | [0, 360]              | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Fairchild2013y`

    Examples
    --------
    >>> IPT = np.array([0.96907232, 1, 1.12179215])
    >>> IPT_hue_angle(IPT)  # doctest: +ELLIPSIS
    48.2852074...
    """

    _I, P, T = tsplit(to_domain_1(IPT))

    hue = np.degrees(np.arctan2(T, P)) % 360

    return as_float(from_range_degrees(hue))
