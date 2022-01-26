# -*- coding: utf-8 -*-
"""
CIE 1960 UCS Colourspace
========================

Defines the *CIE 1960 UCS* colourspace transformations:

-   :func:`colour.XYZ_to_UCS`
-   :func:`colour.UCS_to_XYZ`
-   :func:`colour.UCS_to_uv`
-   :func:`colour.uv_to_UCS`
-   :func:`colour.UCS_uv_to_xy`
-   :func:`colour.xy_to_UCS_uv`

References
----------
-   :cite:`Wikipedia2008` : Wikipedia. (2008). CIE 1960 color space. Retrieved
    February 24, 2014, from http://en.wikipedia.org/wiki/CIE_1960_color_space
-   :cite:`Wikipedia2008c` : Wikipedia. (2008). Relation to CIE XYZ. Retrieved
    February 24, 2014, from
    http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIE_XYZ
"""

from __future__ import annotations

from colour.hints import ArrayLike, Floating, NDArray
from colour.utilities import (
    as_float_scalar,
    from_range_1,
    full,
    to_domain_1,
    tsplit,
    tstack,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'XYZ_to_UCS',
    'UCS_to_XYZ',
    'UCS_to_uv',
    'uv_to_UCS',
    'UCS_uv_to_xy',
    'xy_to_UCS_uv',
]


def XYZ_to_UCS(XYZ: ArrayLike) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *CIE 1960 UCS* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE 1960 UCS* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2008c`, :cite:`Wikipedia2008`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_UCS(XYZ)  # doctest: +ELLIPSIS
    array([ 0.1376933...,  0.1219722...,  0.1053731...])
    """

    X, Y, Z = tsplit(to_domain_1(XYZ))

    UVW = tstack([2 / 3 * X, Y, 1 / 2 * (-X + 3 * Y + Z)])

    return from_range_1(UVW)


def UCS_to_XYZ(UVW: ArrayLike) -> NDArray:
    """
    Converts from *CIE 1960 UCS* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    UVW
        *CIE 1960 UCS* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2008c`, :cite:`Wikipedia2008`

    Examples
    --------
    >>> import numpy as np
    >>> UVW = np.array([0.13769339, 0.12197225, 0.10537310])
    >>> UCS_to_XYZ(UVW)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    U, V, W = tsplit(to_domain_1(UVW))

    XYZ = tstack([3 / 2 * U, V, 3 / 2 * U - (3 * V) + (2 * W)])

    return from_range_1(XYZ)


def UCS_to_uv(UVW: ArrayLike) -> NDArray:
    """
    Returns the *uv* chromaticity coordinates from given *CIE 1960 UCS*
    colourspace array.

    Parameters
    ----------
    UVW
        *CIE 1960 UCS* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *uv* chromaticity coordinates.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2008c`

    Examples
    --------
    >>> import numpy as np
    >>> UVW = np.array([0.13769339, 0.12197225, 0.10537310])
    >>> UCS_to_uv(UVW)  # doctest: +ELLIPSIS
    array([ 0.3772021...,  0.3341350...])
    """

    U, V, W = tsplit(to_domain_1(UVW))

    U_V_W = U + V + W

    uv = tstack([U / U_V_W, V / U_V_W])

    return uv


def uv_to_UCS(uv: ArrayLike, V: Floating = 1) -> NDArray:
    """
    Returns the *CIE 1960 UCS* colourspace array from given *uv* chromaticity
    coordinates.

    Parameters
    ----------
    uv
        *uv* chromaticity coordinates.
    V
        Optional :math:`V` *luminance* value used to construct the
        *CIE 1960 UCS* colourspace array, the default :math:`V` *luminance* is
        set to 1.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE 1960 UCS* colourspace array.

    References
    ----------
    :cite:`Wikipedia2008c`

    Examples
    --------
    >>> import numpy as np
    >>> uv = np.array([0.37720213, 0.33413508])
    >>> uv_to_UCS(uv)  # doctest: +ELLIPSIS
    array([ 1.1288911...,  1.        ,  0.8639104...])
    """

    u, v = tsplit(uv)
    V = as_float_scalar(to_domain_1(V))

    UVW = tstack([V * u / v, full(u.shape, V), -V * (u + v - 1) / v])

    return from_range_1(UVW)


def UCS_uv_to_xy(uv: ArrayLike) -> NDArray:
    """
    Returns the *CIE xy* chromaticity coordinates from given *CIE 1960 UCS*
    colourspace *uv* chromaticity coordinates.

    Parameters
    ----------
    uv
        *CIE UCS uv* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Wikipedia2008c`

    Examples
    --------
    >>> import numpy as np
    >>> uv = np.array([0.37720213, 0.33413508])
    >>> UCS_uv_to_xy(uv)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...])
    """

    u, v = tsplit(uv)

    d = 2 * u - 8 * v + 4
    xy = tstack([3 * u / d, 2 * v / d])

    return xy


def xy_to_UCS_uv(xy: ArrayLike) -> NDArray:
    """
    Returns the *CIE 1960 UCS* colourspace *uv* chromaticity coordinates from
    given *CIE xy* chromaticity coordinates.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS uv* chromaticity coordinates.

    References
    ----------
    :cite:`Wikipedia2008c`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.54369555, 0.32107941])
    >>> xy_to_UCS_uv(xy)  # doctest: +ELLIPSIS
    array([ 0.3772021...,  0.3341350...])
    """

    x, y = tsplit(xy)

    d = 12 * y - 2 * x + 3
    uv = tstack([4 * x / d, 6 * y / d])

    return uv
