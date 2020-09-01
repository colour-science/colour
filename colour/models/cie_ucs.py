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

from __future__ import division, unicode_literals

from colour.utilities import (from_range_1, full, to_domain_1, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'XYZ_to_UCS', 'UCS_to_XYZ', 'UCS_to_uv', 'uv_to_UCS', 'UCS_uv_to_xy',
    'xy_to_UCS_uv'
]


def XYZ_to_UCS(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to *CIE 1960 UCS* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
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
    >>> import colour.ndarray as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_UCS(XYZ)  # doctest: +ELLIPSIS
    array([ 0.1376933...,  0.1219722...,  0.1053731...])
    """

    X, Y, Z = tsplit(to_domain_1(XYZ))

    UVW = tstack([2 / 3 * X, Y, 1 / 2 * (-X + 3 * Y + Z)])

    return from_range_1(UVW)


def UCS_to_XYZ(UVW):
    """
    Converts from *CIE 1960 UCS* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    UVW : array_like
        *CIE 1960 UCS* colourspace array.

    Returns
    -------
    ndarray
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
    >>> import colour.ndarray as np
    >>> UVW = np.array([0.13769339, 0.12197225, 0.10537310])
    >>> UCS_to_XYZ(UVW)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    U, V, W = tsplit(to_domain_1(UVW))

    XYZ = tstack([3 / 2 * U, V, 3 / 2 * U - (3 * V) + (2 * W)])

    return from_range_1(XYZ)


def UCS_to_uv(UVW):
    """
    Returns the *uv* chromaticity coordinates from given *CIE 1960 UCS*
    colourspace array.

    Parameters
    ----------
    UVW : array_like
        *CIE 1960 UCS* colourspace array.

    Returns
    -------
    ndarray
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
    >>> import colour.ndarray as np
    >>> UVW = np.array([0.13769339, 0.12197225, 0.10537310])
    >>> UCS_to_uv(UVW)  # doctest: +ELLIPSIS
    array([ 0.3772021...,  0.3341350...])
    """

    U, V, W = tsplit(to_domain_1(UVW))

    U_V_W = U + V + W

    uv = tstack([U / U_V_W, V / U_V_W])

    return uv


def uv_to_UCS(uv, V=1):
    """
    Returns the *CIE 1960 UCS* colourspace array from given *uv* chromaticity
    coordinates.

    Parameters
    ----------
    uv : array_like
        *uv* chromaticity coordinates.
    V : numeric, optional
        Optional :math:`V` *luminance* value used to construct the
        *CIE 1960 UCS* colourspace array, the default :math:`V` *luminance* is
        set to 1.

    Returns
    -------
    ndarray
        *CIE 1960 UCS* colourspace array.

    References
    ----------
    :cite:`Wikipedia2008c`

    Examples
    --------
    >>> import colour.ndarray as np
    >>> uv = np.array([0.37720213, 0.33413508])
    >>> uv_to_UCS(uv)  # doctest: +ELLIPSIS
    array([ 1.1288911...,  1.        ,  0.8639104...])
    """

    u, v = tsplit(uv)
    V = full(u.shape, V)

    U = V * u / v
    W = -V * (u + v - 1) / v

    UVW = tstack([U, V, W])

    return from_range_1(UVW)


def UCS_uv_to_xy(uv):
    """
    Returns the *CIE xy* chromaticity coordinates from given *CIE 1960 UCS*
    colourspace *uv* chromaticity coordinates.

    Parameters
    ----------
    uv : array_like
        *CIE UCS uv* chromaticity coordinates.

    Returns
    -------
    ndarray
        *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Wikipedia2008c`

    Examples
    --------
    >>> import colour.ndarray as np
    >>> uv = np.array([0.37720213, 0.33413508])
    >>> UCS_uv_to_xy(uv)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...])
    """

    u, v = tsplit(uv)

    d = 2 * u - 8 * v + 4
    xy = tstack([3 * u / d, 2 * v / d])

    return xy


def xy_to_UCS_uv(xy):
    """
    Returns the *CIE 1960 UCS* colourspace *uv* chromaticity coordinates from
    given *CIE xy* chromaticity coordinates.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    ndarray
        *CIE UCS uv* chromaticity coordinates.

    References
    ----------
    :cite:`Wikipedia2008c`

    Examples
    --------
    >>> import colour.ndarray as np
    >>> xy = np.array([0.54369555, 0.32107941])
    >>> xy_to_UCS_uv(xy)  # doctest: +ELLIPSIS
    array([ 0.3772021...,  0.3341350...])
    """

    x, y = tsplit(xy)

    d = 12 * y - 2 * x + 3
    uv = tstack([4 * x / d, 6 * y / d])

    return uv
