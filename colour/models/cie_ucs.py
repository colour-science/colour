"""
CIE 1960 UCS Colourspace
========================

Define the *CIE 1960 UCS* colourspace transformations.

-   :func:`colour.XYZ_to_UCS`
-   :func:`colour.UCS_to_XYZ`
-   :func:`colour.UCS_to_uv`
-   :func:`colour.uv_to_UCS`
-   :func:`colour.UCS_uv_to_xy`
-   :func:`colour.xy_to_UCS_uv`
-   :func:`colour.XYZ_to_CIE1960UCS`
-   :func:`colour.CIE1960UCS_to_XYZ`

References
----------
-   :cite:`Wikipedia2008` : Wikipedia. (2008). CIE 1960 color space. Retrieved
    February 24, 2014, from http://en.wikipedia.org/wiki/CIE_1960_color_space
-   :cite:`Wikipedia2008c` : Wikipedia. (2008). Relation to CIE XYZ. Retrieved
    February 24, 2014, from
    http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIE_XYZ
"""

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.hints import (  # noqa: TC001
    ArrayLike,
    Domain1,
    NDArrayFloat,
    Range1,
)
from colour.utilities import from_range_1, to_domain_1, tsplit, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "XYZ_to_UCS",
    "UCS_to_XYZ",
    "UCS_to_uv",
    "uv_to_UCS",
    "UCS_uv_to_xy",
    "xy_to_UCS_uv",
    "XYZ_to_CIE1960UCS",
    "CIE1960UCS_to_XYZ",
]


def XYZ_to_UCS(XYZ: Domain1) -> Range1:
    """
    Convert from *CIE XYZ* tristimulus values to *CIE 1960 UCS* :math:`UVW`
    colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE 1960 UCS* :math:`UVW` colourspace array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | 1                     | 1             |
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


def UCS_to_XYZ(UVW: Domain1) -> Range1:
    """
    Convert from *CIE 1960 UCS* :math:`UVW` colourspace to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    UVW
        *CIE 1960 UCS* :math:`UVW` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
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


def UCS_to_uv(UVW: Domain1) -> NDArrayFloat:
    """
    Convert from *CIE 1960 UCS* colourspace to *uv* chromaticity
    coordinates.

    Parameters
    ----------
    UVW
        *CIE 1960 UCS* :math:`UVW` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS uv* chromaticity coordinates.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | 1                     | 1             |
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

    with sdiv_mode():
        return tstack([sdiv(U, U_V_W), sdiv(V, U_V_W)])


def uv_to_UCS(uv: ArrayLike, V: Domain1 = 1) -> Range1:
    """
    Convert from *uv* chromaticity coordinates to *CIE 1960 UCS*
    colourspace.

    Parameters
    ----------
    uv
        *CIE UCS uv* chromaticity coordinates.
    V
        Optional :math:`V` *luminance* value used to construct the
        *CIE 1960 UCS* :math:`UVW` colourspace array, the default :math:`V`
        *luminance* is set to 1.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE 1960 UCS* :math:`UVW` colourspace array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | 1                     | 1             |
    +------------+-----------------------+---------------+

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
    V = to_domain_1(V)

    with sdiv_mode():
        UVW = tstack([V * sdiv(u, v), np.resize(V, u.shape), -V * sdiv(u + v - 1, v)])

    return from_range_1(UVW)


def UCS_uv_to_xy(uv: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE 1960 UCS* :math:`UVW` colourspace *uv* chromaticity
    coordinates to *CIE xy* chromaticity coordinates.

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

    with sdiv_mode():
        return tstack([sdiv(3 * u, d), sdiv(2 * v, d)])


def xy_to_UCS_uv(xy: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE xy* chromaticity coordinates to *CIE 1960 UCS*
    :math:`UVW` colourspace *uv* chromaticity coordinates.

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

    with sdiv_mode():
        return tstack([sdiv(4 * x, d), sdiv(6 * y, d)])


def XYZ_to_CIE1960UCS(
    XYZ: Domain1,
) -> Range1:
    """
    Convert from *CIE XYZ* tristimulus values to *CIE 1960 UCS* :math:`uvV`
    colourspace.

    This colourspace combines the *CIE 1960 UCS* :math:`UVW` colourspace
    :math:`uv` chromaticity coordinates with the luminance :math:`V` from
    the *CIE 1960 UCS* :math:`UVW` colourspace.

    It is a convenient definition for use with the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`uvV` colourspace array.

    Notes
    -----
    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``XYZ``        | 1                     | 1               |
    +----------------+-----------------------+-----------------+

    +----------------+-----------------------+-----------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``uvV``        | 1                     | 1               |
    +----------------+-----------------------+-----------------+

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_CIE1960UCS(XYZ)  # doctest: +ELLIPSIS
    array([ 0.3772021...,  0.3341350...,  0.12197225])
    """

    UVW = XYZ_to_UCS(XYZ)

    _U, V, _W = tsplit(UVW)

    u, v = tsplit(UCS_to_uv(UVW))

    return tstack([u, v, V])


def CIE1960UCS_to_XYZ(
    uvV: Domain1,
) -> Range1:
    """
    Convert from *CIE 1960 UCS* :math:`uvV` colourspace to *CIE XYZ*
    tristimulus values.

    This colourspace combines the *CIE 1960 UCS* :math:`UVW` colourspace
    :math:`uv` chromaticity coordinates with the luminance :math:`V` from
    the *CIE 1960 UCS* :math:`UVW` colourspace.

    It is a convenient definition for use with the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    uvV
        :math:`uvV` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``uvV``        | 1                     | 1               |
    +----------------+-----------------------+-----------------+

    +----------------+-----------------------+-----------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``XYZ``        | 1                     | 1               |
    +----------------+-----------------------+-----------------+

    Examples
    --------
    >>> import numpy as np
    >>> uvV = np.array([0.37720213, 0.33413509, 0.12197225])
    >>> CIE1960UCS_to_XYZ(uvV)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    u, v, V = tsplit(uvV)

    U, _V, W = tsplit(uv_to_UCS(tstack([u, v]), V))

    return UCS_to_XYZ(tstack([U, V, W]))
