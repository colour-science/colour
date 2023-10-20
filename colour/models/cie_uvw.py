"""
CIE 1964 U*V*W* Colourspace
===========================

Defines the *CIE 1964 U\\*V\\*W\\** colourspace transformations:

-   :func:`colour.XYZ_to_UVW`
-   :func:`colour.UVW_to_XYZ`

References
----------
-   :cite:`Wikipedia2008a` : Wikipedia. (2008). CIE 1964 color space.
    Retrieved June 10, 2014, from
    http://en.wikipedia.org/wiki/CIE_1964_color_space
"""

from __future__ import annotations

from colour.algebra import sdiv, sdiv_mode, spow
from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import ArrayLike, NDArrayFloat
from colour.models import (
    UCS_uv_to_xy,
    XYZ_to_xy,
    xy_to_UCS_uv,
    xyY_to_xy,
    xyY_to_XYZ,
)
from colour.utilities import from_range_100, to_domain_100, tsplit, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "XYZ_to_UVW",
    "UVW_to_XYZ",
]


def XYZ_to_UVW(
    XYZ: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *CIE 1964 U\\*V\\*W\\**
    colourspace.

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
        *CIE 1964 U\\*V\\*W\\** colourspace array.

    Notes
    -----
    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``XYZ``        | [0, 100]              | [0, 1]          |
    +----------------+-----------------------+-----------------+
    | ``illuminant`` | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

    +----------------+-----------------------+-----------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``UVW``        | ``U`` : [-100, 100]   | ``U`` : [-1, 1] |
    |                |                       |                 |
    |                | ``V`` : [-100, 100]   | ``V`` : [-1, 1] |
    |                |                       |                 |
    |                | ``W`` : [0, 100]      | ``W`` : [0, 1]  |
    +----------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Wikipedia2008a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
    >>> XYZ_to_UVW(XYZ)  # doctest: +ELLIPSIS
    array([ 94.5503572...,  11.5553652...,  40.5475740...])
    """

    XYZ = to_domain_100(XYZ)

    xy = XYZ_to_xy(XYZ / 100)
    xy_n = xyY_to_xy(illuminant)
    Y = XYZ[..., 1]

    uv = xy_to_UCS_uv(xy)
    uv_0 = xy_to_UCS_uv(xy_n)

    W = 25 * spow(Y, 1 / 3) - 17
    U, V = tsplit(13 * W[..., None] * (uv - uv_0))

    UVW = tstack([U, V, W])

    return from_range_100(UVW)


def UVW_to_XYZ(
    UVW: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
) -> NDArrayFloat:
    """
    Convert *CIE 1964 U\\*V\\*W\\** colourspace to *CIE XYZ* tristimulus
    values.

    Parameters
    ----------
    UVW
        *CIE 1964 U\\*V\\*W\\** colourspace array.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``UVW``        | ``U`` : [-100, 100]   | ``U`` : [-1, 1] |
    |                |                       |                 |
    |                | ``V`` : [-100, 100]   | ``V`` : [-1, 1] |
    |                |                       |                 |
    |                | ``W`` : [0, 100]      | ``W`` : [0, 1]  |
    +----------------+-----------------------+-----------------+
    | ``illuminant`` | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

    +----------------+-----------------------+-----------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``XYZ``        | [0, 100               | [0, 1]          |
    +----------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Wikipedia2008a`

    Examples
    --------
    >>> import numpy as np
    >>> UVW = np.array([94.55035725, 11.55536523, 40.54757405])
    >>> UVW_to_XYZ(UVW)
    array([ 20.654008,  12.197225,   5.136952])
    """

    U, V, W = tsplit(to_domain_100(UVW))

    u_0, v_0 = tsplit(xy_to_UCS_uv(xyY_to_xy(illuminant)))

    Y = ((W + 17) / 25) ** 3

    with sdiv_mode():
        u = sdiv(U, 13 * W) + u_0
        v = sdiv(V, 13 * W) + v_0

    x, y = tsplit(UCS_uv_to_xy(tstack([u, v])))

    XYZ = xyY_to_XYZ(tstack([x, y, Y]))

    return from_range_100(XYZ)
