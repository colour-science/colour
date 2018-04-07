# -*- coding: utf-8 -*-
"""
CIE 1964 U*V*W* Colourspace
===========================

Defines the *CIE 1964 U\*V\*W\** colourspace transformations:

-   :func:`colour.XYZ_to_UVW`
-   :func:`colour.UVW_to_XYZ`

See Also
--------
`CIE U*V*W* Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cie_uvw.ipynb>`_

References
----------
-   :cite:`Wikipediacj` : Wikipedia. (n.d.). CIE 1964 color space. Retrieved
    June 10, 2014, from http://en.wikipedia.org/wiki/CIE_1964_color_space
"""

from __future__ import division, unicode_literals

from colour.colorimetry import ILLUMINANTS
from colour.models import (UCS_to_uv, UCS_uv_to_xy, XYZ_to_UCS, XYZ_to_xyY,
                           xy_to_UCS_uv, xyY_to_XYZ, xyY_to_xy)
from colour.utilities import from_range_100, to_domain_100, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_UVW', 'UVW_to_XYZ']


def XYZ_to_UVW(
        XYZ,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']):
    """
    Converts from *CIE XYZ* tristimulus values to *CIE 1964 U\*V\*W\**
    colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like, optional
        Reference *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    ndarray
        *CIE 1964 U\*V\*W\** colourspace array.

    Warning
    -------
    The input domain and output range of that definition are non standard!

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are normalised to domain [0, 100].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are normalised to domain [0, 1].
    -   Output *CIE 1964 U\*V\*W\** colourspace array is normalised to range
        [0, 100].

    References
    ----------
    -   :cite:`Wikipediacj`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
    >>> XYZ_to_UVW(XYZ)  # doctest: +ELLIPSIS
    array([-28.0579733...,  -0.8819449...,  37.0041149...])
    """

    XYZ = to_domain_100(XYZ)

    xy = xyY_to_xy(illuminant)
    xyY = XYZ_to_xyY(XYZ, xy)
    _x, _y, Y = tsplit(xyY)

    u, v = tsplit(UCS_to_uv(XYZ_to_UCS(XYZ)))
    u_0, v_0 = tsplit(xy_to_UCS_uv(xy))

    W = 25 * Y ** (1 / 3) - 17
    U = 13 * W * (u - u_0)
    V = 13 * W * (v - v_0)

    UVW = tstack((U, V, W))

    return from_range_100(UVW)


def UVW_to_XYZ(
        UVW,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']):
    """
    Converts *CIE 1964 U\*V\*W\** colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    UVW : array_like
        *CIE 1964 U\*V\*W\** colourspace array.
    illuminant : array_like, optional
        Reference *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Warning
    -------
    The input domain and output range of that definition are non standard!

    Notes
    -----
    -   Input *CIE 1964 U\*V\*W\** colourspace array is normalised to domain
        [0, 100].
    -   Output *CIE XYZ* tristimulus values are normalised to range [0, 100].
    -   Output *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are normalised to range [0, 1].

    References
    ----------
    -   :cite:`Wikipediacj`

    Examples
    --------
    >>> import numpy as np
    >>> UVW = np.array([-28.05797333, -0.88194493, 37.00411491])
    >>> UVW_to_XYZ(UVW)  # doctest: +ELLIPSIS
    array([  7.049534...,  10.08    ...,   9.558313...])
    """

    U, V, W = tsplit(to_domain_100(UVW))

    u_0, v_0 = tsplit(xy_to_UCS_uv(xyY_to_xy(illuminant)))

    Y = ((W + 17) / 25) ** 3
    u = U / (13 * W) + u_0
    v = V / (13 * W) + v_0

    x, y = tsplit(UCS_uv_to_xy(tstack((u, v))))

    XYZ = xyY_to_XYZ(tstack((x, y, Y)))

    return from_range_100(XYZ)
