#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE UVW Colourspace
===================

Defines the *CIE U\*V\*W\** colourspace transformations:

-   :func:`XYZ_to_UVW`

See Also
--------
`CIE UVW Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cie_uvw.ipynb>`_

References
----------
.. [1]  Wikipedia. (n.d.). CIE 1964 color space. Retrieved June 10, 2014, from
        http://en.wikipedia.org/wiki/CIE_1964_color_space
"""

from __future__ import division, unicode_literals

from colour.colorimetry import ILLUMINANTS
from colour.models import (
    UCS_to_uv,
    XYZ_to_UCS,
    XYZ_to_xyY,
    xy_to_xyY,
    xyY_to_XYZ,
    xyY_to_xy)
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_UVW']


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

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 100].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *CIE UVW* colourspace array is in range [0, 100].

    Warning
    -------
    The input / output domains of that definition are non standard!

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
    >>> XYZ_to_UVW(XYZ)  # doctest: +ELLIPSIS
    array([-28.0579733...,  -0.8819449...,  37.0041149...])
    """

    xyY = XYZ_to_xyY(XYZ, xyY_to_xy(illuminant))
    _x, _y, Y = tsplit(xyY)

    u, v = tsplit(UCS_to_uv(XYZ_to_UCS(XYZ)))
    u_0, v_0 = tsplit(
        UCS_to_uv(XYZ_to_UCS(xyY_to_XYZ(xy_to_xyY(illuminant)))))

    W = 25 * Y ** (1 / 3) - 17
    U = 13 * W * (u - u_0)
    V = 13 * W * (v - v_0)

    UVW = tstack((U, V, W))

    return UVW
