#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE UVW Colourspace
===================

Defines the *CIE U\*V\*W\** colourspace transformations:

-   :func:`XYZ_to_UVW`

See Also
--------
`CIE UVW Colourspace IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/cie_uvw.ipynb>`_  # noqa

References
----------
.. [1]  Wikipedia. (n.d.). CIE 1964 color space. Retrieved June 10, 2014, from
        http://en.wikipedia.org/wiki/CIE_1964_color_space
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import UCS_to_uv, XYZ_to_UCS, XYZ_to_xyY, xy_to_XYZ

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_UVW']


def XYZ_to_UVW(XYZ,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE XYZ* colourspace to *CIE 1964 U\*V\*W\** colourspace.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray, (3,)
        *CIE 1964 U\*V\*W\** colourspace matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    -   Output *CIE UVW* colourspace matrix is in domain [0, 100].

    Warning
    -------
    The input / output domains of that definition are non standard!

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313]) * 100
    >>> XYZ_to_UVW(XYZ)  # doctest: +ELLIPSIS
    array([-28.0483277...,  -0.8805242...,  37.0041149...])
    """

    x, y, Y = np.ravel(XYZ_to_xyY(XYZ, illuminant))
    u, v = np.ravel(UCS_to_uv(XYZ_to_UCS(XYZ)))
    u0, v0 = np.ravel(UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(illuminant))))

    W = 25 * Y ** (1 / 3) - 17
    U = 13 * W * (u - u0)
    V = 13 * W * (v - v0)

    return np.array([U, V, W])
