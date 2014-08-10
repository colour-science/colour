#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE UCS Colourspace
===================

Defines the *CIE UCS* colourspace transformations:

-   :func:`XYZ_to_UCS`
-   :func:`UCS_to_XYZ`
-   :func:`UCS_to_uv`
-   :func:`UCS_uv_to_xy`

References
----------
.. [1]  http://en.wikipedia.org/wiki/CIE_1960_color_space
        (Last accessed 24 February 2014)
"""

from __future__ import unicode_literals

import numpy as np

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["XYZ_to_UCS",
           "UCS_to_XYZ",
           "UCS_to_uv",
           "UCS_uv_to_xy"]


def XYZ_to_UCS(XYZ):
    """
    Converts from *CIE XYZ* colourspace to *CIE UCS* colourspace.

    Parameters
    ----------
    XYZ : array_like, (3, 1)
        *CIE XYZ* colourspace matrix.

    Returns
    -------
    ndarray, (3, 1)
        *CIE UCS* colourspace matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    -   Output *CIE UCS* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [2]  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> colour.XYZ_to_UCS(np.array([0.1180583421, 0.1034, 0.0515089229]))
    array([[ 0.07870556]
          [ 0.1034    ]
          [ 0.12182529]])
    """

    X, Y, Z = np.ravel(XYZ)

    return np.array([2. / 3. * X,
                     Y,
                     1. / 2. * (-X + 3. * Y + Z)]).reshape((3, 1))


def UCS_to_XYZ(UVW):
    """
    Converts from *CIE UCS* colourspace to *CIE XYZ* colourspace.

    Parameters
    ----------
    UVW : array_like, (3, 1)
        *CIE UCS* colourspace matrix.

    Returns
    -------
    ndarray, (3, 1)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Input *CIE UCS* colourspace matrix is in domain [0, 1].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [3]  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> colour.UCS_to_XYZ(np.array([0.07870556, 0.1034, 0.12182529]))
    array([[ 0.11805834]
           [ 0.1034    ]
           [ 0.05150892]])
    """

    U, V, W = np.ravel(UVW)

    return np.array(
        [3. / 2. * U, V, 3. / 2. * U - (3. * V) + (2. * W)]).reshape((3, 1))


def UCS_to_uv(UVW):
    """
    Returns the *uv* chromaticity coordinates from given *CIE UCS* colourspace
    matrix.

    Parameters
    ----------
    UVW : array_like, (3, 1)
        *CIE UCS* colourspace matrix.

    Returns
    -------
    tuple
        *uv* chromaticity coordinates.

    Notes
    -----
    -   Input *CIE UCS* colourspace matrix is in domain [0, 1].
    -   Output *uv* chromaticity coordinates are in domain [0, 1].

    References
    ----------
    .. [4]  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> colour.UCS_to_uv(np.array([0.1180583421, 0.1034, 0.0515089229]))
    (0.43249999995420696, 0.378800000065942)
    """

    U, V, W = np.ravel(UVW)

    return U / (U + V + W), V / (U + V + W)


def UCS_uv_to_xy(uv):
    """
    Returns the *xy* chromaticity coordinates from given *CIE UCS* colourspace
    *uv* chromaticity coordinates.

    Parameters
    ----------
    uv : array_like
        *CIE UCS uv* chromaticity coordinates.

    Returns
    -------
    tuple
        *xy* chromaticity coordinates.

    Notes
    -----
    -   Input *uv* chromaticity coordinates are in domain [0, 1].
    -   Output *xy* chromaticity coordinates are in domain [0, 1].

    References
    ----------
    .. [5]  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> colour.UCS_uv_to_xy((0.43249999995420696, 0.378800000065942))
    (0.7072386352886122, 0.4129510522116816)
    """

    return (3. * uv[0] / (2. * uv[0] - 8. * uv[1] + 4.),
            2. * uv[1] / (2. * uv[0] - 8. * uv[1] + 4.))