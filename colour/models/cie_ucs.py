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

See Also
--------
`CIE UCS Colourspace IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/cie_ucs.ipynb>`_  # noqa

References
----------
.. [1]  http://en.wikipedia.org/wiki/CIE_1960_color_space
        (Last accessed 24 February 2014)
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_UCS',
           'UCS_to_XYZ',
           'UCS_to_uv',
           'UCS_uv_to_xy']


def XYZ_to_UCS(XYZ):
    """
    Converts from *CIE XYZ* colourspace to *CIE UCS* colourspace.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CIE UCS* colourspace matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    -   Output *CIE UCS* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [2]  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ  # noqa
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> XYZ = np.array([0.1180583421, 0.1034, 0.0515089229])
    >>> XYZ_to_UCS(XYZ)  # doctest: +ELLIPSIS
    array([ 0.0787055...,  0.1034    ,  0.1218252...])
    """

    X, Y, Z = np.ravel(XYZ)

    return np.array([2 / 3 * X,
                     Y,
                     1 / 2 * (-X + 3 * Y + Z)])


def UCS_to_XYZ(UVW):
    """
    Converts from *CIE UCS* colourspace to *CIE XYZ* colourspace.

    Parameters
    ----------
    UVW : array_like, (3,)
        *CIE UCS* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Input *CIE UCS* colourspace matrix is in domain [0, 1].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [3]  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ  # noqa
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> UCS = np.array([0.07870556, 0.1034, 0.12182529])
    >>> UCS_to_XYZ(UCS)  # doctest: +ELLIPSIS
    array([ 0.1180583...,  0.1034    ,  0.0515089...])
    """

    U, V, W = np.ravel(UVW)

    return np.array(
        [3 / 2 * U, V, 3 / 2 * U - (3 * V) + (2 * W)])


def UCS_to_uv(UVW):
    """
    Returns the *uv* chromaticity coordinates from given *CIE UCS* colourspace
    matrix.

    Parameters
    ----------
    UVW : array_like, (3,)
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
    .. [4]  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ  # noqa
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> UCS = np.array([0.1180583421, 0.1034, 0.0515089229])
    >>> UCS_to_uv(UCS)  # doctest: +ELLIPSIS
    (0.4324999..., 0.3788000...)
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
    .. [5]  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ  # noqa
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> uv = (0.43249999995420696, 0.378800000065942)
    >>> UCS_uv_to_xy(uv)  # doctest: +ELLIPSIS
    (0.7072386..., 0.4129510...)
    """

    return (3 * uv[0] / (2 * uv[0] - 8 * uv[1] + 4),
            2 * uv[1] / (2 * uv[0] - 8 * uv[1] + 4))
