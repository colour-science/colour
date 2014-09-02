#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE xyY Colourspace
===================

Defines the *CIE xyY* colourspace transformations:

-   :func:`XYZ_to_xyY`
-   :func:`xyY_to_XYZ`
-   :func:`xy_to_XYZ`
-   :func:`XYZ_to_xy`

See Also
--------
`CIE xyY Colourspace IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/cie_xyy.ipynb>`_  # noqa

References
----------
.. [1]  http://en.wikipedia.org/wiki/CIE_1931_color_space
        (Last accessed 24 February 2014)
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_xyY',
           'xyY_to_XYZ',
           'xy_to_XYZ',
           'XYZ_to_xy']


def XYZ_to_xyY(XYZ,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE XYZ* colourspace to *CIE xyY* colourspace and reference
    *illuminant*.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray, (3,)
        *CIE xyY* colourspace matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    -   Output *CIE xyY* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [2]  http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> XYZ_to_xyY(np.array([0.1180583421, 0.1034, 0.0515089229]))
    array([ 0.4325,  0.3788,  0.1034])
    """

    X, Y, Z = np.ravel(XYZ)

    if X == 0 and Y == 0 and Z == 0:
        return np.array([illuminant[0], illuminant[1], Y])
    else:
        return np.array([X / (X + Y + Z), Y / (X + Y + Z), Y])


def xyY_to_XYZ(xyY):
    """
    Converts from *CIE xyY* colourspace to *CIE XYZ* colourspace.

    Parameters
    ----------
    xyY : array_like, (3,)
        *CIE xyY* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Input *CIE xyY* colourspace matrix is in domain [0, 1].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [3]  http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> xyY_to_XYZ(np.array([0.4325, 0.3788, 0.1034]))  # doctest: +ELLIPSIS
    array([ 0.1180583...,  0.1034    ,  0.0515089...])
    """

    x, y, Y = np.ravel(xyY)

    if y == 0:
        return np.array([0, 0, 0])
    else:
        return np.array([x * Y / y, Y, (1 - x - y) * Y / y])


def xy_to_XYZ(xy):
    """
    Returns the *CIE XYZ* colourspace matrix from given *xy* chromaticity
    coordinates.

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Input *xy* chromaticity coordinates are in domain [0, 1].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    Examples
    --------
    >>> xy_to_XYZ((0.25, 0.25))
    array([ 1.,  1.,  2.])
    """

    return xyY_to_XYZ(np.array([xy[0], xy[1], 1]))


def XYZ_to_xy(XYZ,
              illuminant=ILLUMINANTS.get(
                  'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Returns the *xy* chromaticity coordinates from given *CIE XYZ* colourspace
    matrix.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    tuple
        *xy* chromaticity coordinates.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    -   Output *xy* chromaticity coordinates are in domain [0, 1].

    Examples
    --------
    >>> XYZ_to_xy(np.array([0.97137399, 1, 1.04462134]))  # doctest: +ELLIPSIS
    (0.3220741..., 0.3315655...)
    >>> XYZ_to_xy((0.97137399, 1, 1.04462134))  # doctest: +ELLIPSIS
    (0.3220741..., 0.3315655...)
    """

    xyY = np.ravel(XYZ_to_xyY(XYZ, illuminant))
    return xyY[0], xyY[1]
