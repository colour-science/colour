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
.. [1]  Wikipedia. (n.d.). CIE 1931 color space. Retrieved February 24, 2014,
        from http://en.wikipedia.org/wiki/CIE_1931_color_space
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
    .. [2]  Lindbloom, B. (2003). XYZ to xyY. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> XYZ_to_xyY(XYZ)  # doctest: +ELLIPSIS
    array([ 0.2641477...,  0.3777000...,  0.1008    ])
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
    .. [3]  Lindbloom, B. (2009). xyY to XYZ. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html

    Examples
    --------
    >>> xyY = np.array([0.26414772, 0.37770001, 0.1008])
    >>> xyY_to_XYZ(xyY)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
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
    >>> xy = (0.26414772236966133, 0.37770000704815188)
    >>> xy_to_XYZ(xy)  # doctest: +ELLIPSIS
    array([ 0.6993585...,  1.        ,  0.9482453...])
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
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> XYZ_to_xy(XYZ)  # doctest: +ELLIPSIS
    (0.2641477..., 0.3777000...)
    """

    xyY = np.ravel(XYZ_to_xyY(XYZ, illuminant))
    return xyY[0], xyY[1]
