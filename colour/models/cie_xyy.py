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
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
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
    Converts from *CIE XYZ* tristimulus values to *CIE xyY* colourspace and
    reference *illuminant*.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray
        *CIE xyY* colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].
    -   Output *CIE xyY* colourspace array is in domain [0, 1].

    References
    ----------
    .. [2]  Lindbloom, B. (2003). XYZ to xyY. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_xyY(XYZ)  # doctest: +ELLIPSIS
    array([ 0.2641477...,  0.3777000...,  0.1008    ])
    """

    XYZ = np.asarray(XYZ)
    X, Y, Z = tsplit(XYZ)
    xy_w = np.asarray(illuminant)

    XYZ_n = np.zeros(XYZ.shape)
    XYZ_n[..., 0:2] = xy_w

    xyY = np.where(
        np.all(XYZ == 0, axis=-1)[..., np.newaxis],
        XYZ_n,
        tstack((X / (X + Y + Z), Y / (X + Y + Z), Y)))

    return xyY


def xyY_to_XYZ(xyY):
    """
    Converts from *CIE xyY* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    xyY : array_like
        *CIE xyY* colourspace array.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *CIE xyY* colourspace array is in domain [0, 1].
    -   Output *CIE XYZ* tristimulus values are in domain [0, 1].

    References
    ----------
    .. [3]  Lindbloom, B. (2009). xyY to XYZ. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html

    Examples
    --------
    >>> xyY = np.array([0.26414772, 0.37770001, 0.10080000])
    >>> xyY_to_XYZ(xyY)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    x, y, Y = tsplit(xyY)

    XYZ = np.where((y == 0)[..., np.newaxis],
                   tstack((y, y, y)),
                   tstack((x * Y / y, Y, (1 - x - y) * Y / y)))

    return XYZ


def xy_to_XYZ(xy):
    """
    Returns the *CIE XYZ* tristimulus values from given *xy* chromaticity
    coordinates.

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *xy* chromaticity coordinates are in domain [0, 1].
    -   Output *CIE XYZ* tristimulus values are in domain [0, 1].

    Examples
    --------
    >>> xy = np.array([0.26414772236966133, 0.37770000704815188])
    >>> xy_to_XYZ(xy)  # doctest: +ELLIPSIS
    array([ 0.6993585...,  1.        ,  0.9482453...])
    """

    x, y = tsplit(xy)

    xyY = tstack((x, y, np.ones(x.shape)))
    XYZ = xyY_to_XYZ(xyY)

    return XYZ


def XYZ_to_xy(XYZ,
              illuminant=ILLUMINANTS.get(
                  'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Returns the *xy* chromaticity coordinates from given *CIE XYZ* tristimulus
    values.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray
        *xy* chromaticity coordinates.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].
    -   Output *xy* chromaticity coordinates are in domain [0, 1].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_xy(XYZ)  # doctest: +ELLIPSIS
    array([ 0.2641477...,  0.3777000...])
    """

    xyY = XYZ_to_xyY(XYZ, illuminant)
    xy = xyY[..., 0:2]

    return xy
