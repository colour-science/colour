#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE Lab Colourspace
===================

Defines the *CIE Lab* colourspace transformations:

-   :func:`XYZ_to_Lab`
-   :func:`Lab_to_XYZ`
-   :func:`Lab_to_LCHab`
-   :func:`LCHab_to_Lab`

See Also
--------
`CIE Lab Colourspace IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/cie_lab.ipynb>`_  # noqa

References
----------
.. [1]  http://en.wikipedia.org/wiki/Lab_color_space
        (Last accessed 24 February 2014)
"""

from __future__ import division, unicode_literals

import math
import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.constants import CIE_E, CIE_K
from colour.models import xy_to_XYZ

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_Lab',
           'Lab_to_XYZ',
           'Lab_to_LCHab',
           'LCHab_to_Lab']


def XYZ_to_Lab(XYZ,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE XYZ* colourspace to *CIE Lab* colourspace.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray, (3,)
        *CIE Lab* colourspace matrix.

    Notes
    -----
    -   Input *CIE XYZ* is in domain [0, 1].
    -   Input *illuminant* chromaticity coordinates are in domain [0, 1].
    -   Output *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [2]  http://www.brucelindbloom.com/Eqn_XYZ_to_Lab.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> XYZ_to_Lab(np.array([0.92193107, 1, 1.03744246]))  # doctest: +ELLIPSIS
    array([ 100.        ,   -7.4178784...,  -15.8574210...])
    """

    X, Y, Z = np.ravel(XYZ)
    Xr, Yr, Zr = np.ravel(xy_to_XYZ(illuminant))

    xr = X / Xr
    yr = Y / Yr
    zr = Z / Zr

    fx = xr ** (1 / 3) if xr > CIE_E else (CIE_K * xr + 16) / 116
    fy = yr ** (1 / 3) if yr > CIE_E else (CIE_K * yr + 16) / 116
    fz = zr ** (1 / 3) if zr > CIE_E else (CIE_K * zr + 16) / 116

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.array([L, a, b])


def Lab_to_XYZ(Lab,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE Lab* colourspace to *CIE XYZ* colourspace.

    Parameters
    ----------
    Lab : array_like, (3,)
        *CIE Lab* colourspace matrix.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Input *Lightness* :math:`L^*` is in domain [0, 100].
    -   Input *illuminant* chromaticity coordinates are in domain [0, 1].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [3]  http://www.brucelindbloom.com/Eqn_Lab_to_XYZ.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> Lab = np.array([100, -7.41787844, -15.85742105])
    >>> Lab_to_XYZ(Lab)  # doctest: +ELLIPSIS
    array([ 0.9219310...,  1.        ,  1.0374424...])
    """

    L, a, b = np.ravel(Lab)
    Xr, Yr, Zr = np.ravel(xy_to_XYZ(illuminant))

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    xr = fx ** 3 if fx ** 3 > CIE_E else (116 * fx - 16) / CIE_K
    yr = ((L + 16) / 116) ** 3 if L > CIE_K * CIE_E else L / CIE_K
    zr = fz ** 3 if fz ** 3 > CIE_E else (116 * fz - 16) / CIE_K

    X = xr * Xr
    Y = yr * Yr
    Z = zr * Zr

    return np.array([X, Y, Z])


def Lab_to_LCHab(Lab):
    """
    Converts from *CIE Lab* colourspace to *CIE LCHab* colourspace.

    Parameters
    ----------
    Lab : array_like, (3,)
        *CIE Lab* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CIE LCHab* colourspace matrix.

    Notes
    -----
    -   *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [4]  http://www.brucelindbloom.com/Eqn_Lab_to_LCH.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> Lab = np.array([100, -7.41787844, -15.85742105])
    >>> Lab_to_LCHab(Lab)  # doctest: +ELLIPSIS
    array([ 100.        ,   17.5066479...,  244.9304684...])
    """

    L, a, b = np.ravel(Lab)

    H = 180 * math.atan2(b, a) / math.pi
    if H < 0:
        H += 360

    return np.array([L, math.sqrt(a ** 2 + b ** 2), H])


def LCHab_to_Lab(LCHab):
    """
    Converts from *CIE LCHab* colourspace to *CIE Lab* colourspace.

    Parameters
    ----------
    LCHab : array_like, (3,)
        *CIE LCHab* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CIE Lab* colourspace matrix.

    Notes
    -----
    -   *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [5]  http://www.brucelindbloom.com/Eqn_LCH_to_Lab.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> LCHab = np.array([100, 17.50664796, 244.93046842])
    >>> LCHab_to_Lab(LCHab)  # doctest: +ELLIPSIS
    array([ 100.        ,   -7.4178784...,  -15.8574210...])
    """

    L, C, H = np.ravel(LCHab)

    return np.array([L,
                     C * math.cos(math.radians(H)),
                     C * math.sin(math.radians(H))])
