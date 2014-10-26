#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE Luv Colourspace
===================

Defines the *CIE Luv* colourspace transformations:

-   :func:`XYZ_to_Luv`
-   :func:`Luv_to_XYZ`
-   :func:`Luv_to_uv`
-   :func:`Luv_uv_to_xy`
-   :func:`Luv_to_LCHuv`
-   :func:`LCHuv_to_Luv`

See Also
--------
`CIE Luv Colourspace IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/cie_luv.ipynb>`_  # noqa

References
----------
.. [1]  Wikipedia. (n.d.). CIELUV. Retrieved February 24, 2014, from
        http://en.wikipedia.org/wiki/CIELUV
"""

from __future__ import division, unicode_literals

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

__all__ = ['XYZ_to_Luv',
           'Luv_to_XYZ',
           'Luv_to_uv',
           'Luv_uv_to_xy',
           'Luv_to_LCHuv',
           'LCHuv_to_Luv']


def XYZ_to_Luv(XYZ,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE XYZ* colourspace to *CIE Luv* colourspace.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray, (3,)
        *CIE Luv* colourspace matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    -   Input *illuminant* chromaticity coordinates are in domain [0, 1].
    -   Output :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [2]  Lindbloom, B. (2003). XYZ to Luv. Retrieved February 24, 2014,
            from http://brucelindbloom.com/Eqn_XYZ_to_Luv.html

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> XYZ_to_Luv(XYZ)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -28.7922944...,  -1.3558195...])
    """

    X, Y, Z = np.ravel(XYZ)
    Xr, Yr, Zr = np.ravel(xy_to_XYZ(illuminant))

    yr = Y / Yr

    L = 116 * yr ** (1 / 3) - 16 if yr > CIE_E else CIE_K * yr
    u = (13 * L * ((4 * X / (X + 15 * Y + 3 * Z)) -
                   (4 * Xr / (Xr + 15 * Yr + 3 * Zr))))
    v = (13 * L * ((9 * Y / (X + 15 * Y + 3 * Z)) -
                   (9 * Yr / (Xr + 15 * Yr + 3 * Zr))))

    return np.array([L, u, v])


def Luv_to_XYZ(Luv,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE Luv* colourspace to *CIE XYZ* colourspace.

    Parameters
    ----------
    Luv : array_like, (3,)
        *CIE Luv* colourspace matrix.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Input :math:`L^*` is in domain [0, 100].
    -   Input *illuminant* chromaticity coordinates are in domain [0, 1].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [3]  Lindbloom, B. (2003). Luv to XYZ. Retrieved February 24, 2014,
            from http://brucelindbloom.com/Eqn_Luv_to_XYZ.html

    Examples
    --------
    >>> Luv = np.array([37.9856291, -28.79229446, -1.3558195])
    >>> Luv_to_XYZ(Luv)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    L, u, v = np.ravel(Luv)
    Xr, Yr, Zr = np.ravel(xy_to_XYZ(illuminant))

    Y = ((L + 16) / 116) ** 3 if L > CIE_E * CIE_K else L / CIE_K

    a = 1 / 3 * ((52 * L / (u + 13 * L *
                            (4 * Xr / (Xr + 15 * Yr + 3 * Zr)))) - 1)
    b = -5 * Y
    c = -1 / 3.0
    d = Y * (39 * L / (v + 13 * L *
                       (9 * Yr / (Xr + 15 * Yr + 3 * Zr))) - 5)

    X = (d - b) / (a - c)
    Z = X * a + b

    return np.array([X, Y, Z])


def Luv_to_uv(Luv,
              illuminant=ILLUMINANTS.get(
                  'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Returns the :math:`uv^p` chromaticity coordinates from given *CIE Luv*
    colourspace matrix.

    Parameters
    ----------
    Luv : array_like, (3,)
        *CIE Luv* colourspace matrix.
    illuminant : array_like, optional
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    tuple
        :math:`uv^p` chromaticity coordinates.

    Notes
    -----
    -   Input :math:`L^*` is in domain [0, 100].
    -   Output :math:`uv^p` chromaticity coordinates are in domain [0, 1].

    References
    ----------
    .. [4]  Wikipedia. (n.d.). The forward transformation. Retrieved February
            24, 2014, from
            http://en.wikipedia.org/wiki/CIELUV#The_forward_transformation

    Examples
    --------
    >>> Luv = np.array([37.9856291, -28.79229446, -1.3558195])
    >>> Luv_to_uv(Luv)  # doctest: +ELLIPSIS
    (0.1508530..., 0.4853297...)
    """

    X, Y, Z = np.ravel(Luv_to_XYZ(Luv, illuminant))

    return 4 * X / (X + 15 * Y + 3 * Z), 9 * Y / (X + 15 * Y + 3 * Z)


def Luv_uv_to_xy(uv):
    """
    Returns the *xy* chromaticity coordinates from given *CIE Luv* colourspace
    :math:`uv^p` chromaticity coordinates.

    Parameters
    ----------
    uv : array_like
        *CIE Luv u"v"* chromaticity coordinates.

    Returns
    -------
    tuple
        *xy* chromaticity coordinates.

    Notes
    -----
    -   Input :math:`uv^p` chromaticity coordinates are in domain [0, 1].
    -   Output *xy* is in domain [0, 1].

    References
    ----------
    .. [5]  Wikipedia. (n.d.). The reverse transformation. Retrieved from
            http://en.wikipedia.org/wiki/CIELUV#The_reverse_transformation

    Examples
    --------
    >>> uv = (0.15085309882985695, 0.48532970854318019)
    >>> Luv_uv_to_xy(uv)  # doctest: +ELLIPSIS
    (0.2641477..., 0.3777000...)
    """

    return (9 * uv[0] / (6 * uv[0] - 16 * uv[1] + 12), 4 * uv[1] /
            (6 * uv[0] - 16 * uv[1] + 12))


def Luv_to_LCHuv(Luv):
    """
    Converts from *CIE Luv* colourspace to *CIE LCHuv* colourspace.

    Parameters
    ----------
    Luv : array_like, (3,)
        *CIE Luv* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CIE LCHuv* colourspace matrix.

    Notes
    -----
    -   :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [6]  Lindbloom, B. (2003). Luv to LCH(uv). Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_Luv_to_LCH.html

    Examples
    --------
    >>> Luv = np.array([37.9856291, -28.79229446, -1.3558195])
    >>> Luv_to_LCHuv(Luv)  # doctest: +ELLIPSIS
    array([  37.9856291...,   28.8241993...,  182.6960474...])
    """

    L, u, v = np.ravel(Luv)

    H = 180 * np.arctan2(v, u) / np.pi
    if H < 0:
        H += 360

    return np.array([L, np.sqrt(u ** 2 + v ** 2), H])


def LCHuv_to_Luv(LCHuv):
    """
    Converts from *CIE LCHuv* colourspace to *CIE Luv* colourspace.

    Parameters
    ----------
    LCHuv : array_like, (3,)
        *CIE LCHuv* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CIE Luv* colourspace matrix.

    Notes
    -----
    -   :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [7]  Lindbloom, B. (2006). LCH(uv) to Luv. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_LCH_to_Luv.html

    Examples
    --------
    >>> LCHuv = np.array([37.9856291, 28.82419933, 182.69604747])
    >>> LCHuv_to_Luv(LCHuv)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -28.7922944...,  -1.3558195...])
    """

    L, C, H = np.ravel(LCHuv)

    return np.array([L,
                     C * np.cos(np.radians(H)),
                     C * np.sin(np.radians(H))])
