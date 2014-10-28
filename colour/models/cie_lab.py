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
.. [1]  Wikipedia. (n.d.). Lab color space. Retrieved February 24, 2014, from
        http://en.wikipedia.org/wiki/Lab_color_space
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
    .. [2]  Lindbloom, B. (2003). XYZ to Lab. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_XYZ_to_Lab.html

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> XYZ_to_Lab(XYZ)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -23.6230288...,  -4.4141703...])
    """

    X, Y, Z = np.ravel(XYZ)
    X_r, Y_r, Z_r = np.ravel(xy_to_XYZ(illuminant))

    x_r = X / X_r
    y_r = Y / Y_r
    z_r = Z / Z_r

    f_x = x_r ** (1 / 3) if x_r > CIE_E else (CIE_K * x_r + 16) / 116
    f_y = y_r ** (1 / 3) if y_r > CIE_E else (CIE_K * y_r + 16) / 116
    f_z = z_r ** (1 / 3) if z_r > CIE_E else (CIE_K * z_r + 16) / 116

    L = 116 * f_y - 16
    a = 500 * (f_x - f_y)
    b = 200 * (f_y - f_z)

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
    .. [3]  Lindbloom, B. (2008). Lab to XYZ. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_Lab_to_XYZ.html

    Examples
    --------
    >>> Lab = np.array([37.9856291, -23.62302887, -4.41417036])
    >>> Lab_to_XYZ(Lab)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    L, a, b = np.ravel(Lab)
    X_r, Y_r, Z_r = np.ravel(xy_to_XYZ(illuminant))

    f_y = (L + 16) / 116
    f_x = a / 500 + f_y
    f_z = f_y - b / 200

    x_r = f_x ** 3 if f_x ** 3 > CIE_E else (116 * f_x - 16) / CIE_K
    y_r = ((L + 16) / 116) ** 3 if L > CIE_K * CIE_E else L / CIE_K
    z_r = f_z ** 3 if f_z ** 3 > CIE_E else (116 * f_z - 16) / CIE_K

    X = x_r * X_r
    Y = y_r * Y_r
    Z = z_r * Z_r

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
    .. [4]  Lindbloom, B. (2007). Lab to LCH(ab). Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_Lab_to_LCH.html

    Examples
    --------
    >>> Lab = np.array([37.9856291, -23.62302887, -4.41417036])
    >>> Lab_to_LCHab(Lab)  # doctest: +ELLIPSIS
    array([  37.9856291...,   24.0319036...,  190.5841597...])
    """

    L, a, b = np.ravel(Lab)

    H = 180 * np.arctan2(b, a) / np.pi
    if H < 0:
        H += 360

    return np.array([L, np.sqrt(a ** 2 + b ** 2), H])


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
    .. [5]  Lindbloom, B. (2006). LCH(ab) to Lab. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_LCH_to_Lab.html

    Examples
    --------
    >>> LCHab = np.array([37.9856291, 24.03190365, 190.58415972])
    >>> LCHab_to_Lab(LCHab)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -23.6230288...,  -4.4141703...])
    """

    L, C, H = np.ravel(LCHab)

    return np.array([L,
                     C * np.cos(np.radians(H)),
                     C * np.sin(np.radians(H))])
