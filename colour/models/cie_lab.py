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
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cie_lab.ipynb>`_

References
----------
.. [1]  Wikipedia. (n.d.). Lab color space. Retrieved February 24, 2014, from
        http://en.wikipedia.org/wiki/Lab_color_space
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.constants import CIE_E, CIE_K
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
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
    Converts from *CIE XYZ* tristimulus values to *CIE Lab* colourspace.

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
        *CIE Lab* colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *Lightness* :math:`L^*` is in range [0, 100].

    References
    ----------
    .. [2]  Lindbloom, B. (2003). XYZ to Lab. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_XYZ_to_Lab.html

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_Lab(XYZ)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -23.6290768...,  -4.4174661...])
    """

    XYZ = np.asarray(XYZ)
    XYZ_r = xyY_to_XYZ(xy_to_xyY(illuminant))

    XYZ_f = XYZ / XYZ_r

    XYZ_f = np.where(XYZ_f > CIE_E,
                     np.power(XYZ_f, 1 / 3),
                     (CIE_K * XYZ_f + 16) / 116)

    X_f, Y_f, Z_f = tsplit(XYZ_f)

    L = 116 * Y_f - 16
    a = 500 * (X_f - Y_f)
    b = 200 * (Y_f - Z_f)

    Lab = tstack((L, a, b))

    return Lab


def Lab_to_XYZ(Lab,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE Lab* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Lab : array_like
        *CIE Lab* colourspace array.
    illuminant : array_like, optional
        Reference *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *Lightness* :math:`L^*` is in domain [0, 100].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *CIE XYZ* tristimulus values are in range [0, 1].

    References
    ----------
    .. [3]  Lindbloom, B. (2008). Lab to XYZ. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_Lab_to_XYZ.html

    Examples
    --------
    >>> Lab = np.array([37.98562910, -23.62907688, -4.41746615])
    >>> Lab_to_XYZ(Lab)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    L, a, b = tsplit(Lab)
    XYZ_r = xyY_to_XYZ(xy_to_xyY(illuminant))

    f_y = (L + 16) / 116
    f_x = a / 500 + f_y
    f_z = f_y - b / 200

    x_r = np.where(f_x ** 3 > CIE_E, f_x ** 3, (116 * f_x - 16) / CIE_K)
    y_r = np.where(L > CIE_K * CIE_E, ((L + 16) / 116) ** 3, L / CIE_K)
    z_r = np.where(f_z ** 3 > CIE_E, f_z ** 3, (116 * f_z - 16) / CIE_K)

    XYZ = tstack((x_r, y_r, z_r)) * XYZ_r

    return XYZ


def Lab_to_LCHab(Lab):
    """
    Converts from *CIE Lab* colourspace to *CIE LCHab* colourspace.

    Parameters
    ----------
    Lab : array_like
        *CIE Lab* colourspace array.

    Returns
    -------
    ndarray
        *CIE LCHab* colourspace array.

    Notes
    -----
    -   *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [4]  Lindbloom, B. (2007). Lab to LCH(ab). Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_Lab_to_LCH.html

    Examples
    --------
    >>> Lab = np.array([37.98562910, -23.62907688, -4.41746615])
    >>> Lab_to_LCHab(Lab)  # doctest: +ELLIPSIS
    array([  37.9856291...,   24.0384542...,  190.5892337...])
    """

    L, a, b = tsplit(Lab)

    H = np.array(180 * np.arctan2(b, a) / np.pi)
    H[np.array(H < 0)] += 360

    LCHab = tstack((L, np.sqrt(a ** 2 + b ** 2), H))

    return LCHab


def LCHab_to_Lab(LCHab):
    """
    Converts from *CIE LCHab* colourspace to *CIE Lab* colourspace.

    Parameters
    ----------
    LCHab : array_like
        *CIE LCHab* colourspace array.

    Returns
    -------
    ndarray
        *CIE Lab* colourspace array.

    Notes
    -----
    -   *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [5]  Lindbloom, B. (2006). LCH(ab) to Lab. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_LCH_to_Lab.html

    Examples
    --------
    >>> LCHab = np.array([37.98562910, 24.03845422, 190.58923377])
    >>> LCHab_to_Lab(LCHab)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -23.6290768...,  -4.4174661...])
    """

    L, C, H = tsplit(LCHab)

    Lab = tstack((L,
                  C * np.cos(np.radians(H)),
                  C * np.sin(np.radians(H))))

    return Lab
