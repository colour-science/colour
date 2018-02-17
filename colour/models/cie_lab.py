# -*- coding: utf-8 -*-
"""
CIE L*a*b* Colourspace
======================

Defines the *CIE L\*a\*b\** colourspace transformations:

-   :func:`colour.XYZ_to_Lab`
-   :func:`colour.Lab_to_XYZ`
-   :func:`colour.Lab_to_LCHab`
-   :func:`colour.LCHab_to_Lab`

See Also
--------
`CIE L*a*b* Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cie_lab.ipynb>`_

References
----------
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901-90633-6
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import cartesian_to_polar, polar_to_cartesian
from colour.colorimetry import ILLUMINANTS
from colour.constants import CIE_E, CIE_K
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_Lab', 'Lab_to_XYZ', 'Lab_to_LCHab', 'LCHab_to_Lab']


def XYZ_to_Lab(
        XYZ,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']):
    """
    Converts from *CIE XYZ* tristimulus values to *CIE L\*a\*b\** colourspace.

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
        *CIE L\*a\*b\** colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *Lightness* :math:`L^*` is in range [0, 100].

    References
    ----------
    -   :cite:`CIETC1-482004m`

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
                     np.power(XYZ_f, 1 / 3), (CIE_K * XYZ_f + 16) / 116)

    X_f, Y_f, Z_f = tsplit(XYZ_f)

    L = 116 * Y_f - 16
    a = 500 * (X_f - Y_f)
    b = 200 * (Y_f - Z_f)

    Lab = tstack((L, a, b))

    return Lab


def Lab_to_XYZ(
        Lab,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']):
    """
    Converts from *CIE L\*a\*b\** colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Lab : array_like
        *CIE L\*a\*b\** colourspace array.
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
    -   :cite:`CIETC1-482004m`

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
    Converts from *CIE L\*a\*b\** colourspace to *CIE L\*C\*Hab* colourspace.

    Parameters
    ----------
    Lab : array_like
        *CIE L\*a\*b\** colourspace array.

    Returns
    -------
    ndarray
        *CIE L\*C\*Hab* colourspace array.

    Notes
    -----
    -   *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    -   :cite:`CIETC1-482004m`

    Examples
    --------
    >>> Lab = np.array([37.98562910, -23.62907688, -4.41746615])
    >>> Lab_to_LCHab(Lab)  # doctest: +ELLIPSIS
    array([  37.9856291...,   24.0384542...,  190.5892337...])
    """

    L, a, b = tsplit(Lab)

    C, H = tsplit(cartesian_to_polar(tstack((a, b))))

    LCHab = tstack((L, C, np.degrees(H) % 360))

    return LCHab


def LCHab_to_Lab(LCHab):
    """
    Converts from *CIE L\*C\*Hab* colourspace to *CIE L\*a\*b\** colourspace.

    Parameters
    ----------
    LCHab : array_like
        *CIE L\*C\*Hab* colourspace array.

    Returns
    -------
    ndarray
        *CIE L\*a\*b\** colourspace array.

    Notes
    -----
    -   *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    -   :cite:`CIETC1-482004m`

    Examples
    --------
    >>> LCHab = np.array([37.98562910, 24.03845422, 190.58923377])
    >>> LCHab_to_Lab(LCHab)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -23.6290768...,  -4.4174661...])
    """

    L, C, H = tsplit(LCHab)

    a, b = tsplit(polar_to_cartesian(tstack((C, np.radians(H)))))

    Lab = tstack((L, a, b))

    return Lab
