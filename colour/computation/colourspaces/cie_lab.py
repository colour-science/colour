# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cie_lab.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour *CIE Lab* colourspace objects.

**Others:**

"""

from __future__ import unicode_literals

import math
import numpy

import colour.computation.colourspaces.cie_xyy
import colour.dataset.illuminants.chromaticity_coordinates
from colour.computation.lightness import CIE_E
from colour.computation.lightness import CIE_K

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["XYZ_to_Lab",
           "Lab_to_XYZ",
           "Lab_to_LCHab",
           "LCHab_to_Lab"]


def XYZ_to_Lab(XYZ,
               illuminant=colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                   "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Converts from *CIE XYZ* colourspace to *CIE Lab* colourspace.

    References:

    -  http://www.brucelindbloom.com/Eqn_XYZ_to_Lab.html

    Usage::

        >>> XYZ_to_Lab(numpy.array([0.92193107, 1., 1.03744246]))
        array([[ 100.        ]
               [  -7.41787844]
               [ -15.85742105]])

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *CIE Lab* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *CIE XYZ* is in domain [0, 1].
    :note: *illuminant* is in domain [0, 1].
    :note: *L\** is in domain [0, 100].
    """

    X, Y, Z = numpy.ravel(XYZ)
    Xr, Yr, Zr = numpy.ravel(colour.computation.colourspaces.cie_xyy.xy_to_XYZ(illuminant))

    xr = X / Xr
    yr = Y / Yr
    zr = Z / Zr

    fx = xr ** (1. / 3.) if xr > CIE_E else (CIE_K * xr + 16.) / 116.
    fy = yr ** (1. / 3.) if yr > CIE_E else (CIE_K * yr + 16.) / 116.
    fz = zr ** (1. / 3.) if zr > CIE_E else (CIE_K * zr + 16.) / 116.

    L = 116. * fy - 16.
    a = 500. * (fx - fy)
    b = 200. * (fy - fz)

    return numpy.array([L, a, b]).reshape((3, 1))


def Lab_to_XYZ(Lab,
               illuminant=colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                   "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Converts from *CIE Lab* colourspace to *CIE XYZ* colourspace.

    References:

    -  http://www.brucelindbloom.com/Eqn_Lab_to_XYZ.html'.

    Usage::

        >>> Lab_to_XYZ(numpy.array([100., -7.41787844, -15.85742105]))
        array([[ 0.92193107]
               [ 0.11070565]
               [ 1.03744246]])

    :param Lab: *CIE Lab* colourspace matrix.
    :type Lab: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *CIE Lab* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *L\** is in domain [0, 100].
    :note: *illuminant* is in domain [0, 1].
    :note: *CIE XYZ* is in domain [0, 1].
    """

    L, a, b = numpy.ravel(Lab)
    Xr, Yr, Zr = numpy.ravel(colour.computation.colourspaces.cie_xyy.xy_to_XYZ(illuminant))

    fy = (L + 16.) / 116.
    fx = a / 500. + fy
    fz = fy - b / 200.

    xr = fx ** 3. if fx ** 3. > CIE_E else (116. * fx - 16.) / CIE_K
    yr = ((L + 16.) / 116.) ** 3. if L > CIE_K * CIE_E else L / CIE_K
    zr = fz ** 3. if fz ** 3. > CIE_E else (116. * fz - 16.) / CIE_K

    X = xr * Xr
    Y = yr * Yr
    Z = zr * Zr

    return numpy.array([X, Y, Z]).reshape((3, 1))


def Lab_to_LCHab(Lab):
    """
    Converts from *CIE Lab* colourspace to *CIE LCHab* colourspace.

    References:

    -  http://www.brucelindbloom.com/Eqn_Lab_to_LCH.html

    Usage::

        >>> Lab_to_LCHab(numpy.array([100., -7.41787844, -15.85742105]))
        array([[ 100.        ]
               [  17.50664796]
               [ 244.93046842]])

    :param Lab: *CIE Lab* colourspace matrix.
    :type Lab: array_like (3, 1)
    :return: *CIE LCHab* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *L\** is in domain [0, 100].
    """

    L, a, b = numpy.ravel(Lab)

    H = 180. * math.atan2(b, a) / math.pi
    if H < 0.:
        H += 360.

    return numpy.array([L, math.sqrt(a ** 2 + b ** 2), H]).reshape((3, 1))


def LCHab_to_Lab(LCHab):
    """
    Converts from *CIE LCHab* colourspace to *CIE Lab* colourspace.

    References:

    -  http://www.brucelindbloom.com/Eqn_LCH_to_Lab.html

    Usage::

        >>> LCHab_to_Lab(numpy.array([100., 17.50664796, 244.93046842]))
        array([[ 100.        ]
               [  -7.41787844]
               [ -15.85742105]])

    :param LCHab: *CIE LCHab* colourspace matrix.
    :type LCHab: array_like (3, 1)
    :return: *CIE Lab* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *L\** is in domain [0, 100].
    """

    L, C, H = numpy.ravel(LCHab)

    return numpy.array([L, C * math.cos(math.radians(H)), C * math.sin(math.radians(H))]).reshape((3, 1))
