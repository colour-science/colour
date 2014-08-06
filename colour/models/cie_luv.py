# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cie_luv.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour *CIE Luv* colourspace objects.

**Others:**

"""

from __future__ import unicode_literals

import math

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.constants import CIE_E, CIE_K
from colour.models import xy_to_XYZ

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["XYZ_to_Luv",
           "Luv_to_XYZ",
           "Luv_to_uv",
           "Luv_uv_to_xy",
           "Luv_to_LCHuv",
           "LCHuv_to_Luv"]


def XYZ_to_Luv(XYZ,
               illuminant=ILLUMINANTS.get(
                   "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Converts from *CIE XYZ* colourspace to *CIE Luv* colourspace.

    Examples::

        >>> XYZ_to_Luv(np.array([0.92193107, 1., 1.03744246]))
        array([[ 100.        ]
               [ -20.04304247]
               [ -45.09684555]])

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *CIE Luv* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    :note: Input *illuminant* is in domain [0, 1].
    :note: Output *L\** is in domain [0, 100].

    References:

    -  http://brucelindbloom.com/Eqn_XYZ_to_Luv.html \
    (Last accessed 24 February 2014)
    """

    X, Y, Z = np.ravel(XYZ)
    Xr, Yr, Zr = np.ravel(xy_to_XYZ(illuminant))

    yr = Y / Yr

    L = 116. * yr ** ( 1. / 3.) - 16. if yr > CIE_E else CIE_K * yr
    u = (13. * L * ((4. * X / (X + 15. * Y + 3. * Z)) -
                    (4. * Xr / (Xr + 15. * Yr + 3. * Zr))))
    v = (13. * L * ((9. * Y / (X + 15. * Y + 3. * Z)) -
                    (9. * Yr / (Xr + 15. * Yr + 3. * Zr))))

    return np.array([L, u, v]).reshape((3, 1))


def Luv_to_XYZ(Luv,
               illuminant=ILLUMINANTS.get(
                   "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Converts from *CIE Luv* colourspace to *CIE XYZ* colourspace.

    Examples::

        >>> Luv_to_XYZ(np.array([100., -20.04304247, -19.81676035]))
        array([[ 0.92193107]
               [ 1.        ]
               [ 1.03744246]])

    :param Luv: *CIE Luv* colourspace matrix.
    :type Luv: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *CIE XYZ* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *L\** is in domain [0, 100].
    :note: Input *illuminant* is in domain [0, 1].
    :note: Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    References:

    -  http://brucelindbloom.com/Eqn_Luv_to_XYZ.html \
    (Last accessed 24 February 2014)
    """

    L, u, v = np.ravel(Luv)
    Xr, Yr, Zr = np.ravel(xy_to_XYZ(illuminant))

    Y = ((L + 16.) / 116.) ** 3. if L > CIE_E * CIE_K else L / CIE_K

    a = 1. / 3. * ((52. * L / (u + 13. * L *
                               (4. * Xr / (Xr + 15. * Yr + 3. * Zr)))) - 1.)
    b = -5. * Y
    c = -1. / 3.0
    d = Y * (39. * L / (v + 13. * L *
                        (9. * Yr / (Xr + 15. * Yr + 3. * Zr))) - 5.)

    X = (d - b) / (a - c)
    Z = X * a + b

    return np.array([X, Y, Z]).reshape((3, 1))


def Luv_to_uv(Luv,
              illuminant=ILLUMINANTS.get(
                  "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Returns the *u'v'* chromaticity coordinates from given *CIE Luv* colourspace
    matrix.

    Examples::

        >>> Luv_to_uv(np.array([100., -20.04304247, -19.81676035]))
        (0.19374142100850045, 0.47283165896209456)

    :param Luv: *CIE Luv* colourspace matrix.
    :type Luv: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *u'v'* chromaticity coordinates.
    :rtype: tuple

    :note: Input *L\** is in domain [0, 100].
    :note: Output *u'v'* is in domain [0, 1].

    References:

    -  http://en.wikipedia.org/wiki/CIELUV#The_forward_transformation \
    (Last accessed 24 February 2014)
    """

    X, Y, Z = np.ravel(Luv_to_XYZ(Luv, illuminant))

    return 4. * X / (X + 15. * Y + 3. * Z), 9. * Y / (X + 15. * Y + 3. * Z)


def Luv_uv_to_xy(uv):
    """
    Returns the *xy* chromaticity coordinates from given *CIE Luv* colourspace
    *u'v'* chromaticity coordinates.

    Examples::

        >>> Luv_uv_to_xy((0.2033733344733139, 0.3140500001549052))
        (0.32207410281368043, 0.33156550013623537)

    :param uv: *CIE Luv u'v'* chromaticity coordinate.
    :type uv: array_like
    :return: *xy* chromaticity coordinates.
    :rtype: tuple

    :note: Input *u'v'* is in domain [0, 1].
    :note: Output *xy* is in domain [0, 1].

    References:

    -  http://en.wikipedia.org/wiki/CIELUV#The_reverse_transformation \
    (Last accessed 24 February 2014)
    """

    return (9. * uv[0] / (6. * uv[0] - 16. * uv[1] + 12.), 4. * uv[1] /
            (6. * uv[0] - 16. * uv[1] + 12.))


def Luv_to_LCHuv(Luv):
    """
    Converts from *CIE Luv* colourspace to *CIE LCHuv* colourspace.

    Examples::

        >>> Luv_to_LCHuv(np.array([100., -20.04304247, -19.81676035]))
        array([[ 100.        ]
               [  28.18559104]
               [ 224.6747382 ]])

    :param Luv: *CIE Luv* colourspace matrix.
    :type Luv: array_like (3, 1)
    :return: *CIE LCHuv* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *L\** is in domain [0, 100].

    References:

    -  http://www.brucelindbloom.com/Eqn_Luv_to_LCH.html \
    (Last accessed 24 February 2014)
    """

    L, u, v = np.ravel(Luv)

    H = 180. * math.atan2(v, u) / math.pi
    if H < 0.:
        H += 360.

    return np.array([L, math.sqrt(u ** 2 + v ** 2), H]).reshape((3, 1))


def LCHuv_to_Luv(LCHuv):
    """
    Converts from *CIE LCHuv* colourspace to *CIE Luv* colourspace.

    Examples::

        >>> LCHuv_to_Luv(np.array([100., 28.18559104, 224.6747382]))
        array([[ 100.        ]
               [ -20.04304247]
               [ -19.81676035]])

    :param LCHuv: *CIE LCHuv* colourspace matrix.
    :type LCHuv: array_like (3, 1)
    :return: *CIE Luv* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *L\** is in domain [0, 100].

    References:

    -  http://www.brucelindbloom.com/Eqn_LCH_to_Luv.html \
    (Last accessed 24 February 2014)
    """

    L, C, H = np.ravel(LCHuv)

    return np.array([L,
                     C * math.cos(math.radians(H)),
                     C * math.sin(math.radians(H))]).reshape((3, 1))