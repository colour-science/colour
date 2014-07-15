# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**lightness.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Luminance*, *Munsell value* and *Lightness* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.computation.colourspaces.rgb.derivation
import colour.utilities.verbose


__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["CIE_E",
           "CIE_K",
           "get_luminance_equation",
           "get_luminance",
           "luminance_1943",
           "luminance_1976",
           "lightness_1958",
           "lightness_1964",
           "lightness_1976",
           "LIGHTNESS_FUNCTIONS",
           "get_lightness"]

LOGGER = colour.utilities.verbose.install_logger()

CIE_E = 216. / 24389.0
CIE_K = 24389. / 27.0


def get_luminance_equation(primaries, whitepoint):
    """
    Returns the *luminance equation* from given *primaries* and *whitepoint* matrices.

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: 3.3.8 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Usage::

        >>> primaries = numpy.matrix([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]).reshape((3, 2))
        >>> whitepoint = (0.32168, 0.33767)
        >>> get_luminance_equation(primaries, whitepoint)
        Y = 0.343966449765(R) + 0.728166096613(G) + -0.0721325463786(B)

    :param primaries: Primaries chromaticity coordinate matrix.
    :type primaries: matrix (3x2)
    :param whitepoint: Illuminant / whitepoint chromaticity coordinates.
    :type whitepoint: tuple
    :return: *Luminance* equation.
    :rtype: unicode
    """

    return "Y = {0}(R) + {1}(G) + {2}(B)".format(
        *numpy.ravel(
            colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix(primaries, whitepoint))[3:6])


def get_luminance(RGB, primaries, whitepoint):
    """
    Returns the *luminance* of given *RGB* components from given *primaries* and *whitepoint* matrices.

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: 3.3.3 - 3.3.6 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Usage::

        >>> RGB = numpy.matrix([40.6, 4.2, 67.4]).reshape((3, 1))
        >>> primaries = numpy.matrix([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]).reshape((3, 2))
        >>> whitepoint = (0.32168, 0.33767)
        >>> get_luminance(primaries, whitepoint)
        12.1616018403

    :param RGB: *RGB* chromaticity coordinate matrix.
    :type RGB: matrix (3x1)
    :param primaries: Primaries chromaticity coordinate matrix.
    :type primaries: matrix (3x2)
    :param whitepoint: Illuminant / whitepoint chromaticity coordinates.
    :type whitepoint: tuple
    :return: *Luminance*.
    :rtype: float
    """

    R, G, B = numpy.ravel(RGB)
    X, Y, Z = numpy.ravel(
        colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix(primaries, whitepoint))[3:6]

    return X * R + Y * G + Z * B


def luminance_1943(V):
    """
    Returns the *luminance* *Y* of given *Munsell value* *V* using 1943 *Newhall, Nickerson, and Judd* method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> luminance_1943(3.74629715382)
        10.4089874577

    :param V: *Munsell value* *V*.
    :type V: float
    :return: *Luminance* *Y*.
    :rtype: float

    :note: *V* is in domain [0, 10].
    :note: *Y* is in domain [0, 100].
    """

    Y = 1.2219 * V - 0.23111 * (V * V) + 0.23951 * (V ** 3) - 0.021009 * (V ** 4) + 0.0008404 * (V ** 5)

    return Y


def luminance_1976(L, Yn=100.):
    """
    Returns the *luminance* *Y* of given *Lightness* (*L\**) with given reference white *luminance*.

    References:

    -  http://www.poynton.com/PDFs/GammaFAQ.pdf

    Usage::

        >>> luminance_1976(37.9856290977)
        10.08

    :param L: *Lightness* (*L\**)
    :type L: float
    :param Yn: White reference *luminance*.
    :type Yn: float
    :return: *Luminance* *Y*.
    :rtype: float

    :note: *L* is in domain [0, 10].
    :note: *Yn* is in domain [0, 100].
    """

    Y = (((L + 16.) / 116.) ** 3.) * Yn if L > CIE_K * CIE_E else (L / CIE_K) * Yn

    return Y


def lightness_1958(Y):
    """
    Returns the *Lightness* (*L\**) of given *luminance* *Y* using 1958 *Glasser et al.* method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> lightness_1958(10.08)
        36.2505626458

    :param Y: Luminance.
    :type Y: float
    :return: *Lightness* *L\**.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *L\** is in domain [0, 100].
    """

    L_star = 25.29 * (Y ** (1. / 3.)) - 18.38

    return L_star


def lightness_1964(Y):
    """
    Returns the *Lightness* (*W\**) of given *luminance* *Y* using 1964 *Wyszecki* method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> lightness_1964(10.08)
        37.0041149128

    :param Y: Luminance.
    :type Y: float
    :return: *Lightness* *W\**.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *W\** is in domain [0, 100].
    """

    if not 1. < Y < 98.:
        LOGGER.warning(
            "!> {0} | 'W*' lightness calculation is only applicable for 1% < 'Y' < 98%, unpredictable results may occur!".format(
                __name__))

    W = 25. * (Y ** (1. / 3.)) - 17.

    return W


def lightness_1976(Y, Yn=100.):
    """
    Returns the *Lightness* (*L\**) of given *luminance* *Y* using given reference white *luminance*.

    References:

    -  http://www.poynton.com/PDFs/GammaFAQ.pdf

    Usage::

        >>> lightness_1976(10.08, 100.)
        37.9856290977

    :param Y: *Luminance* *Y*.
    :type Y: float
    :param Yn: White reference *luminance*.
    :type Yn: float
    :return: *Lightness* *L\**.
    :rtype: float

    :note: *Y* and *Yn* are in domain [0, 100].
    :note: *L\** is in domain [0, 100].
    """

    ratio = Y / Yn
    L = CIE_K * ratio if ratio <= CIE_E else 116. * ratio ** (1. / 3.) - 16

    return L


LIGHTNESS_FUNCTIONS = {"Lightness 1958": lightness_1958,
                       "Lightness 1964": lightness_1964,
                       "Lightness 1976": lightness_1976}


def get_lightness(Y, Yn=100., method="Lightness 1976"):
    """
    Returns the *Lightness* (*L\**) of given *luminance* *Y* using given reference white *luminance*.

    References:

    -  http://en.wikipedia.org/wiki/Lightness
    -  http://www.poynton.com/PDFs/GammaFAQ.pdf

    Usage::

        >>> get_lightness(10.08, 100)
        37.9856290977

    :param Y: *Luminance* *Y*.
    :type Y: float
    :param Yn: White reference *luminance*.
    :type Yn: float
    :return: *Lightness* *L\**.
    :type method: unicode ("Lightness 1958", "Lightness 1964", "Lightness 1976")
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* and *Yn* are in domain [0, 100].
    :note: *L\** is in domain [0, 100].
    """

    if Yn is None:
        return LIGHTNESS_FUNCTIONS.get(method)(Y)
    else:
        return lightness_1976(Y, Yn)


def get_lightness(Y, Yn=100., method="Lightness 1976"):
    """
    Returns the *Lightness* (*L\**) of given *luminance* *Y* using given reference white *luminance*.

    References:

    -  http://en.wikipedia.org/wiki/Lightness
    -  http://www.poynton.com/PDFs/GammaFAQ.pdf

    Usage::

        >>> get_lightness(10.08, 100)
        37.9856290977

    :param Y: *Luminance* *Y*.
    :type Y: float
    :param Yn: White reference *luminance*.
    :type Yn: float
    :return: *Lightness* *L\**.
    :type method: unicode ("Lightness 1958", "Lightness 1964", "Lightness 1976")
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* and *Yn* are in domain [0, 100].
    :note: *L\** is in domain [0, 100].
    """

    if Yn is None:
        return LIGHTNESS_FUNCTIONS.get(method)(Y)
    else:
        return lightness_1976(Y, Yn)
