#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**difference.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour *difference* computation objects.

**Others:**

"""

from __future__ import unicode_literals

import math
import numpy

import colour.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal - Michael Parsons - The Moving picture Company"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["delta_E_CIE_1976",
           "delta_E_CIE_1994",
           "delta_E_CIE_2000",
           "delta_E_CMC"]

LOGGER = colour.utilities.verbose.install_logger()


def delta_E_CIE_1976(lab1, lab2):
    """
    Returns the difference between two given *CIE Lab* colours using *CIE 1976* recommendation.

    References:

    -  http://brucelindbloom.com/Eqn_DeltaE_CIE76.html

    Usage::

        >>> lab1 = numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1))
        >>> lab2 = numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))
        >>> delta_E_CIE_1976(lab1, lab2)
        451.713301974

    :param lab1: *CIE Lab* colour 1.
    :type lab1: matrix (3x1)
    :param lab2: *CIE Lab* colour 2.
    :type lab2: matrix (3x1)
    :return: Colour difference.
    :rtype: float
    """

    return numpy.linalg.norm(numpy.array(lab1) - numpy.array(lab2))


def delta_E_CIE_1994(lab1, lab2, textiles=True):
    """
    Returns the difference between two given *CIE Lab* colours using *CIE 1994* recommendation.

    References:

    -  http://brucelindbloom.com/Eqn_DeltaE_CIE94.html

    Usage::

        >>> lab1 = numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1))
        >>> lab2 = numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))
        >>> delta_E_CIE_1994(lab1, lab2)
        88.3355530575

    :param lab1: *CIE Lab* colour 1.
    :type lab1: matrix (3x1)
    :param lab2: *CIE Lab* colour 2.
    :type lab2: matrix (3x1)
    :param textiles: Application specific weights.
    :type textiles: bool
    :return: Colour difference.
    :rtype: float
    """

    k1 = 0.048 if textiles else 0.045
    k2 = 0.014 if textiles else 0.015
    kL = 2. if textiles else 1.
    kC = 1.
    kH = 1.

    L1, a1, b1 = numpy.ravel(lab1)
    L2, a2, b2 = numpy.ravel(lab2)

    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)

    sL = 1
    sC = 1 + k1 * C1
    sH = 1 + k2 * C1

    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_A = a1 - a2
    delta_B = b1 - b2

    try:
        delta_H = math.sqrt(delta_A ** 2 + delta_B ** 2 - delta_C ** 2)
    except ValueError:
        delta_H = 0.0

    L = (delta_L / (kL * sL)) ** 2
    C = (delta_C / (kC * sC)) ** 2
    H = (delta_H / (kH * sH)) ** 2

    return math.sqrt(L + C + H)


def delta_E_CIE_2000(lab1, lab2):
    """
    Returns the difference between two given *CIE Lab* colours using *CIE 2000* recommendation.

    References:

    -  http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html

    Usage::

        >>> lab1 = numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1))
        >>> lab2 = numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))
        >>> delta_E_CIE_2000(lab1, lab2)
        94.0356490267

    :param lab1: *CIE Lab* colour 1.
    :type lab1: matrix (3x1)
    :param lab2: *CIE Lab* colour 2.
    :type lab2: matrix (3x1)
    :return: Colour difference.
    :rtype: float
    """

    L1, a1, b1 = numpy.ravel(lab1)
    L2, a2, b2 = numpy.ravel(lab2)

    kL = 1.
    kC = 1.
    kH = 1.

    l_bar_prime = 0.5 * (L1 + L2)

    c1 = math.sqrt(a1 * a1 + b1 * b1)
    c2 = math.sqrt(a2 * a2 + b2 * b2)

    c_bar = 0.5 * (c1 + c2)
    c_bar7 = math.pow(c_bar, 7)

    g = 0.5 * (1. - math.sqrt(c_bar7 / (c_bar7 + 25. ** 7)))

    a1_prime = a1 * (1. + g)
    a2_prime = a2 * (1. + g)
    c1_prime = math.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = math.sqrt(a2_prime * a2_prime + b2 * b2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    h1_prime = (math.atan2(b1, a1_prime) * 180.) / math.pi
    if h1_prime < 0.:
        h1_prime += 360.

    h2_prime = (math.atan2(b2, a2_prime) * 180.) / math.pi
    if h2_prime < 0.0:
        h2_prime += 360.

    h_bar_prime = 0.5 * (h1_prime + h2_prime + 360.) if math.fabs(h1_prime -
                                                                  h2_prime) > 180. else 0.5 * (h1_prime + h2_prime)
    t = 1. - 0.17 * math.cos(math.pi * (h_bar_prime - 30.) / 180.) + 0.24 * math.cos(
        math.pi * (2. * h_bar_prime) / 180.) + \
        0.32 * math.cos(math.pi * (3. * h_bar_prime + 6.) / 180.) - 0.20 * math.cos(
        math.pi * (4. * h_bar_prime - 63.) / 180.)

    if math.fabs(h2_prime - h1_prime) <= 180.:
        delta_h_prime = h2_prime - h1_prime
    else:
        delta_h_prime = h2_prime - h1_prime + 360. if h2_prime <= h1_prime else h2_prime - h1_prime - 360.

    delta_L_prime = L2 - L1
    delta_C_prime = c2_prime - c1_prime
    delta_H_prime = 2. * math.sqrt(c1_prime * c2_prime) * math.sin(math.pi * (0.5 * delta_h_prime) / 180.)

    sL = 1. + ((0.015 * (l_bar_prime - 50.) * (l_bar_prime - 50.)) /
               math.sqrt(20. + (l_bar_prime - 50.) * (l_bar_prime - 50.)))
    sC = 1. + 0.045 * c_bar_prime
    sH = 1. + 0.015 * c_bar_prime * t

    delta_theta = 30. * math.exp(-((h_bar_prime - 275.) / 25.) * ((h_bar_prime - 275.) / 25.))

    c_bar_prime7 = c_bar_prime * c_bar_prime * c_bar_prime * c_bar_prime * c_bar_prime * c_bar_prime * c_bar_prime

    rC = math.sqrt(c_bar_prime7 / (c_bar_prime7 + 25. ** 7))
    rT = -2. * rC * math.sin(math.pi * (2. * delta_theta) / 180.)

    return math.sqrt((delta_L_prime / (kL * sL)) * (delta_L_prime / (kL * sL)) +
                     (delta_C_prime / (kC * sC)) * (delta_C_prime / (kC * sC)) +
                     (delta_H_prime / (kH * sH)) * (delta_H_prime / (kH * sH)) +
                     (delta_C_prime / (kC * sC)) * (delta_H_prime / (kH * sH)) * rT)


def delta_E_CMC(lab1, lab2, l=2., c=1.):
    """
    Returns the difference between two given *CIE Lab* colours using *Colour Measurement Committee* recommendation.
    The quasimetric has two parameters: *Lightness* (l) and *chroma* (c), allowing the users to weight the difference based on the ratio of l:c.
    Commonly used values are 2:1 for acceptability and 1:1 for the threshold of imperceptibility.

    References:

    -  http://brucelindbloom.com/Eqn_DeltaE_CMC.html

    Usage::

        >>> lab1 = numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1))
        >>> lab2 = numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))
        >>> delta_E_CMC(lab1, lab2)
        172.704771287

    :param lab1: *CIE Lab* colour 1.
    :type lab1: matrix (3x1)
    :param lab2: *CIE Lab* colour 2.
    :type lab2: matrix (3x1)
    :param l: Lightness weighting factor.
    :type l: float
    :param c: Chroma weighting factor.
    :type c: float
    :return: Colour difference.
    :rtype: float
    """

    L1, a1, b1 = numpy.ravel(lab1)
    L2, a2, b2 = numpy.ravel(lab2)

    c1 = math.sqrt(a1 * a1 + b1 * b1)
    c2 = math.sqrt(a2 * a2 + b2 * b2)
    sl = 0.511 if L1 < 16. else (0.040975 * L1) / (1. + 0.01765 * L1)
    sc = 0.0638 * c1 / (1. + 0.0131 * c1) + 0.638
    h1 = 0. if c1 < 0.000001 else (math.atan2(b1, a1) * 180.) / math.pi

    while h1 < 0.:
        h1 += 360.

    while h1 >= 360.:
        h1 -= 360.

    t = 0.56 + math.fabs(0.2 * math.cos((math.pi * (h1 + 168.)) / 180.)) if h1 >= 164. and h1 <= 345. else \
        0.36 + math.fabs(0.4 * math.cos((math.pi * (h1 + 35.)) / 180.))
    c4 = c1 * c1 * c1 * c1
    f = math.sqrt(c4 / (c4 + 1900.))
    sh = sc * (f * t + 1. - f)

    delta_L = L1 - L2
    delta_C = c1 - c2
    delta_A = a1 - a2
    delta_B = b1 - b2
    delta_H2 = delta_A * delta_A + delta_B * delta_B - delta_C * delta_C

    v1 = delta_L / (l * sl)
    v2 = delta_C / (c * sc)
    v3 = sh

    return math.sqrt(v1 * v1 + v2 * v2 + (delta_H2 / (v3 * v3)))
