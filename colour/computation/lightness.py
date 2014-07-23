# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**lightness.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Lightness* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import colour.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["CIE_E",
           "CIE_K",
           "lightness_glasser1958",
           "lightness_wyszecki1964",
           "lightness_1976",
           "LIGHTNESS_FUNCTIONS",
           "get_lightness"]

CIE_E = 216. / 24389.0
CIE_K = 24389. / 27.0


def lightness_glasser1958(Y):
    """
    Returns the *Lightness* (*L\**) of given *luminance* *Y* using *Glasser et al.* 1958 method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> lightness_glasser1958(10.08)
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


def lightness_wyszecki1964(Y):
    """
    Returns the *Lightness* (*W\**) of given *luminance* *Y* using *Wyszecki* 1964 method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> lightness_wyszecki1964(10.08)
        37.0041149128

    :param Y: Luminance.
    :type Y: float
    :return: *Lightness* *W\**.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *W\** is in domain [0, 100].
    """

    if not 1. < Y < 98.:
        colour.utilities.verbose.warning(
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


LIGHTNESS_FUNCTIONS = {"Lightness Glasser 1958": lightness_glasser1958,
                       "Lightness Wyszecki 1964": lightness_wyszecki1964,
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
    :param method: Computation method.
    :type method: unicode ("Lightness Glasser 1958", "Lightness Wyszecki 1964", "Lightness 1976")
    :return: *Lightness* *L\**.
    :rtype: float

    :note: *Y* and *Yn* are in domain [0, 100].
    :note: *L\** is in domain [0, 100].
    """

    if Yn is None or method is not None:
        return LIGHTNESS_FUNCTIONS.get(method)(Y)
    else:
        return lightness_1976(Y, Yn)