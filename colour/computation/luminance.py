# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**luminance.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Luminance* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["CIE_E",
           "CIE_K",
           "luminance_1943",
           "luminance_1976",
           "luminance_ASTM_D1535_08",
           "LUMINANCE_FUNCTIONS",
           "get_luminance"]

CIE_E = 216. / 24389.0
CIE_K = 24389. / 27.0


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

    :note: *L* is in domain [0, 100].
    :note: *Yn* is in domain [0, 100].
    """

    Y = (((L + 16.) / 116.) ** 3.) * Yn if L > CIE_K * CIE_E else (L / CIE_K) * Yn

    return Y


def luminance_ASTM_D1535_08(V):
    """
    Returns the *luminance* *Y* of given *Munsell value* *V* using *ASTM D1535-08e1* method.

    References:

    -  http://www.scribd.com/doc/89648322/ASTM-D1535-08e1-Standard-Practice-for-Specifying-Color-by-the-Munsell-System

    Usage::

        >>> luminance_ASTM_D1535_08(3.74629715382)
        10.1488096782

    :param V: *Munsell value* *V*.
    :type V: float
    :return: *Luminance* *Y*.
    :rtype: float

    :note: *V* is in domain [0, 10].
    :note: *Y* is in domain [0, 100].
    """

    Y = 1.1914 * V - 0.22533 * (V * V) + 0.23352 * (V ** 3) - 0.020484 * (V ** 4) + 0.00081939 * (V ** 5)

    return Y


LUMINANCE_FUNCTIONS = {"Luminance 1943": luminance_1943,
                       "Luminance 1976": luminance_1976,
                       "Luminance ASTM D1535-08": luminance_ASTM_D1535_08}


def get_luminance(LV, Yn=100., method="Luminance 1976"):
    """
    Returns the *luminance* *Y* of given *Lightness* *L* using given reference white *luminance* or given *Munsell value* *V*.

    References:

    -  http://en.wikipedia.org/wiki/Lightness
    -  http://www.poynton.com/PDFs/GammaFAQ.pdf
    -  http://www.scribd.com/doc/89648322/ASTM-D1535-08e1-Standard-Practice-for-Specifying-Color-by-the-Munsell-System

    Usage::

        >>> get_luminance(3.74629715382)
        37.9856290977

    :param LV: *Lightness* *L* or *Munsell value* *V*.
    :type LV: float
    :param Yn: White reference *luminance*.
    :type Yn: float
    :param method: Computation method.
    :type method: unicode ("Luminance 1943", "Luminance 1976", "Luminance ASTM D1535-08")
    :return: *Luminance* *Y*.
    :rtype: float

    :note: *LV* is in domain [0, 100] or [0, 10] and *Yn* is in domain [0, 100].
    :note: *Y\** is in domain [0, 100].
    """

    if Yn is None or method is not None:
        return LUMINANCE_FUNCTIONS.get(method)(LV)
    else:
        return luminance_1976(LV, Yn)
