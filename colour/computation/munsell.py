# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**munsell.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Munsell Color Theory* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import math

import colour.computation.colourspaces.rgb.derivation
import colour.utilities.verbose


__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["munsell_value_1920",
           "munsell_value_1933",
           "munsell_value_1943",
           "munsell_value_1944",
           "munsell_value_1955",
           "MUNSELL_VALUE_FUNCTIONS",
           "get_munsell_value"]

LOGGER = colour.utilities.verbose.install_logger()


def munsell_value_1920(Y):
    """
    Returns the *Munsell value* *V* of given *luminance* *Y* using 1920 *Priest et al.* method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> munsell_value_1920(10.08)
        3.17490157328

    :param Y: *Luminance* *Y*.
    :type Y: float
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *V* is in domain [0, 10].
    """

    Y /= 100.
    V = 10. * math.sqrt(Y)

    return V


def munsell_value_1933(Y):
    """
    Returns the *Munsell value* *V* of given *luminance* *Y* using 1933 *Munsell, Sloan, and Godlove* method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> munsell_value_1933(10.08)
        3.79183555086

    :param Y: *Luminance* *Y*.
    :type Y: float
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *V* is in domain [0, 10].
    """

    V = math.sqrt(1.4742 * Y - 0.004743 * (Y * Y))

    return V


def munsell_value_1943(Y):
    """
    Returns the *Munsell value* *V* of given *luminance* *Y* using 1943 *Moon and Spencer* method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> munsell_value_1943(10.08)
        3.74629715382

    :param Y: *Luminance* *Y*.
    :type Y: float
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *V* is in domain [0, 10].
    """

    V = 1.4 * Y ** 0.426

    return V


def munsell_value_1944(Y):
    """
    Returns the *Munsell value* *V* of given *luminance* *Y* using 1944 *Saunderson and Milner* method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> munsell_value_1944(10.08)
        3.68650805994

    :param Y: *Luminance* *Y*.
    :type Y: float
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *V* is in domain [0, 10].
    """

    V = 2.357 * (Y ** 0.343) - 1.52

    return V


def munsell_value_1955(Y):
    """
    Returns the *Munsell value* *V* of given *luminance* *Y* using 1955 *Ladd and Pinney* method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> munsell_value_1955(10.08)
        3.69528622419

    :param Y: *Luminance* *Y*.
    :type Y: float
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *V* is in domain [0, 10].
    """

    V = 2.468 * (Y ** (1. / 3.)) - 1.636

    return V


MUNSELL_VALUE_FUNCTIONS = {"Munsell Value 1920": munsell_value_1920,
                           "Munsell Value 1933": munsell_value_1933,
                           "Munsell Value 1943": munsell_value_1943,
                           "Munsell Value 1944": munsell_value_1944,
                           "Munsell Value 1955": munsell_value_1955}


def get_munsell_value(Y, method="Munsell Value 1955"):
    """
    Returns the *Munsell value* *V* of given *luminance* *Y* using given method.

    References:

    -  http://en.wikipedia.org/wiki/Lightness

    Usage::

        >>> get_munsell_value(10.08)
        3.69528622419

    :param Y: *Luminance* *Y*.
    :type Y: float
    :param method: *Luminance* *Y*.
    :type method: unicode ("Munsell Value 1920", "Munsell Value 1933", "Munsell Value 1943", "Munsell Value 1944", "Munsell Value 1955")
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *V* is in domain [0, 10].
    """

    return MUNSELL_VALUE_FUNCTIONS.get(method)(Y)
