# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**munsell.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Munsell Renotation Sytem* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import math
import re

import colour.computation.luminance
import colour.utilities.exceptions
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

__FPNP = "[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
MUNSELL_RENOTATION_SYSTEM_GRAY_PATTERN = "N(?P<value>{0})".format(__FPNP)
MUNSELL_RENOTATION_SYSTEM_COLOR_PATTERN = \
    "(?P<hue>{0})\s*(?P<letter>BG|GY|YR|RP|PB|B|G|Y|R|P)\s*(?P<value>{0})\s*\/\s*(?P<chroma>[-+]?{0})".format(__FPNP)

MUNSELL_HUE_LETTER_CODES = {
    "BG": 2,
    "GY": 4,
    "YR": 6,
    "RP": 8,
    "PB": 10,
    "B": 1,
    "G": 3,
    "Y": 5,
    "R": 7,
    "P": 9}


def parse_munsell_color(munsell_color):
    match = re.match(MUNSELL_RENOTATION_SYSTEM_GRAY_PATTERN, munsell_color, flags=re.IGNORECASE)
    if match:
        return (float(match.group("value")),)
    match = re.match(MUNSELL_RENOTATION_SYSTEM_COLOR_PATTERN, munsell_color, flags=re.IGNORECASE)
    if match:
        return (float(match.group("hue")),
                float(match.group("value")),
                float(match.group("chroma")),
                MUNSELL_HUE_LETTER_CODES.get(match.group("letter").upper()))

    raise colour.utilities.exceptions.ProgrammingError(
        "{0} is not a valid 'Munsell Renotation Sytem' color specification!".format(munsell_color))


def normalize_munsell_color_specification(specification):
    if len(specification) == 1:
        return specification
    else:
        hue, value, chroma, code = specification
        if hue == 0:
            # 0YR is equivalent to 10R.
            hue, code = 10., (code + 1) % 10
        return (value, ) if chroma == 0 else (hue, value, chroma, code)


def munsell_color_to_xyY(munsell_color):
    specification = normalize_munsell_color_specification(parse_munsell_color(munsell_color))
    if not len(specification) == 4:
        return # raise Mmmmh :)

    hue, value, chroma, code = specification
    if  not 0 < hue < 10:
        return # raise Hue should be in domain [0, 10]

    if  not 0 < value < 10:
        return # raise Value should be in domain [0, 10]

    Y = colour.computation.luminance.luminance_ASTM_D1535_08(value)
    print Y
    return hue, value, chroma, code


from colour.dataset.munsell import MUNSELL_COLORS

print munsell_color_to_xyY("N5")
print munsell_color_to_xyY("4.2R8.1/5.3")
print munsell_color_to_xyY("0YR8.1/5.3")
print munsell_color_to_xyY("10R8.1/5.3")
print munsell_color_to_xyY("4.2R8.1/0")


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
    :param method: Computation method.
    :type method: unicode ("Munsell Value 1920", "Munsell Value 1933", "Munsell Value 1943", "Munsell Value 1944", "Munsell Value 1955")
    :return: *Munsell value* *V*.
    :rtype: float

    :note: *Y* is in domain [0, 100].
    :note: *V* is in domain [0, 10].
    """

    return MUNSELL_VALUE_FUNCTIONS.get(method)(Y)

