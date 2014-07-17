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
import sys

import colour.algebra.common
import colour.computation.luminance
import colour.dataset.illuminants.chromaticity_coordinates
import colour.utilities.exceptions
import colour.utilities.verbose
from colour.dataset.munsell import MUNSELL_COLOURS
from colour.utilities.data_structures import Lookup


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
MUNSELL_RENOTATION_SYSTEM_COLOUR_PATTERN = \
    "(?P<hue>{0})\s*(?P<letter>BG|GY|YR|RP|PB|B|G|Y|R|P)\s*(?P<value>{0})\s*\/\s*(?P<chroma>[-+]?{0})".format(__FPNP)

MUNSELL_RENOTATION_SYSTEM_GRAY_FORMAT = "N{0}"
MUNSELL_RENOTATION_SYSTEM_COLOUR_FORMAT = "{0} {1}/{2}"
MUNSELL_RENOTATION_SYSTEM_GRAY_EXTENDED_FORMAT = "N{0:{1}}"
MUNSELL_RENOTATION_SYSTEM_COLOUR_EXTENDED_FORMAT = "{0:{1}}{2} {3:{4}}/{5:{6}}"

MUNSELL_HUE_LETTER_CODES = Lookup({
    "BG": 2,
    "GY": 4,
    "YR": 6,
    "RP": 8,
    "PB": 10,
    "B": 1,
    "G": 3,
    "Y": 5,
    "R": 7,
    "P": 9})

EVEN_INTEGER_TRESHOLD = 0.001


def __get_munsell_specifications():
    specifications_attribute = "__MUNSELL_SPECIFICATIONS"
    module = sys.modules[__name__]
    if not hasattr(module, specifications_attribute):
        specifications = [munsell_colour_to_specification(MUNSELL_RENOTATION_SYSTEM_COLOUR_FORMAT.format(*colour[0])) \
                          for colour in MUNSELL_COLOURS]
        setattr(module, specifications_attribute, specifications)
    return getattr(module, specifications_attribute)


def parse_munsell_colour(munsell_color):
    match = re.match(MUNSELL_RENOTATION_SYSTEM_GRAY_PATTERN, munsell_color, flags=re.IGNORECASE)
    if match:
        return float(match.group("value"))
    match = re.match(MUNSELL_RENOTATION_SYSTEM_COLOUR_PATTERN, munsell_color, flags=re.IGNORECASE)
    if match:
        return (float(match.group("hue")),
                float(match.group("value")),
                float(match.group("chroma")),
                MUNSELL_HUE_LETTER_CODES.get(match.group("letter").upper()))

    raise colour.utilities.exceptions.ProgrammingError(
        "{0} is not a valid 'Munsell Renotation Sytem' colour specification!".format(munsell_color))


def normalize_munsell_colour_specification(specification):
    if is_grey_munsell_color(specification):
        return specification
    else:
        hue, value, chroma, code = specification
        if hue == 0:
            # 0YR is equivalent to 10R.
            hue, code = 10., (code + 1) % 10
        return value if chroma == 0 else (hue, value, chroma, code)


def munsell_colour_to_specification(munsell_colour):
    return normalize_munsell_colour_specification(parse_munsell_colour(munsell_colour))


def is_grey_munsell_color(specification):
    return colour.algebra.common.is_number(specification)


def specification_to_munsell_colour(specification, hue_decimals=2, value_decimals=2, chroma_decimals=2):
    if is_grey_munsell_color(specification):
        return MUNSELL_RENOTATION_SYSTEM_GRAY_EXTENDED_FORMAT.format(specification, value_decimals)
    else:
        hue, value, chroma, code = specification
        if value == 0:
            return MUNSELL_RENOTATION_SYSTEM_GRAY_EXTENDED_FORMAT.format(specification, value_decimals)
        else:
            letter = MUNSELL_HUE_LETTER_CODES.get_first_key_from_value(code)
            return MUNSELL_RENOTATION_SYSTEM_COLOUR_EXTENDED_FORMAT.format(hue,
                                                                           hue_decimals,
                                                                           letter,
                                                                           value,
                                                                           value_decimals,
                                                                           chroma,
                                                                           chroma_decimals)


def get_xyY_from_renotation(specification):
    specifications = __get_munsell_specifications()
    try:
        return MUNSELL_COLOURS[specifications.index(specification)][1]
    except ValueError as error:
        # TODO: Raise for specification not in renotation
        raise


def get_bounding_hues_from_renotation(hue, code):
    if hue % 2.5 == 0:
        if hue == 0:
            hue_cw = 10
            code_cw = (code + 1) % 10
        else:
            hue_cw = hue
            code_cw = code
        hue_ccw = hue_cw
        code_ccw = code_cw
    else:
        hue_cw = 2.5 * math.floor(hue / 2.5)
        hue_ccw = (hue_cw + 2.5) % 10
        if hue_ccw == 0:
            hue_ccw = 10
        code_ccw = code

        if hue_cw == 0:
            hue_cw = 10
            code_cw = (code + 1) % 10
            if code_cw == 0:
                code_cw = 10
        else:
            code_cw = code
        code_ccw = code

    return (hue_cw, code_cw), (hue_ccw, code_ccw)


def get_xy_from_renotation_ovoid(specification):
    if is_grey_munsell_color(specification):
        return colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
            "CIE 1931 2 Degree Standard Observer").get("C")
    else:
        hue, value, chroma, code = specification
        if not 1 <= value <= 9:
            # TODO: Raise for value not in domain [1, 9]
            raise

        if abs(value - round(value)) > EVEN_INTEGER_TRESHOLD:
            # TODO: Raise for value not being even integer.
            raise

        value = round(value)

        if chroma < 2:
            # TODO: Raise for chroma not being in domain [2, :]
            raise

        if abs(2 * (chroma / 2 - round(chroma / 2))) > EVEN_INTEGER_TRESHOLD:
            # TODO: Raise for chroma not being even integer.
            raise

        chroma = 2 * round(chroma / 2)

        # Checking if renotation data is available without interpolation using given treshold.
        threshold = 0.001
        if abs(hue) < threshold or \
                        abs(hue - 2.5) < threshold or \
                        abs(hue - 5) < threshold or \
                        abs(hue - 7.5) < threshold or \
                        abs(hue - 10) < threshold:
            hue = 2.5 * round(hue / 2.5)
            x, y, Y = get_xyY_from_renotation((hue, value, chroma, code))
            return x, y

        hue_cw, hue_ccw = get_bounding_hues_from_renotation(hue, code)
        hue_minus, code_minus = hue_cw
        hue_plus, code_plus = hue_ccw

        specification_minus = (hue_minus, value, chroma, code_minus)
        specification_plus = (hue_plus, value, chroma, code_plus)

        x_grey, y_grey = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
            "CIE 1931 2 Degree Standard Observer").get("C")

        x_plus, y_plus, Y_plus = get_xyY_from_renotation(specification_plus)
        print x_plus, y_plus, Y_plus
        return -1, -1


def munsell_color_to_xy(specification):
    # print specification_to_munsell_colour(specification)
    if is_grey_munsell_color(specification):
        # ## return get_xyY_from_renotation ---> MunsellToxyYfromExtrapolatedRenotation
        pass
    else:
        hue, value, chroma, code = specification
        if chroma % 2 == 0:
            chroma_minus = plus_chroma = chroma
        else:
            chroma_minus = 2 * math.floor(chroma / 2)
            value_plus = chroma_minus + 2

        if chroma_minus == 0:
            # Smallest chroma ovoid collapses to illuminant chromaticity coordinates.
            x_minus, y_minus = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                "CIE 1931 2 Degree Standard Observer").get("C")
        else:
            x_minus, y_minus = get_xy_from_renotation_ovoid((hue, value, chroma_minus, code))
            print x_minus, y_minus
    return None, None


def munsell_color_to_xyY(munsell_color):
    specification = munsell_colour_to_specification(munsell_color)

    if is_grey_munsell_color(specification):
        value = specification
    else:
        hue, value, chroma, code = specification

        if not 0 <= hue <= 10:
            # TODO: Raise for hue not in domain [0, 10]
            raise

        if not 0 <= value <= 10:
            # TODO: Raise for value not in domain [0, 10]
            raise

    Y = colour.computation.luminance.luminance_ASTM_D1535_08(value)

    if abs(value - round(value)) < EVEN_INTEGER_TRESHOLD:
        value_minus = value_plus = round(value)
    else:
        value_minus = math.floor(value)
        value_plus = value + 1

    minus_specification = value_minus if is_grey_munsell_color(specification) else (hue, value_minus, chroma, code)

    x_minus, y_minus = munsell_color_to_xy(minus_specification)
    # return hue, value, chroma, code


# print munsell_color_to_xyY("N5")
# print munsell_color_to_xyY("4.2R 8.1/5.3")
print munsell_color_to_xyY("4.2YR 8.1/5.3")
# print munsell_color_to_xyY("0YR 8.1/5.3")
# munsell_color_to_xyY("10R 8.1/5.3")
# munsell_color_to_xyY(MUNSELL_RENOTATION_SYSTEM_COLOUR_FORMAT.format("10Y", 9.0, 2.0))
# print munsell_color_to_xyY("4.2R 8.1/0")

# ###############################################################################################################################################
# ###############################################################################################################################################
# ###############################################################################################################################################
# ###############################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

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

