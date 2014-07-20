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
import numpy
import re
import sys

import colour.algebra.common
import colour.algebra.coordinates.transformations
import colour.computation.luminance
import colour.dataset.illuminants.chromaticity_coordinates
import colour.utilities.exceptions
import colour.utilities.verbose
from colour.algebra.interpolation import LinearInterpolator
from colour.dataset.munsell import MUNSELL_COLOURS
from colour.utilities.data_structures import Lookup


__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["MUNSELL_RENOTATION_SYSTEM_GRAY_PATTERN",
           "MUNSELL_RENOTATION_SYSTEM_COLOUR_PATTERN",
           "MUNSELL_RENOTATION_SYSTEM_GRAY_FORMAT",
           "MUNSELL_RENOTATION_SYSTEM_COLOUR_FORMAT",
           "MUNSELL_RENOTATION_SYSTEM_GRAY_EXTENDED_FORMAT",
           "MUNSELL_RENOTATION_SYSTEM_COLOUR_EXTENDED_FORMAT",
           "MUNSELL_HUE_LETTER_CODES",
           "EVEN_INTEGER_THRESHOLD",
           "MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES",
           "parse_munsell_colour",
           "is_grey_munsell_colour",
           "normalize_munsell_specification",
           "munsell_colour_to_munsell_specification",
           "munsell_specification_to_munsell_colour",
           "get_xyY_from_renotation",
           "is_specification_in_renotation",
           "get_bounding_hues_from_renotation",
           "hue_to_hue_angle",
           "hue_to_ASTM_hue",
           "get_interpolation_method_from_renotation_ovoid",
           "get_xy_from_renotation_ovoid",
           "munsell_specification_to_xy",
           "munsell_colour_to_xyY",
           "munsell_value_1920",
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
MUNSELL_RENOTATION_SYSTEM_GRAY_EXTENDED_FORMAT = "N{0:.{1}f}"
MUNSELL_RENOTATION_SYSTEM_COLOUR_EXTENDED_FORMAT = "{0:.{1}f}{2} {3:.{4}f}/{5:.{6}f}"

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

EVEN_INTEGER_THRESHOLD = 0.001

MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES = \
    colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get("CIE 1931 2 Degree Standard Observer").get("C")


def __get_munsell_specifications():
    """
    Returns the *Munsell Renotation Sytem* specifications.
    The *Munsell Renotation Sytem* data is stored in :attr:`colour.MUNSELL_COLOURS` attribute in a 2 columns form:

    (("2.5GY", 0.2, 2.0), (0.713, 1.414, 0.237)),
    (("5GY", 0.2, 2.0), (0.449, 1.145, 0.237)),
    (("7.5GY", 0.2, 2.0), (0.262, 0.837, 0.237)),
    ...,)

    The first column is converted from *Munsell* colour to specification using :def:`munsell_colour_to_munsell_specification`
    definition:

    ("2.5GY", 0.2, 2.0) ---> (2.5, 0.2, 2.0, 4)


    :return: *Munsell Renotation Sytem* specifications.
    :rtype: list
    """

    specifications_attribute = "__MUNSELL_SPECIFICATIONS"
    module = sys.modules[__name__]
    if not hasattr(module, specifications_attribute):
        specifications = [
            munsell_colour_to_munsell_specification(MUNSELL_RENOTATION_SYSTEM_COLOUR_FORMAT.format(*colour[0])) \
            for colour in MUNSELL_COLOURS]
        setattr(module, specifications_attribute, specifications)
    return getattr(module, specifications_attribute)


def parse_munsell_colour(munsell_colour):
    """
    Parses given *Munsell* colour and returns an intermediate *Munsell* specification.

    Usage::

        >>> parse_munsell_colour("N5.2")
        5.2
        >>> parse_munsell_colour("0YR 2.0/4.0")
        (0.0, 2.0, 4.0, 6)

    :param munsell_colour: *Munsell* colour.
    :type munsell_colour: unicode
    :return: Intermediate *Munsell* specification.
    :rtype: tuple or float
    """

    match = re.match(MUNSELL_RENOTATION_SYSTEM_GRAY_PATTERN, munsell_colour, flags=re.IGNORECASE)
    if match:
        return float(match.group("value"))
    match = re.match(MUNSELL_RENOTATION_SYSTEM_COLOUR_PATTERN, munsell_colour, flags=re.IGNORECASE)
    if match:
        return (float(match.group("hue")),
                float(match.group("value")),
                float(match.group("chroma")),
                MUNSELL_HUE_LETTER_CODES.get(match.group("letter").upper()))

    raise colour.utilities.exceptions.ProgrammingError(
        "'{0}' is not a valid 'Munsell Renotation Sytem' colour specification!".format(munsell_colour))


def is_grey_munsell_colour(specification):
    """
    Returns if given *Munsell* specification is a single number form used for grey colour.

    Usage::

        >>> is_grey_munsell_colour((0.0, 2.0, 4.0, 6))
        False
        >>> is_grey_munsell_colour(0.5)
        True

    :param specification: *Munsell* specification.
    :type specification: tuple
    :return: Is specification a grey colour.
    :rtype: bool
    """

    return colour.algebra.common.is_number(specification)


def normalize_munsell_specification(specification):
    """
    Normalises given *Munsell* specification.

    Usage::

        >>> normalize_munsell_specification((0.0, 2.0, 4.0, 6))
        (10.0, 2.0, 4.0, 7)

    :param specification: *Munsell* specification.
    :type specification: tuple or float
    :return: Normalised *Munsell* specification.
    :rtype: tuple or float
    """

    if is_grey_munsell_colour(specification):
        return specification
    else:
        hue, value, chroma, code = specification
        if hue == 0:
            # 0YR is equivalent to 10R.
            hue, code = 10., (code + 1) % 10
        return value if chroma == 0 else (hue, value, chroma, code)


def munsell_colour_to_munsell_specification(munsell_colour):
    """
    Convenient definition to retrieve a normalised *Munsell* specification from given *Munsell* colour.

    Usage::

        >>> munsell_colour_to_munsell_specification("N5.2")
        5.2
        >>> munsell_colour_to_munsell_specification("0YR 2.0/4.0")
        (10.0, 2.0, 4.0, 7)

    :param munsell_colour: *Munsell* colour.
    :type munsell_colour: unicode
    :return: Normalised *Munsell* specification.
    :rtype: tuple or float
    """

    return normalize_munsell_specification(parse_munsell_colour(munsell_colour))


def munsell_specification_to_munsell_colour(specification, hue_decimals=1, value_decimals=1, chroma_decimals=1):
    """
    Converts from *Munsell* specification to given *Munsell* colour.

    Usage::

        >>> munsell_specification_to_munsell_colour(5.2)
        N5.2
        >>> munsell_specification_to_munsell_colour((10., 2.0, 4.0, 7))
        10.0R 2.0/4.0

    :param specification: *Munsell* specification.
    :type specification: specification
    :param hue_decimals: Hue formatting decimals.
    :type hue_decimals: int
    :param value_decimals: Value formatting decimals.
    :type value_decimals: int
    :param chroma_decimals: Chroma formatting decimals.
    :type chroma_decimals: int
    :return: *Munsell* colour.
    :rtype: unicode
    """

    if is_grey_munsell_colour(specification):
        return MUNSELL_RENOTATION_SYSTEM_GRAY_EXTENDED_FORMAT.format(specification, value_decimals)
    else:
        hue, value, chroma, code = specification
        code_values = MUNSELL_HUE_LETTER_CODES.values()

        assert 2.5 <= hue <= 10, "'{0}' specification hue must be in domain [2.5, 10]!".format(specification)
        assert 0 <= value <= 10, "'{0}' specification value must be in domain [0, 10]!".format(specification)
        assert 2 <= chroma <= 50, "'{0}' specification chroma must be in domain [2, 50]!".format(specification)
        assert code in code_values, "'{0}' specification code must one of '{1}'!".format(specification, code_values)

        if hue == 0:
            hue, code = 10, (code + 1) % 10

        if value == 0:
            return MUNSELL_RENOTATION_SYSTEM_GRAY_EXTENDED_FORMAT.format(specification, value_decimals)
        else:
            hue_letter = MUNSELL_HUE_LETTER_CODES.get_first_key_from_value(code)
            return MUNSELL_RENOTATION_SYSTEM_COLOUR_EXTENDED_FORMAT.format(hue,
                                                                           hue_decimals,
                                                                           hue_letter,
                                                                           value,
                                                                           value_decimals,
                                                                           chroma,
                                                                           chroma_decimals)


def get_xyY_from_renotation(specification):
    """
    Returns given existing *Munsell* specification *CIE xyY* colourspace vector from *Munsell Renotation Sytem* data.

    Usage::

        >>> get_xyY_from_renotation((2.5, 0.2, 2.0, 4))
        (0.713, 1.414, 0.237)

    :param specification: *Munsell* specification.
    :type specification: specification
    :return: *CIE xyY* colourspace vector.
    :rtype: tuple
    """

    specifications = __get_munsell_specifications()
    try:
        return MUNSELL_COLOURS[specifications.index(specification)][1]
    except ValueError as error:
        raise colour.utilities.exceptions.ProgrammingError(
            "'{0}' specification does not exists in 'Munsell Renotation Sytem' data!".format(specification))


def is_specification_in_renotation(specification):
    """
    Returns if given *Munsell* specification is in *Munsell Renotation Sytem* data.

    Usage::

        >>> is_specification_in_renotation((2.5, 0.2, 2.0, 4))
        True
        >>> is_specification_in_renotation((64, 0.2, 2.0, 4))
        False

    :param specification: *Munsell* specification.
    :type specification: specification
    :return: Is specification in *Munsell Renotation Sytem* data.
    :rtype: bool
    """

    try:
        get_xyY_from_renotation(specification)
        return True
    except colour.utilities.exceptions.ProgrammingError as error:
        return False


def get_bounding_hues_from_renotation(hue, code):
    """
    Returns for a given hue the two bounding hues from *Munsell Renotation Sytem* data.

    References:

    -  *The Munsell and Kubelka-Munk Toolbox*: \
    *MunsellAndKubelkaMunkToolboxApr2014/MunsellSystemRoutines/BoundingRenotationHues.m*

    Usage::

        >>> get_bounding_hues_from_renotation(3.2, 4)
        ((2.5, 4), (5.0, 4))

    :param hue: *Munsell* specification hue.
    :type hue: float
    :param code: *Munsell* specification code.
    :type code: float
    :return: Bounding hues.
    :rtype: tuple
    """

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


def hue_to_hue_angle(hue, code):
    """
    Converts from the *Munsell* specification hue to hue angle in degrees.

    References:

    -  *The Munsell and Kubelka-Munk Toolbox*: \
    *MunsellAndKubelkaMunkToolboxApr2014/MunsellRenotationRoutines/MunsellHueToChromDiagHueAngle.m*

    Usage::

        >>> hue_to_hue_angle(3.2, 4)
        65.5

    :param hue: *Munsell* specification hue.
    :type hue: float
    :param code: *Munsell* specification code.
    :type code: float
    :return: Hue angle in degrees.
    :rtype: float
    """

    single_hue = ((17 - code) % 10 + (hue / 10) - 0.5) % 10
    return float(LinearInterpolator(numpy.array([0, 2, 3, 4, 5, 6, 8, 9, 10]),
                                    numpy.array([0, 45, 70, 135, 160, 225, 255, 315, 360]))(single_hue))


def hue_to_ASTM_hue(hue, code):
    """
    Converts from the *Munsell* specification hue to *ASTM* hue number in domain [0, 100].

    References:

    -  *The Munsell and Kubelka-Munk Toolbox*: \
    *MunsellAndKubelkaMunkToolboxApr2014/MunsellRenotationRoutines/MunsellHueToASTMHue.m*

    Usage::

        >>> hue_to_ASTM_hue(3.2, 4)
        33.2

    :param hue: *Munsell* specification hue.
    :type hue: float
    :param code: *Munsell* specification code.
    :type code: float
    :return: *ASM* hue number.
    :rtype: float
    """

    ASTM_hue = 10 * ((7 - code) % 10) + hue
    return 100 if ASTM_hue == 0 else ASTM_hue


def get_interpolation_method_from_renotation_ovoid(specification):
    """
    Returns whether to use linear or radial interpolation when drawing ovoids through data points
    in the *Munsell Renotation Sytem* data from given specification.

    References:

    -  *The Munsell and Kubelka-Munk Toolbox*: \
    *MunsellAndKubelkaMunkToolboxApr2014/MunsellSystemRoutines/LinearVsRadialInterpOnRenotationOvoid.m*

    Usage::

        >>> get_interpolation_method_from_renotation_ovoid((2.5, 5.0, 12.0, 4))
        Radial

    :param specification: *Munsell* specification.
    :type specification: specification
    :return: Interpolation method.
    :rtype: unicode or None ("Linear", "Radial", None)

    :note: *Munsell* specification value must be an even integer in domain [0, 10].
    :note: *Munsell* specification chroma must be an even integer and a multiple of 2 in domain [2, 50].
    """

    interpolation_methods = {0: None,
                             1: "Linear",
                             2: "Radial"}
    interpolation_method = 0
    if is_grey_munsell_colour(specification):
        # No interpolation needed for grey colours.
        interpolation_method = 0
    else:
        hue, value, chroma, code = specification

        assert 0 <= value <= 10, "'{0}' specification value must be in domain [0, 10]!".format(specification)
        assert abs(value - round(value)) <= EVEN_INTEGER_THRESHOLD, \
            "'{0}' specification value must be an even integer!".format(specification)

        value = round(value)

        # Ideal white, no interpolation needed.
        if value == 10:
            interpolation_method = 0

        assert 2 <= chroma <= 50, "'{0}' specification chroma must be in domain [2, 50]!".format(specification)
        assert abs(2 * (chroma / 2 - round(chroma / 2))) <= EVEN_INTEGER_THRESHOLD, \
            "'{0}' specification chroma must be an even integer and multiple of 2!".format(specification)

        chroma = 2 * round(chroma / 2)

        # Standard Munsell Renotation Sytem hue, no interpolation needed.
        if hue % 2.5 == 0:
            interpolation_method = 0

        ASTM_hue = hue_to_ASTM_hue(hue, code)

        if value == 1:
            if chroma == 2:
                if 15 < ASTM_hue < 30 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 4:
                if 12.5 < ASTM_hue < 27.5 or 57.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6:
                if 55 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 8:
                if 67.5 < ASTM_hue < 77.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 10:
                if 72.5 < ASTM_hue < 77.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1
        elif value == 2:
            if chroma == 2:
                if 15 < ASTM_hue < 27.5 or 77.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 4:
                if 12.5 < ASTM_hue < 30 or 62.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6:
                if 7.5 < ASTM_hue < 22.5 or 62.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 8:
                if 7.5 < ASTM_hue < 15 or 60 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 10:
                if 65 < ASTM_hue < 77.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1
        elif value == 3:
            if chroma == 2:
                if 10 < ASTM_hue < 37.5 or 65 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 4:
                if 5 < ASTM_hue < 37.5 or 55 < ASTM_hue < 72.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6 or chroma == 8 or chroma == 10:
                if 7.5 < ASTM_hue < 37.5 or 57.5 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 12:
                if 7.5 < ASTM_hue < 42.5 or 57.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1
        elif value == 4:
            if chroma == 2 or chroma == 4:
                if 7.5 < ASTM_hue < 42.5 or 57.5 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6 or chroma == 8:
                if 7.5 < ASTM_hue < 40 or 57.5 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 10:
                if 7.5 < ASTM_hue < 40 or 57.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1
        elif value == 5:
            if chroma == 2:
                if 5 < ASTM_hue < 37.5 or 55 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 4 or chroma == 6 or chroma == 8:
                if 2.5 < ASTM_hue < 42.5 or 55 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 10:
                if 2.5 < ASTM_hue < 42.5 or 55 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1
        elif value == 6:
            if chroma == 2 or chroma == 4:
                if 5 < ASTM_hue < 37.5 or 55 < ASTM_hue < 87.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6:
                if 5 < ASTM_hue < 42.5 or 57.5 < ASTM_hue < 87.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 8 or chroma == 10:
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 12 or chroma == 14:
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 16:
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1
        elif value == 7:
            if chroma == 2 or chroma == 4 or chroma == 6:
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 8:
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 10:
                if 30 < ASTM_hue < 42.5 or 5 < ASTM_hue < 25 or 60 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 12:
                if 30 < ASTM_hue < 42.5 or 7.5 < ASTM_hue < 27.5 or 80 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 14:
                if 32.5 < ASTM_hue < 40 or 7.5 < ASTM_hue < 15 or 80 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1
        elif value == 8:
            if chroma == 2 or chroma == 4 or chroma == 6 or chroma == 8 or chroma == 10 or chroma == 12:
                if 5 < ASTM_hue < 40 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 14:
                if 32.5 < ASTM_hue < 40 or 5 < ASTM_hue < 15 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1
        elif value == 9:
            if chroma == 2 or chroma == 4:
                if 5 < ASTM_hue < 40 or 55 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6 or chroma == 8 or chroma == 10 or chroma == 12 or chroma == 14:
                if 5 < ASTM_hue < 42.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 16:
                if 35 < ASTM_hue < 42.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:
                interpolation_method = 1

    return interpolation_methods.get(interpolation_method)


def get_xy_from_renotation_ovoid(specification):
    """
    Converts given *Munsell* specification to *xy* chromaticity coordinates on *Munsell Renotation Sytem* ovoid.
    The *xy* point will be on the ovoid about the achromatic point, corresponding to the *Munsell* specification
    value and chroma.

    References:

    -  *The Munsell and Kubelka-Munk Toolbox*: \
    *MunsellAndKubelkaMunkToolboxApr2014/MunsellRenotationRoutines/FindHueOnRenotationOvoid.m*

    Usage::

        >>> get_xy_from_renotation_ovoid((2.5, 5.0, 12.0, 4))
        (0.4333, 0.5602)
        >>> get_xy_from_renotation_ovoid(8)
        (0.31006, 0.31616)

    :param specification: *Munsell* specification.
    :type specification: specification
    :return: *xy* chromaticity coordinates.
    :rtype: tuple

    :note: *Munsell* specification value must be an even integer in domain [1, 9].
    :note: *Munsell* specification chroma must be an even integer and a multiple of 2 in domain [2, 50].
    """

    if is_grey_munsell_colour(specification):
        return MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES
    else:
        hue, value, chroma, code = specification

        assert 1 <= value <= 9, "'{0}' specification value must be in domain [1, 9]!".format(specification)
        assert abs(value - round(value)) <= EVEN_INTEGER_THRESHOLD, \
            "'{0}' specification value must be an even integer!".format(specification)

        value = round(value)

        assert 2 <= chroma <= 50, "'{0}' specification chroma must be in domain [2, 50]!".format(specification)
        assert abs(2 * (chroma / 2 - round(chroma / 2))) <= EVEN_INTEGER_THRESHOLD, \
            "'{0}' specification chroma must be an even integer and multiple of 2!".format(specification)

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

        x_grey, y_grey = MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES

        specification_minus = (hue_minus, value, chroma, code_minus)
        x_minus, y_minus, Y_minus = get_xyY_from_renotation(specification_minus)
        z_minus, theta_minus, rho_minus = colour.algebra.coordinates.transformations.cartesian_to_cylindrical(
            (x_minus - x_grey, y_minus - y_grey, Y_minus))
        theta_minus = math.degrees(theta_minus)

        specification_plus = (hue_plus, value, chroma, code_plus)
        x_plus, y_plus, Y_plus = get_xyY_from_renotation(specification_plus)
        z_plus, theta_plus, rho_plus = colour.algebra.coordinates.transformations.cartesian_to_cylindrical(
            (x_plus - x_grey, y_plus - y_grey, Y_plus))
        theta_plus = math.degrees(theta_plus)

        lower_hue_angle = hue_to_hue_angle(hue_minus, code_minus)
        hue_angle = hue_to_hue_angle(hue, code)
        upper_hue_angle = hue_to_hue_angle(hue_plus, code_plus)

        if theta_minus - theta_plus > 180:
            theta_plus += +360

        if lower_hue_angle == 0:
            lower_hue_angle = 360

        if lower_hue_angle > upper_hue_angle:
            if lower_hue_angle > hue_angle:
                lower_hue_angle -= 360
            else:
                lower_hue_angle -= 360
                hue_angle -= 360

        interpolation_method = get_interpolation_method_from_renotation_ovoid(specification)

        if interpolation_method == "Linear":
            x = float(LinearInterpolator(numpy.array([lower_hue_angle, upper_hue_angle]),
                                         numpy.array([x_minus, x_plus]))(hue_angle))
            y = float(LinearInterpolator(numpy.array([lower_hue_angle, upper_hue_angle]),
                                         numpy.array([y_minus, y_plus]))(hue_angle))
        elif interpolation_method == "Radial":
            theta = float(LinearInterpolator(numpy.array([lower_hue_angle, upper_hue_angle]),
                                             numpy.array([theta_minus, theta_plus]))(hue_angle))
            rho = float(LinearInterpolator(numpy.array([lower_hue_angle, upper_hue_angle]),
                                           numpy.array([rho_minus, rho_plus]))(hue_angle))

            x = rho * math.cos(math.radians(theta)) + x_grey
            y = rho * math.sin(math.radians(theta)) + y_grey
        else:
            raise colour.utilities.exceptions.ProgrammingError(
                "Invalid interpolation method: '{0}'".format(interpolation_method))

        return x, y


def munsell_specification_to_xy(specification):
    """
    Converts given *Munsell* specification to *xy* chromaticity coordinates by interpolating over
    *Munsell Renotation Sytem* data.

    References:

    -  *The Munsell and Kubelka-Munk Toolbox*: \
    *MunsellAndKubelkaMunkToolboxApr2014/MunsellRenotationRoutines/MunsellToxyForIntegerMunsellValue.m*

    Usage::

        >>> munsell_specification_to_xy((2.1, 8.0, 17.9, 4))
        (0.4400632, 0.5522428)
        >>> munsell_specification_to_xy(8)
        (0.31006, 0.31616)

    :param specification: *Munsell* specification.
    :type specification: specification
    :return: *xy* chromaticity coordinates.
    :rtype: tuple

    :note: *Munsell* specification value must be an even integer in domain [0, 10].
    """

    if is_grey_munsell_colour(specification):
        return MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES
    else:
        hue, value, chroma, code = specification

        assert 0 <= value <= 10, "'{0}' specification value must be in domain [0, 10]!".format(specification)
        assert abs(value - round(value)) <= EVEN_INTEGER_THRESHOLD, \
            "'{0}' specification value must be an even integer!".format(specification)

        value = round(value)

        if chroma % 2 == 0:
            chroma_minus = chroma_plus = chroma
        else:
            chroma_minus = 2 * math.floor(chroma / 2)
            chroma_plus = chroma_minus + 2

        if chroma_minus == 0:
            # Smallest chroma ovoid collapses to illuminant chromaticity coordinates.
            x_minus, y_minus = MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES
        else:
            x_minus, y_minus = get_xy_from_renotation_ovoid((hue, value, chroma_minus, code))

        x_plus, y_plus = get_xy_from_renotation_ovoid((hue, value, chroma_plus, code))

        if chroma_minus == chroma_plus:
            x = x_minus
            y = y_minus
        else:
            x = float(LinearInterpolator(numpy.array([chroma_minus, chroma_plus]),
                                         numpy.array([x_minus, x_plus]))(chroma))
            y = float(LinearInterpolator(numpy.array([chroma_minus, chroma_plus]),
                                         numpy.array([y_minus, y_plus]))(chroma))

        return x, y


def munsell_colour_to_xyY(munsell_colour):
    """
    Converts given *Munsell* colour to *CIE xyY* colourspace.

    References:

    -  *The Munsell and Kubelka-Munk Toolbox*: \
    *MunsellAndKubelkaMunkToolboxApr2014/MunsellRenotationRoutines/MunsellToxyY.m*

    Usage::

        >>> munsell_colour_to_xyY("4.2YR 8.1/5.3")
        [[  0.38736945]
         [  0.35751656]
         [ 59.36200043]]
        >>> munsell_colour_to_xyY("N8.9")
        [[  0.31006   ]
         [  0.31616   ]
         [ 74.61344983]]

    :param munsell_colour: *Munsell* colour.
    :type munsell_colour: unicode
    :return: *CIE xyY* matrix.
    :rtype: matrix (3x1)

    :note: *Munsell* specification hue must be in domain [0, 10].
    :note: *Munsell* specification value must be in domain [0, 10].
    """

    specification = munsell_colour_to_munsell_specification(munsell_colour)

    if is_grey_munsell_colour(specification):
        value = specification
    else:
        hue, value, chroma, code = specification

        assert 0 <= hue <= 10, "'{0}' specification hue must be in domain [0, 10]!".format(specification)
        assert 0 <= value <= 10, "'{0}' specification value must be in domain [0, 10]!".format(specification)

    Y = colour.computation.luminance.luminance_ASTM_D1535_08(value)

    if abs(value - round(value)) < EVEN_INTEGER_THRESHOLD:
        value_minus = value_plus = round(value)
    else:
        value_minus = math.floor(value)
        value_plus = value_minus + 1

    minus_specification = value_minus if is_grey_munsell_colour(specification) else (hue, value_minus, chroma, code)
    x_minus, y_minus = munsell_specification_to_xy(minus_specification)

    plus_specification = value_plus if is_grey_munsell_colour(specification) or \
                                       value_plus == 10 else (hue, value_plus, chroma, code)
    x_plus, y_plus = munsell_specification_to_xy(plus_specification)

    if value_minus == value_plus:
        x = x_minus
        y = y_minus
    else:
        Y_minus = colour.computation.luminance.luminance_ASTM_D1535_08(value_minus)
        Y_plus = colour.computation.luminance.luminance_ASTM_D1535_08(value_plus)
        # TODO: Handle the float interp thing.
        x = float(LinearInterpolator(numpy.array([Y_minus, Y_plus]),
                                     numpy.array([x_minus, x_plus]))(Y))
        # TODO: Handle the float interp thing.
        y = float(LinearInterpolator(numpy.array([Y_minus, Y_plus]),
                                     numpy.array([y_minus, y_plus]))(Y))

    return numpy.matrix([x, y, Y]).reshape((3, 1))


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

