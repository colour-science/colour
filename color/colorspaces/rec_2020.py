#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**rec_2020.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *Rec 2020* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.data_structures
import color.exceptions
import color.illuminants
import color.verbose
from color.colorspaces.colorspace import Colorspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "REC_2020_PRIMARIES",
           "REC_2020_WHITEPOINT",
           "REC_2020_TO_XYZ_MATRIX",
           "XYZ_TO_REC_2020_MATRIX",
           "REC_2020_CONSTANTS",
           "REC_2020_TRANSFER_FUNCTION",
           "REC_2020_INVERSE_TRANSFER_FUNCTION",
           "REC_2020_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-I!!PDF-E.pdf
REC_2020_PRIMARIES = numpy.matrix([0.708, 0.292,
                                   0.170, 0.797,
                                   0.131, 0.046]).reshape((3, 2))

REC_2020_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D65")

REC_2020_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(REC_2020_PRIMARIES, REC_2020_WHITEPOINT)

XYZ_TO_REC_2020_MATRIX = REC_2020_TO_XYZ_MATRIX.getI()

REC_2020_CONSTANTS = color.data_structures.Structure(alpha=lambda x: 1.099 if x else 1.0993,
                                                     beta=lambda x: 0.018 if x else 0.0181)


def __rec_2020_transfer_function(value, is_10_bits_system=True):
    """
    Defines the *Rec. 2020* colorspace transfer function.

    Reference: http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-I!!PDF-E.pdf: Signal Format

    :param value: value.
    :type value: float
    :param is_10_bits_system: *Rec. 709* *alpha* and *beta* constants are used if system is 10 bit.
    :type is_10_bits_system: bool
    :return: Companded value.
    :rtype: float
    """

    return value * 4.5 if value < REC_2020_CONSTANTS.beta(is_10_bits_system) else \
        REC_2020_CONSTANTS.alpha(is_10_bits_system) * (value ** 0.45) - (
            REC_2020_CONSTANTS.alpha(is_10_bits_system) - 1.)


def __rec_2020_inverse_transfer_function(value, is_10_bits_system=True):
    """
    Defines the *Rec. 2020* colorspace inverse transfer function.

    Reference: http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-I!!PDF-E.pdf: Signal Format

    :param value: value.
    :type value: float
    :param is_10_bits_system: *Rec. 709* *alpha* and *beta* constants are used if system is 10 bit.
    :type is_10_bits_system: bool
    :return: Companded value.
    :rtype: float
    """

    return value / 4.5 if value < REC_2020_CONSTANTS.beta(is_10_bits_system) else \
        ((value + (REC_2020_CONSTANTS.alpha(is_10_bits_system) - 1.)) / REC_2020_CONSTANTS.alpha(is_10_bits_system)) ** \
        (1. / 0.45)


REC_2020_TRANSFER_FUNCTION = __rec_2020_transfer_function

REC_2020_INVERSE_TRANSFER_FUNCTION = __rec_2020_inverse_transfer_function

REC_2020_COLORSPACE = Colorspace("Rec. 2020",
                                 REC_2020_PRIMARIES,
                                 REC_2020_WHITEPOINT,
                                 REC_2020_TO_XYZ_MATRIX,
                                 XYZ_TO_REC_2020_MATRIX,
                                 REC_2020_TRANSFER_FUNCTION,
                                 REC_2020_INVERSE_TRANSFER_FUNCTION)
