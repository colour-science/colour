#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**beta_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *Ekta Space PS 5* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.derivation
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
           "EKTA_SPACE_PS_5_PRIMARIES",
           "EKTA_SPACE_PS_5_WHITEPOINT",
           "EKTA_SPACE_PS_5_TO_XYZ_MATRIX",
           "XYZ_TO_EKTA_SPACE_PS_5_MATRIX",
           "EKTA_SPACE_PS_5_TRANSFER_FUNCTION",
           "EKTA_SPACE_PS_5_INVERSE_TRANSFER_FUNCTION",
           "EKTA_SPACE_PS_5_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://www.josephholmes.com/Ekta_Space.zip
EKTA_SPACE_PS_5_PRIMARIES = numpy.matrix([0.6947368421052631, 0.30526315789473685,
                                          0.26000000000000001, 0.69999999999999996,
                                          0.10972850678733032, 0.0045248868778280547]).reshape((3, 2))

EKTA_SPACE_PS_5_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

EKTA_SPACE_PS_5_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(EKTA_SPACE_PS_5_PRIMARIES,
                                                                               EKTA_SPACE_PS_5_WHITEPOINT)

XYZ_TO_EKTA_SPACE_PS_5_MATRIX = EKTA_SPACE_PS_5_TO_XYZ_MATRIX.getI()


def __ekta_space_ps5_transfer_function(RGB):
    """
    Defines the *Ekta Space PS 5* colorspace transfer function.

    :param RGB: RGB Matrix.
    :type RGB: Matrix (3x1)
    :return: Companded RGB Matrix.
    :rtype: Matrix (3x1)
    """

    RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
    return numpy.matrix(RGB).reshape((3, 1))


def __ekta_space_ps5_inverse_transfer_function(RGB):
    """
    Defines the *Ekta Space PS 5* colorspace inverse transfer function.

    :param RGB: RGB Matrix.
    :type RGB: Matrix (3x1)
    :return: Companded RGB Matrix.
    :rtype: Matrix (3x1)
    """

    RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
    return numpy.matrix(RGB).reshape((3, 1))


EKTA_SPACE_PS_5_TRANSFER_FUNCTION = __ekta_space_ps5_transfer_function

EKTA_SPACE_PS_5_INVERSE_TRANSFER_FUNCTION = __ekta_space_ps5_inverse_transfer_function

EKTA_SPACE_PS_5_COLORSPACE = Colorspace("Ekta Space PS 5",
                                        EKTA_SPACE_PS_5_PRIMARIES,
                                        EKTA_SPACE_PS_5_WHITEPOINT,
                                        EKTA_SPACE_PS_5_TO_XYZ_MATRIX,
                                        XYZ_TO_EKTA_SPACE_PS_5_MATRIX,
                                        EKTA_SPACE_PS_5_TRANSFER_FUNCTION,
                                        EKTA_SPACE_PS_5_INVERSE_TRANSFER_FUNCTION)
