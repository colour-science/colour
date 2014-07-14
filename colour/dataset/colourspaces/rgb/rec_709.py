# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**rec_709.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Rec 709* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.dataset.illuminants
import colour.utilities.exceptions
import colour.utilities.verbose
from colour.computation.colourspaces.rgb.colourspace import Colourspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["REC_709_PRIMARIES",
           "REC_709_WHITEPOINT",
           "REC_709_TO_XYZ_MATRIX",
           "XYZ_TO_REC_709_MATRIX",
           "REC_709_TRANSFER_FUNCTION",
           "REC_709_INVERSE_TRANSFER_FUNCTION",
           "REC_709_COLOURSPACE"]

LOGGER = colour.utilities.verbose.install_logger()

# http://www.color.org/srgb.pdf
# http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf: 1 Opto-electronic conversion
REC_709_PRIMARIES = numpy.matrix([0.6400, 0.3300,
                                  0.3000, 0.6000,
                                  0.1500, 0.0600]).reshape((3, 2))

REC_709_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D65")

REC_709_TO_XYZ_MATRIX = numpy.matrix([0.41238656, 0.35759149, 0.18045049,
                                      0.21263682, 0.71518298, 0.0721802,
                                      0.01933062, 0.11919716, 0.95037259]).reshape((3, 3))

XYZ_TO_REC_709_MATRIX = REC_709_TO_XYZ_MATRIX.getI()

REC_709_TRANSFER_FUNCTION = lambda x: x * 4.5 if x < 0.018 else 1.099 * (x ** 0.45) - 0.099

REC_709_INVERSE_TRANSFER_FUNCTION = lambda x: x / 4.5 if x < 0.018 else ((x + 0.099) / 1.099) ** (1 / 0.45)

REC_709_COLOURSPACE = Colourspace("Rec. 709",
                                REC_709_PRIMARIES,
                                REC_709_WHITEPOINT,
                                REC_709_TO_XYZ_MATRIX,
                                XYZ_TO_REC_709_MATRIX,
                                REC_709_TRANSFER_FUNCTION,
                                REC_709_INVERSE_TRANSFER_FUNCTION)
