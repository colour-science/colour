# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**smptec_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *SMPTE-C RGB* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.dataset.illuminants.chromaticity_coordinates
import colour.computation.derivation
import colour.utilities.exceptions
import colour.utilities.verbose
from colour.computation.colourspace import Colourspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["SMPTE_C_RGB_PRIMARIES",
           "SMPTE_C_RGB_WHITEPOINT",
           "SMPTE_C_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_SMPTE_C_RGB_MATRIX",
           "SMPTE_C_RGB_TRANSFER_FUNCTION",
           "SMPTE_C_RGB_INVERSE_TRANSFER_FUNCTION",
           "SMPTE_C_RGB_COLOURSPACE"]

LOGGER = colour.utilities.verbose.install_logger()

# http://standards.smpte.org/content/978-1-61482-164-9/rp-145-2004/SEC1.body.pdf
SMPTE_C_RGB_PRIMARIES = numpy.matrix([0.630, 0.340,
                                      0.310, 0.595,
                                      0.155, 0.070]).reshape((3, 2))

SMPTE_C_RGB_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D65")

SMPTE_C_RGB_TO_XYZ_MATRIX = colour.computation.derivation.get_normalised_primary_matrix(
    SMPTE_C_RGB_PRIMARIES, SMPTE_C_RGB_WHITEPOINT)

XYZ_TO_SMPTE_C_RGB_MATRIX = SMPTE_C_RGB_TO_XYZ_MATRIX.getI()

SMPTE_C_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

SMPTE_C_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

SMPTE_C_RGB_COLOURSPACE = Colourspace("SMPTE-C RGB",
                                    SMPTE_C_RGB_PRIMARIES,
                                    SMPTE_C_RGB_WHITEPOINT,
                                    SMPTE_C_RGB_TO_XYZ_MATRIX,
                                    XYZ_TO_SMPTE_C_RGB_MATRIX,
                                    SMPTE_C_RGB_TRANSFER_FUNCTION,
                                    SMPTE_C_RGB_INVERSE_TRANSFER_FUNCTION)
