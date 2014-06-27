# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**pal_secam_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Pal/Secam RGB* colourspace.

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

__all__ = ["PAL_SECAM_RGB_PRIMARIES",
           "PAL_SECAM_RGB_WHITEPOINT",
           "PAL_SECAM_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_PAL_SECAM_RGB_MATRIX",
           "PAL_SECAM_RGB_TRANSFER_FUNCTION",
           "PAL_SECAM_RGB_INVERSE_TRANSFER_FUNCTION",
           "PAL_SECAM_RGB_COLOURSPACE"]

LOGGER = colour.utilities.verbose.install_logger()

# http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf
PAL_SECAM_RGB_PRIMARIES = numpy.matrix([0.64, 0.33,
                                        0.29, 0.60,
                                        0.15, 0.06]).reshape((3, 2))

PAL_SECAM_RGB_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D65")

PAL_SECAM_RGB_TO_XYZ_MATRIX = colour.computation.derivation.get_normalized_primary_matrix(PAL_SECAM_RGB_PRIMARIES,
                                                                                          PAL_SECAM_RGB_WHITEPOINT)

XYZ_TO_PAL_SECAM_RGB_MATRIX = PAL_SECAM_RGB_TO_XYZ_MATRIX.getI()

PAL_SECAM_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.8)

PAL_SECAM_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.8

PAL_SECAM_RGB_COLOURSPACE = Colourspace("Pal/Secam RGB",
                                      PAL_SECAM_RGB_PRIMARIES,
                                      PAL_SECAM_RGB_WHITEPOINT,
                                      PAL_SECAM_RGB_TO_XYZ_MATRIX,
                                      XYZ_TO_PAL_SECAM_RGB_MATRIX,
                                      PAL_SECAM_RGB_TRANSFER_FUNCTION,
                                      PAL_SECAM_RGB_INVERSE_TRANSFER_FUNCTION)
