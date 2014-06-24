# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**eci_rgb_v2.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *ECI RGB v2* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.computation.derivation
import color.dataset.illuminants.chromaticity_coordinates
import color.computation.lightness
import color.utilities.exceptions
import color.utilities.verbose
from color.computation.colorspace import Colorspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["ECI_RGB_V2_PRIMARIES",
           "ECI_RGB_V2_WHITEPOINT",
           "ECI_RGB_V2_TO_XYZ_MATRIX",
           "XYZ_TO_ECI_RGB_V2_MATRIX",
           "ECI_RGB_V2_TRANSFER_FUNCTION",
           "ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION",
           "ECI_RGB_V2_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://www.eci.org/_media/downloads/icc_profiles_from_eci/ecirgbv20.zip
ECI_RGB_V2_PRIMARIES = numpy.matrix([0.67010309278350522, 0.32989690721649484,
                                     0.20990566037735847, 0.70990566037735836,
                                     0.14006179196704427, 0.080329557157569509]).reshape((3, 2))

ECI_RGB_V2_WHITEPOINT = color.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

ECI_RGB_V2_TO_XYZ_MATRIX = color.computation.derivation.get_normalized_primary_matrix(ECI_RGB_V2_PRIMARIES,
                                                                                       ECI_RGB_V2_WHITEPOINT)

XYZ_TO_ECI_RGB_V2_MATRIX = ECI_RGB_V2_TO_XYZ_MATRIX.getI()

ECI_RGB_V2_TRANSFER_FUNCTION = lambda x: color.computation.lightness.lightness_1976(x * 100.) / 100.

ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION = lambda x: color.computation.lightness.luminance_1976(x * 100.) / 100.

ECI_RGB_V2_COLORSPACE = Colorspace("ECI RGB v2",
                                   ECI_RGB_V2_PRIMARIES,
                                   ECI_RGB_V2_WHITEPOINT,
                                   ECI_RGB_V2_TO_XYZ_MATRIX,
                                   XYZ_TO_ECI_RGB_V2_MATRIX,
                                   ECI_RGB_V2_TRANSFER_FUNCTION,
                                   ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION)
