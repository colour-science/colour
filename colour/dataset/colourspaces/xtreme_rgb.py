# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**xtreme_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Xtreme RGB* colourspace.

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

__all__ = ["XTREME_RGB_PRIMARIES",
           "XTREME_RGB_WHITEPOINT",
           "XTREME_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_XTREME_RGB_MATRIX",
           "XTREME_RGB_TRANSFER_FUNCTION",
           "XTREME_RGB_INVERSE_TRANSFER_FUNCTION",
           "XTREME_RGB_COLOURSPACE"]

LOGGER = colour.utilities.verbose.install_logger()

# http://www.hutchcolor.com/profiles/XtremeRGB.zip
XTREME_RGB_PRIMARIES = numpy.matrix([1., 0.,
                                     0., 1.,
                                     0., 0.]).reshape((3, 2))

XTREME_RGB_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

XTREME_RGB_TO_XYZ_MATRIX = colour.computation.derivation.get_normalised_primary_matrix(XTREME_RGB_PRIMARIES,
                                                                                       XTREME_RGB_WHITEPOINT)

XYZ_TO_XTREME_RGB_MATRIX = XTREME_RGB_TO_XYZ_MATRIX.getI()

XTREME_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

XTREME_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

XTREME_RGB_COLOURSPACE = Colourspace("Xtreme RGB",
                                   XTREME_RGB_PRIMARIES,
                                   XTREME_RGB_WHITEPOINT,
                                   XTREME_RGB_TO_XYZ_MATRIX,
                                   XYZ_TO_XTREME_RGB_MATRIX,
                                   XTREME_RGB_TRANSFER_FUNCTION,
                                   XTREME_RGB_INVERSE_TRANSFER_FUNCTION)
