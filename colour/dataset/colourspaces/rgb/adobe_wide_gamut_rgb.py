# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**adobe_wide_gamut_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Adobe Wide Gamut RGB* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.computation.colourspaces.rgb.derivation
import colour.dataset.illuminants.chromaticity_coordinates
import colour.utilities.exceptions
import colour.utilities.verbose
from colour.computation.colourspaces.rgb.colourspace import Colourspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["ADOBE_WIDE_GAMUT_RGB_PRIMARIES",
           "ADOBE_WIDE_GAMUT_RGB_WHITEPOINT",
           "ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX",
           "ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION",
           "ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION",
           "ADOBE_WIDE_GAMUT_RGB_COLOURSPACE"]

LOGGER = colour.utilities.verbose.install_logger()

# http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space
ADOBE_WIDE_GAMUT_RGB_PRIMARIES = numpy.matrix([0.7347, 0.2653,
                                               0.1152, 0.8264,
                                               0.1566, 0.0177]).reshape((3, 2))

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix(
    ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
    ADOBE_WIDE_GAMUT_RGB_WHITEPOINT)

XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX = ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX.getI()

ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / (563. / 256.))

ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** (563. / 256.)

ADOBE_WIDE_GAMUT_RGB_COLOURSPACE = Colourspace("Adobe Wide Gamut RGB",
                                             ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
                                             ADOBE_WIDE_GAMUT_RGB_WHITEPOINT,
                                             ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
                                             XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX,
                                             ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION,
                                             ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION)
