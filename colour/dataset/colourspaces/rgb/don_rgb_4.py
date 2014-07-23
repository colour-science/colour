# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**don_rgb_4.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Don RGB 4* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.computation.colourspaces.rgb.derivation
import colour.dataset.illuminants.chromaticity_coordinates
from colour.computation.colourspaces.rgb.colourspace import Colourspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["DON_RGB_4_PRIMARIES",
           "DON_RGB_4_WHITEPOINT",
           "DON_RGB_4_TO_XYZ_MATRIX",
           "XYZ_TO_DON_RGB_4_MATRIX",
           "DON_RGB_4_TRANSFER_FUNCTION",
           "DON_RGB_4_INVERSE_TRANSFER_FUNCTION",
           "DON_RGB_4_COLOURSPACE"]


# http://www.hutchcolor.com/profiles/DonRGB4.zip
DON_RGB_4_PRIMARIES = numpy.array([0.69612068965517238, 0.29956896551724138,
                                   0.21468298109010012, 0.7652947719688542,
                                   0.12993762993762992, 0.035343035343035345]).reshape((3, 2))

DON_RGB_4_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

DON_RGB_4_TO_XYZ_MATRIX = colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix(
    DON_RGB_4_PRIMARIES,
    DON_RGB_4_WHITEPOINT)

XYZ_TO_DON_RGB_4_MATRIX = numpy.linalg.inv(DON_RGB_4_TO_XYZ_MATRIX)

DON_RGB_4_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

DON_RGB_4_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

DON_RGB_4_COLOURSPACE = Colourspace("Don RGB 4",
                                    DON_RGB_4_PRIMARIES,
                                    DON_RGB_4_WHITEPOINT,
                                    DON_RGB_4_TO_XYZ_MATRIX,
                                    XYZ_TO_DON_RGB_4_MATRIX,
                                    DON_RGB_4_TRANSFER_FUNCTION,
                                    DON_RGB_4_INVERSE_TRANSFER_FUNCTION)
