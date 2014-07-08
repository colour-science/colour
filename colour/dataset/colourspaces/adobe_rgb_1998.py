# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**adobe_rgb_1998.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Adobe RGB 1998* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.dataset.illuminants
import colour.utilities.exceptions
import colour.utilities.verbose
from colour.computation.colourspace import Colourspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["ADOBE_RGB_1998_PRIMARIES",
           "ADOBE_RGB_1998_WHITEPOINT",
           "ADOBE_RGB_1998_TO_XYZ_MATRIX",
           "XYZ_TO_ADOBE_RGB_1998_MATRIX",
           "ADOBE_RGB_1998_TRANSFER_FUNCTION",
           "ADOBE_RGB_1998_INVERSE_TRANSFER_FUNCTION",
           "ADOBE_RGB_1998_COLOURSPACE"]

LOGGER = colour.utilities.verbose.install_logger()

# http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
ADOBE_RGB_1998_PRIMARIES = numpy.matrix([0.6400, 0.3300,
                                         0.2100, 0.7100,
                                         0.1500, 0.0600]).reshape((3, 2))

ADOBE_RGB_1998_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D65")

# http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf: 4.3.5.3 Converting RGB to normalised XYZ values
ADOBE_RGB_1998_TO_XYZ_MATRIX = numpy.matrix([0.57666809, 0.18556195, 0.1881985,
                                             0.29734449, 0.62737611, 0.0752794,
                                             0.02703132, 0.07069027, 0.99117879]).reshape((3, 3))

XYZ_TO_ADOBE_RGB_1998_MATRIX = ADOBE_RGB_1998_TO_XYZ_MATRIX.getI()

ADOBE_RGB_1998_TRANSFER_FUNCTION = lambda x: x ** (1 / (563. / 256.))

ADOBE_RGB_1998_INVERSE_TRANSFER_FUNCTION = lambda x: x ** (563. / 256.)

ADOBE_RGB_1998_COLOURSPACE = Colourspace("Adobe RGB 1998",
                                       ADOBE_RGB_1998_PRIMARIES,
                                       ADOBE_RGB_1998_WHITEPOINT,
                                       ADOBE_RGB_1998_TO_XYZ_MATRIX,
                                       XYZ_TO_ADOBE_RGB_1998_MATRIX,
                                       ADOBE_RGB_1998_TRANSFER_FUNCTION,
                                       ADOBE_RGB_1998_INVERSE_TRANSFER_FUNCTION)
