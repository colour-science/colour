#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**s_log.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *S-Log* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import math

import numpy

import color.derivation
import color.illuminants
import color.utilities.exceptions
import color.utilities.verbose
from color.colorspaces.colorspace import Colorspace


__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["S_LOG_PRIMARIES",
           "S_LOG_WHITEPOINT",
           "S_LOG_TO_XYZ_MATRIX",
           "XYZ_TO_S_LOG_MATRIX",
           "S_LOG_TRANSFER_FUNCTION",
           "S_LOG_INVERSE_TRANSFER_FUNCTION",
           "S_LOG_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/slog_manual.pdf
S_LOG_PRIMARIES = numpy.matrix([0.73, 0.28,
                                0.14, 0.855,
                                0.10, -0.05]).reshape((3, 2))

S_LOG_WHITEPOINT = color.illuminants.ILLUMINANTS.get("CIE 1931 2 Degree Standard Observer").get("D65")

S_LOG_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(S_LOG_PRIMARIES, S_LOG_WHITEPOINT)

XYZ_TO_S_LOG_MATRIX = S_LOG_TO_XYZ_MATRIX.getI()

S_LOG_TRANSFER_FUNCTION = lambda x: (0.432699 * math.log10(x + 0.037584) + 0.616596) + 0.03

S_LOG_INVERSE_TRANSFER_FUNCTION = lambda x: (math.pow(10., ((x - 0.616596 - 0.03) / 0.432699)) - 0.037584)

S_LOG_COLORSPACE = Colorspace("S-Log",
                              S_LOG_PRIMARIES,
                              S_LOG_WHITEPOINT,
                              S_LOG_TO_XYZ_MATRIX,
                              XYZ_TO_S_LOG_MATRIX,
                              S_LOG_TRANSFER_FUNCTION,
                              S_LOG_INVERSE_TRANSFER_FUNCTION)
