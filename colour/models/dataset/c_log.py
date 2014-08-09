#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**c_log.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *C-Log* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import math
import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, get_normalised_primary_matrix

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["C_LOG_PRIMARIES",
           "C_LOG_WHITEPOINT",
           "C_LOG_TO_XYZ_MATRIX",
           "XYZ_TO_C_LOG_MATRIX",
           "C_LOG_TRANSFER_FUNCTION",
           "C_LOG_INVERSE_TRANSFER_FUNCTION",
           "C_LOG_COLOURSPACE"]

# http://downloads.canon.com/CDLC/Canon-Log_Transfer_Characteristic_6-20-2012.pdf
# Assuming *sRGB* / *Rec. 709* primaries.
C_LOG_PRIMARIES = np.array(
    [0.6400, 0.3300,
     0.3000, 0.6000,
     0.1500, 0.0600]).reshape((3, 2))

C_LOG_WHITEPOINT = ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D65")

C_LOG_TO_XYZ_MATRIX = get_normalised_primary_matrix(C_LOG_PRIMARIES,
                                                    C_LOG_WHITEPOINT)

XYZ_TO_C_LOG_MATRIX = np.linalg.inv(C_LOG_TO_XYZ_MATRIX)

C_LOG_TRANSFER_FUNCTION = lambda x: \
    0.529136 * math.log10(10.1596 * x + 1) + 0.0730597

C_LOG_INVERSE_TRANSFER_FUNCTION = lambda x: \
    -0.0716226 * (1.37427 - math.exp(1) ** (4.35159 * x))

C_LOG_COLOURSPACE = RGB_Colourspace(
    "C-Log",
    C_LOG_PRIMARIES,
    C_LOG_WHITEPOINT,
    C_LOG_TO_XYZ_MATRIX,
    XYZ_TO_C_LOG_MATRIX,
    C_LOG_TRANSFER_FUNCTION,
    C_LOG_INVERSE_TRANSFER_FUNCTION)
