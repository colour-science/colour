#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**dci_p3.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *DCI-P3* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.exceptions
import color.illuminants
import color.verbose
from color.colorspaces.colorspace import Colorspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
		   "DCI_P3_PRIMARIES",
		   "DCI_P3_WHITEPOINT",
		   "DCI_P3_TO_XYZ_MATRIX",
		   "XYZ_TO_DCI_P3_MATRIX",
		   "DCI_P3_TRANSFER_FUNCTION",
		   "DCI_P3_INVERSE_TRANSFER_FUNCTION",
		   "DCI_P3_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://www.hp.com/united-states/campaigns/workstations/pdfs/lp2480zx-dci--p3-emulation.pdf
DCI_P3_PRIMARIES = numpy.matrix([0.680, 0.320,
								 0.265, 0.690,
								 0.150, 0.060]).reshape((3, 2))

DCI_P3_WHITEPOINT = (0.314, 0.351)

DCI_P3_TO_XYZ_MATRIX = numpy.matrix([0.44516982, 0.27713441, 0.17228267,
									 0.20949168, 0.72159525, 0.06891307,
									 0., 0.04706056, 0.90735539]).reshape((3, 3))

XYZ_TO_DCI_P3_MATRIX = DCI_P3_TO_XYZ_MATRIX.getI()

DCI_P3_TRANSFER_FUNCTION = lambda x: x

DCI_P3_INVERSE_TRANSFER_FUNCTION = lambda x: x

DCI_P3_COLORSPACE = Colorspace("DCI-P3",
							   DCI_P3_PRIMARIES,
							   DCI_P3_WHITEPOINT,
							   DCI_P3_TO_XYZ_MATRIX,
							   XYZ_TO_DCI_P3_MATRIX,
							   DCI_P3_TRANSFER_FUNCTION,
							   DCI_P3_INVERSE_TRANSFER_FUNCTION)
