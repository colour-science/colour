# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**illuminants.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *illuminants* data.

**Others:**

"""

from __future__ import unicode_literals

import color.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["ILLUMINANTS_STANDARD_CIE_1931_2_DEGREE_OBSERVER_CHROMATICITY_COORDINATES",
           "ILLUMINANTS_STANDARD_CIE_1964_10_DEGREE_OBSERVER_CHROMATICITY_COORDINATES",
           "ILLUMINANTS"]

LOGGER = color.utilities.verbose.install_logger()

# http://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants
ILLUMINANTS_STANDARD_CIE_1931_2_DEGREE_OBSERVER_CHROMATICITY_COORDINATES = {
    "A": (0.44757, 0.40745),
    "B": (0.34842, 0.35161),
    "C": (0.31006, 0.31616),
    "D50": (0.34567, 0.35850),
    "D55": (0.33242, 0.34743),
    "D60": (0.32168, 0.33767),
    "D65": (0.31271, 0.32902),
    "D75": (0.29902, 0.31485),
    "E": (1. / 3., 1. / 3.),
    "F1": (0.31310, 0.33727),
    "F2": (0.37208, 0.37529),
    "F3": (0.40910, 0.39430),
    "F4": (0.44018, 0.40329),
    "F5": (0.31379, 0.34531),
    "F6": (0.37790, 0.38835),
    "F7": (0.31292, 0.32933),
    "F8": (0.34588, 0.35875),
    "F9": (0.37417, 0.37281),
    "F10": (0.34609, 0.35986),
    "F12": (0.43695, 0.40441)}

ILLUMINANTS_STANDARD_CIE_1964_10_DEGREE_OBSERVER_CHROMATICITY_COORDINATES = {
    "A": (0.45117, 0.40594),
    "B": (0.34980, 0.35270),
    "C": (0.31039, 0.31905),
    "D50": (0.34773, 0.35952),
    "D55": (0.33411, 0.34877),
    "D65": (0.31382, 0.33100),
    "D75": (0.29968, 0.31740),
    "E": (1. / 3., 1. / 3.),
    "F1": (0.31811, 0.33559),
    "F2": (0.37925, 0.36733),
    "F3": (0.41761, 0.38324),
    "F4": (0.44920, 0.39074),
    "F5": (0.31975, 0.34246),
    "F6": (0.38660, 0.37847),
    "F7": (0.31569, 0.32960),
    "F8": (0.34902, 0.35939),
    "F9": (0.37829, 0.37045),
    "F10": (0.35090, 0.35444),
    "F11": (0.38541, 0.37123),
    "F12": (0.44256, 0.39717)}

ILLUMINANTS = {
    "CIE 1931 2 Degree Standard Observer": ILLUMINANTS_STANDARD_CIE_1931_2_DEGREE_OBSERVER_CHROMATICITY_COORDINATES,
    "CIE 1964 10 Degree Standard Observer": ILLUMINANTS_STANDARD_CIE_1964_10_DEGREE_OBSERVER_CHROMATICITY_COORDINATES}

# Add calculated *CIE D60 Illuminant* *xy* chromaticity coordinates for *CIE 1964 10 Degree Standard Observer*.
# cmfs=color.STANDARD_OBSERVERS_CMFS.get("CIE 1964 10 Degree Standard Observer")
# spd=color.ILLUMINANTS_RELATIVE_SPDS.get("D60")
# xy = color.XYZ_to_xy(color.spectrum.transformations.spectral_to_XYZ(spd, cmfs))
ILLUMINANTS_STANDARD_CIE_1964_10_DEGREE_OBSERVER_CHROMATICITY_COORDINATES["D60"] = (0.32299152277736748,
                                                                                    0.33912831290965012)
