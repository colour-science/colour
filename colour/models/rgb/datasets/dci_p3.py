# -*- coding: utf-8 -*-
"""
DCI-P3 & DCI-P3+ Colourspaces
=============================

Defines the *DCI-P3* and *DCI-P3+* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_DCI_P3`.
-   :attr:`colour.models.RGB_COLOURSPACE_DCI_P3_P`.

References
----------
-   :cite:`Canon2014a` : Canon. (2014). EOS C500 Firmware Update. Retrieved
    August 27, 2016, from
    https://www.usa.canon.com/internet/portal/us/home/explore/\
product-showcases/cameras-and-lenses/cinema-eos-firmware/c500
-   :cite:`DigitalCinemaInitiatives2007b` : Digital Cinema Initiatives. (2007).
    Digital Cinema System Specification - Version 1.1.
    http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
-   :cite:`Hewlett-PackardDevelopmentCompany2009a` : Hewlett-Packard
    Development Company. (2009). Understanding the HP DreamColor LP2480zx
    DCI-P3 Emulation Color Space (pp. 1-3).
    http://www.hp.com/united-states/campaigns/workstations/pdfs/\
lp2480zx-dci--p3-emulation.pdf
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
from functools import partial

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, gamma_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_DCI_P3', 'PRIMARIES_DCI_P3_P', 'WHITEPOINT_NAME_DCI_P3',
    'CCS_WHITEPOINT_DCI_P3', 'MATRIX_DCI_P3_TO_XYZ', 'MATRIX_XYZ_TO_DCI_P3',
    'MATRIX_DCI_P3_P_TO_XYZ', 'MATRIX_XYZ_TO_DCI_P3_P',
    'RGB_COLOURSPACE_DCI_P3', 'RGB_COLOURSPACE_DCI_P3_P'
]

PRIMARIES_DCI_P3 = np.array([
    [0.6800, 0.3200],
    [0.2650, 0.6900],
    [0.1500, 0.0600],
])
"""
*DCI-P3* colourspace primaries.

PRIMARIES_DCI_P3 : ndarray, (3, 2)
"""

PRIMARIES_DCI_P3_P = np.array([
    [0.7400, 0.2700],
    [0.2200, 0.7800],
    [0.0900, -0.0900],
])
"""
*DCI-P3+* colourspace primaries.

PRIMARIES_DCI_P3_P : ndarray, (3, 2)
"""

WHITEPOINT_NAME_DCI_P3 = 'DCI-P3'
"""
*DCI-P3* colourspace whitepoint name.

WHITEPOINT_NAME_DCI_P3 : unicode

Warning
-------
DCI-P3 illuminant has no associated spectral distribution. DCI has no
official reference spectral measurement for this whitepoint. The closest
matching spectral distribution is Kinoton 75P projector.
"""

CCS_WHITEPOINT_DCI_P3 = (CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
                         [WHITEPOINT_NAME_DCI_P3])
"""
*DCI-P3* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_DCI_P3 : ndarray
"""

MATRIX_DCI_P3_TO_XYZ = normalised_primary_matrix(PRIMARIES_DCI_P3,
                                                 CCS_WHITEPOINT_DCI_P3)
"""
*DCI-P3* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_DCI_P3_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_DCI_P3 = np.linalg.inv(MATRIX_DCI_P3_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *DCI-P3* colourspace matrix.

MATRIX_XYZ_TO_DCI_P3 : array_like, (3, 3)
"""

MATRIX_DCI_P3_P_TO_XYZ = normalised_primary_matrix(PRIMARIES_DCI_P3_P,
                                                   CCS_WHITEPOINT_DCI_P3)
"""
*DCI-P3+* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_DCI_P3_P_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_DCI_P3_P = np.linalg.inv(MATRIX_DCI_P3_P_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *DCI-P3+* colourspace matrix.

MATRIX_XYZ_TO_DCI_P3_P : array_like, (3, 3)
"""

RGB_COLOURSPACE_DCI_P3 = RGB_Colourspace(
    'DCI-P3',
    PRIMARIES_DCI_P3,
    CCS_WHITEPOINT_DCI_P3,
    WHITEPOINT_NAME_DCI_P3,
    MATRIX_DCI_P3_TO_XYZ,
    MATRIX_XYZ_TO_DCI_P3,
    partial(gamma_function, exponent=1 / 2.6),
    partial(gamma_function, exponent=2.6),
)
RGB_COLOURSPACE_DCI_P3.__doc__ = """
*DCI-P3* colourspace.

References
----------
:cite:`DigitalCinemaInitiatives2007b`,
:cite:`Hewlett-PackardDevelopmentCompany2009a`

RGB_COLOURSPACE_DCI_P3 : RGB_Colourspace
"""

RGB_COLOURSPACE_DCI_P3_P = RGB_Colourspace(
    'DCI-P3+',
    PRIMARIES_DCI_P3_P,
    CCS_WHITEPOINT_DCI_P3,
    WHITEPOINT_NAME_DCI_P3,
    MATRIX_DCI_P3_P_TO_XYZ,
    MATRIX_XYZ_TO_DCI_P3_P,
    partial(gamma_function, exponent=1 / 2.6),
    partial(gamma_function, exponent=2.6),
)
RGB_COLOURSPACE_DCI_P3_P.__doc__ = """
*DCI-P3+* colourspace.

References
----------
:cite:`Canon2014a`

RGB_COLOURSPACE_DCI_P3_P : RGB_Colourspace
"""
