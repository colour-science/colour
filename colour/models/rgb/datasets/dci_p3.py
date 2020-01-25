# -*- coding: utf-8 -*-
"""
DCI-P3 & DCI-P3+ Colourspaces
=============================

Defines the *DCI-P3* and *DCI-P3+* colourspaces:

-   :attr:`colour.models.DCI_P3_COLOURSPACE`.
-   :attr:`colour.models.DCI_P3_P_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Canon2014a` : Canon. (2014). EOS C500 Firmware Update. Retrieved
    August 27, 2016, from https://www.usa.canon.com/internet/portal/us/home/\
explore/product-showcases/cameras-and-lenses/cinema-eos-firmware/c500
-   :cite:`DigitalCinemaInitiatives2007b` : Digital Cinema Initiatives. (2007).
    Digital Cinema System Specification - Version 1.1. Retrieved from
    http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
-   :cite:`Hewlett-PackardDevelopmentCompany2009a` : Hewlett-Packard
    Development Company. (2009). Understanding the HP DreamColor LP2480zx
    DCI-P3 Emulation Color Space. Retrieved from http://www.hp.com/\
united-states/campaigns/workstations/pdfs/lp2480zx-dci--p3-emulation.pdf
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, gamma_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'DCI_P3_PRIMARIES', 'DCI_P3_P_PRIMARIES', 'DCI_P3_WHITEPOINT_NAME',
    'DCI_P3_WHITEPOINT', 'DCI_P3_TO_XYZ_MATRIX', 'XYZ_TO_DCI_P3_MATRIX',
    'DCI_P3_P_TO_XYZ_MATRIX', 'XYZ_TO_DCI_P3_P_MATRIX', 'DCI_P3_COLOURSPACE',
    'DCI_P3_P_COLOURSPACE'
]

DCI_P3_PRIMARIES = np.array([
    [0.6800, 0.3200],
    [0.2650, 0.6900],
    [0.1500, 0.0600],
])
"""
*DCI-P3* colourspace primaries.

DCI_P3_PRIMARIES : ndarray, (3, 2)
"""

DCI_P3_P_PRIMARIES = np.array([
    [0.7400, 0.2700],
    [0.2200, 0.7800],
    [0.0900, -0.0900],
])
"""
*DCI-P3+* colourspace primaries.

DCI_P3_P_PRIMARIES : ndarray, (3, 2)
"""

DCI_P3_WHITEPOINT_NAME = 'DCI-P3'
"""
*DCI-P3* colourspace whitepoint name.

DCI_P3_WHITEPOINT_NAME : unicode

Warning
-------
DCI-P3 illuminant has no associated spectral distribution. DCI has no
official reference spectral measurement for this whitepoint. The closest
matching spectral distribution is Kinoton 75P projector.
"""

DCI_P3_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][DCI_P3_WHITEPOINT_NAME])
"""
*DCI-P3* colourspace whitepoint.

DCI_P3_WHITEPOINT : ndarray
"""

DCI_P3_TO_XYZ_MATRIX = normalised_primary_matrix(DCI_P3_PRIMARIES,
                                                 DCI_P3_WHITEPOINT)
"""
*DCI-P3* colourspace to *CIE XYZ* tristimulus values matrix.

DCI_P3_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DCI_P3_MATRIX = np.linalg.inv(DCI_P3_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *DCI-P3* colourspace matrix.

XYZ_TO_DCI_P3_MATRIX : array_like, (3, 3)
"""

DCI_P3_P_TO_XYZ_MATRIX = normalised_primary_matrix(DCI_P3_P_PRIMARIES,
                                                   DCI_P3_WHITEPOINT)
"""
*DCI-P3+* colourspace to *CIE XYZ* tristimulus values matrix.

DCI_P3_P_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DCI_P3_P_MATRIX = np.linalg.inv(DCI_P3_P_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *DCI-P3+* colourspace matrix.

XYZ_TO_DCI_P3_P_MATRIX : array_like, (3, 3)
"""

DCI_P3_COLOURSPACE = RGB_Colourspace(
    'DCI-P3',
    DCI_P3_PRIMARIES,
    DCI_P3_WHITEPOINT,
    DCI_P3_WHITEPOINT_NAME,
    DCI_P3_TO_XYZ_MATRIX,
    XYZ_TO_DCI_P3_MATRIX,
    partial(gamma_function, exponent=1 / 2.6),
    partial(gamma_function, exponent=2.6),
)
DCI_P3_COLOURSPACE.__doc__ = """
*DCI-P3* colourspace.

References
----------
:cite:`DigitalCinemaInitiatives2007b`,
:cite:`Hewlett-PackardDevelopmentCompany2009a`

DCI_P3_COLOURSPACE : RGB_Colourspace
"""

DCI_P3_P_COLOURSPACE = RGB_Colourspace(
    'DCI-P3+',
    DCI_P3_P_PRIMARIES,
    DCI_P3_WHITEPOINT,
    DCI_P3_WHITEPOINT_NAME,
    DCI_P3_P_TO_XYZ_MATRIX,
    XYZ_TO_DCI_P3_P_MATRIX,
    partial(gamma_function, exponent=1 / 2.6),
    partial(gamma_function, exponent=2.6),
)
DCI_P3_P_COLOURSPACE.__doc__ = """
*DCI-P3+* colourspace.

References
----------
:cite:`Canon2014a`

DCI_P3_P_COLOURSPACE : RGB_Colourspace
"""
