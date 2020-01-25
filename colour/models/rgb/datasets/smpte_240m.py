# -*- coding: utf-8 -*-
"""
SMPTE 240M Colourspace
======================

Defines the *SMPTE 240M* colourspace:

-   :attr:`colour.models.SMPTE_240M_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`SocietyofMotionPictureandTelevisionEngineers1999b` : Society of
    Motion Picture and Television Engineers. (1999). ANSI/SMPTE 240M-1995 -
    Signal Parameters - 1125-Line High-Definition Production Systems. Retrieved
    from http://car.france3.mars.free.fr/HD/INA- 26 jan 06/\
SMPTE normes et confs/s240m.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, normalised_primary_matrix,
                               oetf_SMPTE240M, eotf_SMPTE240M)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SMPTE_240M_PRIMARIES', 'SMPTE_240M_WHITEPOINT_NAME',
    'SMPTE_240M_WHITEPOINT', 'SMPTE_240M_TO_XYZ_MATRIX',
    'XYZ_TO_SMPTE_240M_MATRIX', 'SMPTE_240M_COLOURSPACE'
]

SMPTE_240M_PRIMARIES = np.array([
    [0.6300, 0.3400],
    [0.3100, 0.5950],
    [0.1550, 0.0700],
])
"""
*SMPTE 240M* colourspace primaries.

SMPTE_240M_PRIMARIES : ndarray, (3, 2)
"""

SMPTE_240M_WHITEPOINT_NAME = 'D65'
"""
*SMPTE 240M* colourspace whitepoint name.

SMPTE_240M_WHITEPOINT_NAME : unicode
"""

SMPTE_240M_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    SMPTE_240M_WHITEPOINT_NAME])
"""
*SMPTE 240M* colourspace whitepoint.

SMPTE_240M_WHITEPOINT : ndarray
"""

SMPTE_240M_TO_XYZ_MATRIX = normalised_primary_matrix(SMPTE_240M_PRIMARIES,
                                                     SMPTE_240M_WHITEPOINT)
"""
*SMPTE 240M* colourspace to *CIE XYZ* tristimulus values matrix.

SMPTE_240M_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_SMPTE_240M_MATRIX = np.linalg.inv(SMPTE_240M_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *SMPTE 240M* colourspace matrix.

XYZ_TO_SMPTE_240M_MATRIX : array_like, (3, 3)
"""

SMPTE_240M_COLOURSPACE = RGB_Colourspace(
    'SMPTE 240M',
    SMPTE_240M_PRIMARIES,
    SMPTE_240M_WHITEPOINT,
    SMPTE_240M_WHITEPOINT_NAME,
    SMPTE_240M_TO_XYZ_MATRIX,
    XYZ_TO_SMPTE_240M_MATRIX,
    oetf_SMPTE240M,
    eotf_SMPTE240M,
)
SMPTE_240M_COLOURSPACE.__doc__ = """
*SMPTE 240M* colourspace.

References
----------
:cite:`SocietyofMotionPictureandTelevisionEngineers1999b`,

SMPTE_240M_COLOURSPACE : RGB_Colourspace
"""
