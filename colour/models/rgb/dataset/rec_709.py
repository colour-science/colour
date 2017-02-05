#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rec. 709 Colourspace
====================

Defines the *Rec. 709* colourspace:

-   :attr:`REC_709_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (2015). Recommendation
        ITU-R BT.709-6 - Parameter values for the HDTV standards for
        production and international programme exchange BT Series Broadcasting
        service (Vol. 5). Retrieved from https://www.itu.int/dms_pubrec/\
itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    oetf_BT709,
    eotf_BT709,
    normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['REC_709_PRIMARIES',
           'REC_709_WHITEPOINT',
           'REC_709_ILLUMINANT',
           'REC_709_TO_XYZ_MATRIX',
           'XYZ_TO_REC_709_MATRIX',
           'REC_709_COLOURSPACE']

REC_709_PRIMARIES = np.array(
    [[0.6400, 0.3300],
     [0.3000, 0.6000],
     [0.1500, 0.0600]])
"""
*Rec. 709* colourspace primaries.

REC_709_PRIMARIES : ndarray, (3, 2)
"""

REC_709_ILLUMINANT = 'D65'
"""
*Rec. 709* colourspace whitepoint name as illuminant.

REC_709_ILLUMINANT : unicode
"""

REC_709_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][REC_709_ILLUMINANT])
"""
*Rec. 709* colourspace whitepoint.

REC_709_WHITEPOINT : ndarray
"""

REC_709_TO_XYZ_MATRIX = normalised_primary_matrix(
    REC_709_PRIMARIES, REC_709_WHITEPOINT)
"""
*Rec. 709* colourspace to *CIE XYZ* tristimulus values matrix.

REC_709_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_REC_709_MATRIX = np.linalg.inv(REC_709_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Rec. 709* colourspace matrix.

XYZ_TO_REC_709_MATRIX : array_like, (3, 3)
"""

REC_709_COLOURSPACE = RGB_Colourspace(
    'Rec. 709',
    REC_709_PRIMARIES,
    REC_709_WHITEPOINT,
    REC_709_ILLUMINANT,
    REC_709_TO_XYZ_MATRIX,
    XYZ_TO_REC_709_MATRIX,
    oetf_BT709,
    eotf_BT709)
"""
*Rec. 709* colourspace.

REC_709_COLOURSPACE : RGB_Colourspace
"""
