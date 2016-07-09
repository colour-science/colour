#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ALEXA Wide Gamut RGB Colourspace
================================

Defines the *ALEXA Wide Gamut RGB* colourspace:

-   :attr:`ALEXA_WIDE_GAMUT_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  ARRI. (2012). ALEXA - Log C Curve - Usage in VFX. Retrieved from
        http://www.arri.com/?eID=registration&file_uid=8026
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    log_encoding_ALEXALogC,
    log_decoding_ALEXALogC)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ALEXA_WIDE_GAMUT_RGB_PRIMARIES',
           'ALEXA_WIDE_GAMUT_RGB_ILLUMINANT',
           'ALEXA_WIDE_GAMUT_RGB_WHITEPOINT',
           'ALEXA_WIDE_GAMUT_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_ALEXA_WIDE_GAMUT_RGB_MATRIX',
           'ALEXA_WIDE_GAMUT_RGB_COLOURSPACE']

ALEXA_WIDE_GAMUT_RGB_PRIMARIES = np.array(
    [[0.6840, 0.3130],
     [0.2210, 0.8480],
     [0.0861, -0.1020]])
"""
*ALEXA Wide Gamut RGB* colourspace primaries.

ALEXA_WIDE_GAMUT_RGB_PRIMARIES : ndarray, (3, 2)
"""

ALEXA_WIDE_GAMUT_RGB_ILLUMINANT = 'D65'
"""
*ALEXA Wide Gamut RGB* colourspace whitepoint name as illuminant.

ALEXA_WIDE_GAMUT_RGB_WHITEPOINT : unicode
"""

ALEXA_WIDE_GAMUT_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(ALEXA_WIDE_GAMUT_RGB_ILLUMINANT)
"""
*ALEXA Wide Gamut RGB* colourspace whitepoint.

ALEXA_WIDE_GAMUT_RGB_WHITEPOINT : tuple
"""

ALEXA_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = np.array(
    [[0.638008, 0.214704, 0.097744],
     [0.291954, 0.823841, -0.115795],
     [0.002798, -0.067034, 1.153294]])
"""
*ALEXA Wide Gamut RGB* colourspace to *CIE XYZ* tristimulus values matrix.

ALEXA_WIDE_GAMUT_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ALEXA_WIDE_GAMUT_RGB_MATRIX = np.linalg.inv(
    ALEXA_WIDE_GAMUT_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *ALEXA Wide Gamut RGB* colourspace matrix.

XYZ_TO_ALEXA_WIDE_GAMUT_RGB_MATRIX : array_like, (3, 3)
"""

ALEXA_WIDE_GAMUT_RGB_COLOURSPACE = RGB_Colourspace(
    'ALEXA Wide Gamut RGB',
    ALEXA_WIDE_GAMUT_RGB_PRIMARIES,
    ALEXA_WIDE_GAMUT_RGB_WHITEPOINT,
    ALEXA_WIDE_GAMUT_RGB_ILLUMINANT,
    ALEXA_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ALEXA_WIDE_GAMUT_RGB_MATRIX,
    log_encoding_ALEXALogC,
    log_decoding_ALEXALogC)
"""
*ALEXA Wide Gamut RGB* colourspace.

ALEXA_WIDE_GAMUT_RGB_COLOURSPACE : RGB_Colourspace
"""
