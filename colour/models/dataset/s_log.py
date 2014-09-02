#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sony S-Log Colourspace
======================

Defines the *S-Log* colourspace:

-   :attr:`S_LOG_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  `S-Log: A new LUT for digital production mastering and interchange
        applications
        <http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/slog_manual.pdf>`_  # noqa
        (Last accessed 13 April 2014)
"""

from __future__ import division, unicode_literals

import math
import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['S_LOG_PRIMARIES',
           'S_LOG_WHITEPOINT',
           'S_LOG_TO_XYZ_MATRIX',
           'XYZ_TO_S_LOG_MATRIX',
           'S_LOG_TRANSFER_FUNCTION',
           'S_LOG_INVERSE_TRANSFER_FUNCTION',
           'S_LOG_COLOURSPACE']

S_LOG_PRIMARIES = np.array(
    [[0.73, 0.28],
     [0.14, 0.855],
     [0.10, -0.05]])
"""
*S-Log* colourspace primaries.

S_LOG_PRIMARIES : ndarray, (3, 2)
"""

S_LOG_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get('D65')
"""
*S-Log* colourspace whitepoint.

S_LOG_WHITEPOINT : tuple
"""

S_LOG_TO_XYZ_MATRIX = normalised_primary_matrix(S_LOG_PRIMARIES,
                                                S_LOG_WHITEPOINT)
"""
*S-Log* colourspace to *CIE XYZ* colourspace matrix.

S_LOG_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_S_LOG_MATRIX = np.linalg.inv(S_LOG_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *S-Log* colourspace matrix.

XYZ_TO_S_LOG_MATRIX : array_like, (3, 3)
"""

S_LOG_TRANSFER_FUNCTION = lambda x: (
    (0.432699 * math.log10(x + 0.037584) + 0.616596) + 0.03)
"""
Transfer function from linear to *S-Log* colourspace.

S_LOG_TRANSFER_FUNCTION : object
"""

S_LOG_INVERSE_TRANSFER_FUNCTION = lambda x: (
    (math.pow(10, ((x - 0.616596 - 0.03) / 0.432699)) - 0.037584))
"""
Inverse transfer function from *S-Log* colourspace to linear.

S_LOG_INVERSE_TRANSFER_FUNCTION : object
"""

S_LOG_COLOURSPACE = RGB_Colourspace(
    'S-Log',
    S_LOG_PRIMARIES,
    S_LOG_WHITEPOINT,
    S_LOG_TO_XYZ_MATRIX,
    XYZ_TO_S_LOG_MATRIX,
    S_LOG_TRANSFER_FUNCTION,
    S_LOG_INVERSE_TRANSFER_FUNCTION)
"""
*S-Log* colourspace.

S_LOG_COLOURSPACE : RGB_Colourspace
"""
