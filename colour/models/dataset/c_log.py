#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canon C-Log Colourspace
=======================

Defines the *C-Log* colourspace:

-   :attr:`C_LOG_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  Thorpe, L. (2012). CANON-LOG TRANSFER CHARACTERISTIC. Retrieved from
        http://downloads.canon.com/CDLC/Canon-Log_Transfer_Characteristic_6-20-2012.pdf  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['C_LOG_PRIMARIES',
           'C_LOG_ILLUMINANT',
           'C_LOG_WHITEPOINT',
           'C_LOG_TO_XYZ_MATRIX',
           'XYZ_TO_C_LOG_MATRIX',
           'C_LOG_TRANSFER_FUNCTION',
           'C_LOG_INVERSE_TRANSFER_FUNCTION',
           'C_LOG_COLOURSPACE']

C_LOG_PRIMARIES = np.array(
    [[0.6400, 0.3300],
     [0.3000, 0.6000],
     [0.1500, 0.0600]])
"""
*C-Log* colourspace primaries,

C_LOG_PRIMARIES : ndarray, (3, 2)
"""

C_LOG_ILLUMINANT = 'D65'
"""
*C-Log* colourspace whitepoint name as illuminant.

C_LOG_ILLUMINANT : unicode
"""

C_LOG_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(C_LOG_ILLUMINANT)
"""
*C-Log* colourspace whitepoint.

C_LOG_WHITEPOINT : tuple
"""

C_LOG_TO_XYZ_MATRIX = normalised_primary_matrix(C_LOG_PRIMARIES,
                                                C_LOG_WHITEPOINT)
"""
*C-Log* colourspace to *CIE XYZ* colourspace matrix.

C_LOG_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_C_LOG_MATRIX = np.linalg.inv(C_LOG_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *C-Log* colourspace matrix.

XYZ_TO_C_LOG_MATRIX : array_like, (3, 3)
"""


def _c_log_transfer_function(value):
    """
    Defines the *C-Log* value colourspace transfer function.

    Parameters
    ----------
    value : numeric
        value.

    Returns
    -------
    numeric
        Companded value.
    """

    return 0.529136 * np.log10(10.1596 * value + 1) + 0.0730597


def _c_log_inverse_transfer_function(value):
    """
    Defines the *C-Log* value colourspace inverse transfer
    function.

    Parameters
    ----------
    value : numeric
        value.

    Returns
    -------
    numeric
        Companded value.
    """

    return -0.071622555735168 * (
        1.3742747797867 - np.exp(1) ** (4.3515940948906 * value))


C_LOG_TRANSFER_FUNCTION = _c_log_transfer_function
"""
Transfer function from linear to *C-Log* colourspace.

C_LOG_TRANSFER_FUNCTION : object
"""

C_LOG_INVERSE_TRANSFER_FUNCTION = _c_log_inverse_transfer_function
"""
Inverse transfer function from *C-Log* colourspace to linear.

C_LOG_INVERSE_TRANSFER_FUNCTION : object
"""

C_LOG_COLOURSPACE = RGB_Colourspace(
    'C-Log',
    C_LOG_PRIMARIES,
    C_LOG_WHITEPOINT,
    C_LOG_ILLUMINANT,
    C_LOG_TO_XYZ_MATRIX,
    XYZ_TO_C_LOG_MATRIX,
    C_LOG_TRANSFER_FUNCTION,
    C_LOG_INVERSE_TRANSFER_FUNCTION)
"""
*C-Log* colourspace.

C_LOG_COLOURSPACE : RGB_Colourspace
"""
