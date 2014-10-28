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
.. [1]  Gaggioni, H., Dhanendra, P., Yamashita, J., Kawada, N., Endo, K., &
        Clark, C. (n.d.). S-Log: A new LUT for digital production mastering
        and interchange applications. Retrieved from
        http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/slog_manual.pdf  # noqa
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

__all__ = ['S_LOG_PRIMARIES',
           'S_LOG_ILLUMINANT',
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

S_LOG_ILLUMINANT = 'D65'
"""
*S-Log* colourspace whitepoint name as illuminant.

S_LOG_ILLUMINANT : unicode
"""

S_LOG_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(S_LOG_ILLUMINANT)
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


def _s_log_transfer_function(value):
    """
    Defines the *S-Log* value colourspace transfer function.

    Parameters
    ----------
    value : numeric
        value.

    Returns
    -------
    numeric
        Companded value.
    """

    return (0.432699 * np.log10(value + 0.037584) + 0.616596) + 0.03


def _s_log_inverse_transfer_function(value):
    """
    Defines the *S-Log* value colourspace inverse transfer
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

    return (np.power(10., ((value - 0.616596 - 0.03) / 0.432699)) - 0.037584)


S_LOG_TRANSFER_FUNCTION = _s_log_transfer_function
"""
Transfer function from linear to *S-Log* colourspace.

S_LOG_TRANSFER_FUNCTION : object
"""

S_LOG_INVERSE_TRANSFER_FUNCTION = _s_log_inverse_transfer_function
"""
Inverse transfer function from *S-Log* colourspace to linear.

S_LOG_INVERSE_TRANSFER_FUNCTION : object
"""

S_LOG_COLOURSPACE = RGB_Colourspace(
    'S-Log',
    S_LOG_PRIMARIES,
    S_LOG_WHITEPOINT,
    S_LOG_ILLUMINANT,
    S_LOG_TO_XYZ_MATRIX,
    XYZ_TO_S_LOG_MATRIX,
    S_LOG_TRANSFER_FUNCTION,
    S_LOG_INVERSE_TRANSFER_FUNCTION)
"""
*S-Log* colourspace.

S_LOG_COLOURSPACE : RGB_Colourspace
"""
