#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ekta Space PS 5 Colourspace
===========================

Defines the *Ekta Space PS 5* colourspace:

-   :attr:`EKTA_SPACE_PS_5_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  Holmes, J. (n.d.). Ekta Space PS 5. Retrieved from
        http://www.josephholmes.com/Ekta_Space.zip
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

__all__ = ['EKTA_SPACE_PS_5_PRIMARIES',
           'EKTA_SPACE_PS_5_V_ILLUMINANT',
           'EKTA_SPACE_PS_5_WHITEPOINT',
           'EKTA_SPACE_PS_5_TO_XYZ_MATRIX',
           'XYZ_TO_EKTA_SPACE_PS_5_MATRIX',
           'EKTA_SPACE_PS_5_TRANSFER_FUNCTION',
           'EKTA_SPACE_PS_5_INVERSE_TRANSFER_FUNCTION',
           'EKTA_SPACE_PS_5_COLOURSPACE']

EKTA_SPACE_PS_5_PRIMARIES = np.array(
    [[0.6947368421052631, 0.30526315789473685],
     [0.26000000000000001, 0.69999999999999996],
     [0.10972850678733032, 0.0045248868778280547]])
"""
*Ekta Space PS 5* colourspace primaries.

EKTA_SPACE_PS_5_PRIMARIES : ndarray, (3, 2)
"""

EKTA_SPACE_PS_5_V_ILLUMINANT = 'D50'
"""
*Ekta Space PS 5* colourspace whitepoint name as illuminant.

EKTA_SPACE_PS_5_V_ILLUMINANT : unicode
"""

EKTA_SPACE_PS_5_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(EKTA_SPACE_PS_5_V_ILLUMINANT)
"""
*Ekta Space PS 5* colourspace whitepoint.

EKTA_SPACE_PS_5_WHITEPOINT : tuple
"""

EKTA_SPACE_PS_5_TO_XYZ_MATRIX = normalised_primary_matrix(
    EKTA_SPACE_PS_5_PRIMARIES, EKTA_SPACE_PS_5_WHITEPOINT)
"""
*Ekta Space PS 5* colourspace to *CIE XYZ* colourspace matrix.

EKTA_SPACE_PS_5_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_EKTA_SPACE_PS_5_MATRIX = np.linalg.inv(EKTA_SPACE_PS_5_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *Ekta Space PS 5* colourspace matrix.

XYZ_TO_EKTA_SPACE_PS_5_MATRIX : array_like, (3, 3)
"""


def _ekta_space_ps_5_transfer_function(value):
    """
    Defines the *Ekta Space PS 5* value colourspace transfer function.

    Parameters
    ----------
    value : numeric
        value.

    Returns
    -------
    numeric
        Companded value.
    """

    return value ** (1 / 2.2)


def _ekta_space_ps_5_inverse_transfer_function(value):
    """
    Defines the *Ekta Space PS 5* value colourspace inverse transfer
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

    return value ** 2.2


EKTA_SPACE_PS_5_TRANSFER_FUNCTION = _ekta_space_ps_5_transfer_function
"""
Transfer function from linear to *Ekta Space PS 5* colourspace.

EKTA_SPACE_PS_5_TRANSFER_FUNCTION : object
"""

EKTA_SPACE_PS_5_INVERSE_TRANSFER_FUNCTION = (
    _ekta_space_ps_5_inverse_transfer_function)
"""
Inverse transfer function from *Ekta Space PS 5* colourspace to linear.

EKTA_SPACE_PS_5_INVERSE_TRANSFER_FUNCTION : object
"""

EKTA_SPACE_PS_5_COLOURSPACE = RGB_Colourspace(
    'Ekta Space PS 5',
    EKTA_SPACE_PS_5_PRIMARIES,
    EKTA_SPACE_PS_5_WHITEPOINT,
    EKTA_SPACE_PS_5_V_ILLUMINANT,
    EKTA_SPACE_PS_5_TO_XYZ_MATRIX,
    XYZ_TO_EKTA_SPACE_PS_5_MATRIX,
    EKTA_SPACE_PS_5_TRANSFER_FUNCTION,
    EKTA_SPACE_PS_5_INVERSE_TRANSFER_FUNCTION)
"""
*Ekta Space PS 5* colourspace.

EKTA_SPACE_PS_5_COLOURSPACE : RGB_Colourspace
"""
