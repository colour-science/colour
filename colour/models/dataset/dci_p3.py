#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DCI-P3 Colourspace
==================

Defines the *DCI-P3* colourspace:

-   :attr:`DCI_P3_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  http://www.hp.com/united-states/campaigns/workstations/pdfs/lp2480zx-dci--p3-emulation.pdf  # noqa
        (Last accessed 24 February 2014)
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models import RGB_Colourspace

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['DCI_P3_PRIMARIES',
           'DCI_P3_WHITEPOINT',
           'DCI_P3_TO_XYZ_MATRIX',
           'XYZ_TO_DCI_P3_MATRIX',
           'DCI_P3_TRANSFER_FUNCTION',
           'DCI_P3_INVERSE_TRANSFER_FUNCTION',
           'DCI_P3_COLOURSPACE']

DCI_P3_PRIMARIES = np.array(
    [[0.680, 0.320],
     [0.265, 0.690],
     [0.150, 0.060]])
"""
*DCI-P3* colourspace primaries.

DCI_P3_PRIMARIES : ndarray, (3, 2)
"""

DCI_P3_WHITEPOINT = (0.314, 0.351)
"""
*DCI-P3* colourspace whitepoint.

DCI_P3_WHITEPOINT : tuple
"""

DCI_P3_TO_XYZ_MATRIX = np.array(
    [[0.44516982, 0.27713441, 0.17228267],
     [0.20949168, 0.72159525, 0.06891307],
     [0, 0.04706056, 0.90735539]])
"""
*DCI-P3* colourspace to *CIE XYZ* colourspace matrix.

DCI_P3_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DCI_P3_MATRIX = np.linalg.inv(DCI_P3_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *DCI-P3* colourspace matrix.

XYZ_TO_DCI_P3_MATRIX : array_like, (3, 3)
"""

DCI_P3_TRANSFER_FUNCTION = lambda x: x
"""
Transfer function from linear to *DCI-P3* colourspace.

DCI_P3_TRANSFER_FUNCTION : object
"""

DCI_P3_INVERSE_TRANSFER_FUNCTION = lambda x: x
"""
Inverse transfer function from *DCI-P3* colourspace to linear.

DCI_P3_INVERSE_TRANSFER_FUNCTION : object
"""

DCI_P3_COLOURSPACE = RGB_Colourspace(
    'DCI-P3',
    DCI_P3_PRIMARIES,
    DCI_P3_WHITEPOINT,
    DCI_P3_TO_XYZ_MATRIX,
    XYZ_TO_DCI_P3_MATRIX,
    DCI_P3_TRANSFER_FUNCTION,
    DCI_P3_INVERSE_TRANSFER_FUNCTION)
"""
*DCI-P3* colourspace.

DCI_P3_COLOURSPACE : RGB_Colourspace
"""
