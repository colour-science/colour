#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DCI-P3 & DCI-P3+ Colourspaces
=============================

Defines the *DCI-P3* and *DCI-P3+* colourspaces:

-   :attr:`DCI_P3_COLOURSPACE`.
-   :attr:`DCI_P3_p_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Hewlett-Packard Development Company. (2009). Understanding the HP
        DreamColor LP2480zx DCI-P3 Emulation Color Space. Retrieved from
        http://www.hp.com/united-states/campaigns/workstations/pdfs/\
lp2480zx-dci--p3-emulation.pdf
.. [2]  Digital Cinema Initiatives. (2007). Digital Cinema System
        Specification - Version 1.1. Retrieved from
        http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
.. [3]  Canon. (2014). EOS C500 Firmware Update. Retrieved January 14, 2015,
        from http://www.usa.canon.com/cusa/professional/standard_display/\
cinema-firmware-c500
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['DCI_P3_PRIMARIES',
           'DCI_P3_P_PRIMARIES',
           'DCI_P3_ILLUMINANT',
           'DCI_P3_WHITEPOINT',
           'DCI_P3_TO_XYZ_MATRIX',
           'XYZ_TO_DCI_P3_MATRIX',
           'DCI_P3_P_TO_XYZ_MATRIX',
           'XYZ_TO_DCI_P3_P_MATRIX',
           'DCI_P3_OECF',
           'DCI_P3_EOCF',
           'DCI_P3_COLOURSPACE',
           'DCI_P3_P_COLOURSPACE']

DCI_P3_PRIMARIES = np.array(
    [[0.680, 0.320],
     [0.265, 0.690],
     [0.150, 0.060]])
"""
*DCI-P3* colourspace primaries.

DCI_P3_PRIMARIES : ndarray, (3, 2)
"""

DCI_P3_P_PRIMARIES = np.array(
    [[0.7400, 0.2700],
     [0.2200, 0.7800],
     [0.0900, -0.0900]])
"""
*DCI-P3+* colourspace primaries.

DCI_P3_P_PRIMARIES : ndarray, (3, 2)
"""

DCI_P3_ILLUMINANT = 'DCI-P3'
"""
*DCI-P3* colourspace whitepoint name as illuminant.

DCI_P3_ILLUMINANT : unicode

Warning
-------
DCI-P3 illuminant has no associated spectral power distribution. DCI has no
official reference spectral measurement for this whitepoint. The closest
matching spectral power distribution is Kinoton 75P projector.
"""

DCI_P3_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(DCI_P3_ILLUMINANT)
"""
*DCI-P3* colourspace whitepoint.

DCI_P3_WHITEPOINT : tuple
"""

DCI_P3_TO_XYZ_MATRIX = normalised_primary_matrix(
    DCI_P3_PRIMARIES,
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

DCI_P3_P_TO_XYZ_MATRIX = normalised_primary_matrix(
    DCI_P3_P_PRIMARIES,
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


def _dci_p3_OECF(value):
    """
    Defines the *DCI-P3* colourspace opto-electronic conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return 4095 * (value / 52.37) ** (1 / 2.6)


def _dci_p3_EOCF(value):
    """
    Defines the *DCI-P3* colourspace electro-optical conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return 52.37 * (value / 4095) ** 2.6


DCI_P3_OECF = _dci_p3_OECF
"""
Opto-electronic conversion function of *DCI-P3* colourspace.

DCI_P3_OECF : object
"""

DCI_P3_EOCF = _dci_p3_EOCF
"""
Electro-optical conversion function of *DCI-P3* colourspace.

DCI_P3_EOCF : object
"""

DCI_P3_COLOURSPACE = RGB_Colourspace(
    'DCI-P3',
    DCI_P3_PRIMARIES,
    DCI_P3_WHITEPOINT,
    DCI_P3_ILLUMINANT,
    DCI_P3_TO_XYZ_MATRIX,
    XYZ_TO_DCI_P3_MATRIX,
    DCI_P3_OECF,
    DCI_P3_EOCF)
"""
*DCI-P3* colourspace.

DCI_P3_COLOURSPACE : RGB_Colourspace
"""

DCI_P3_P_COLOURSPACE = RGB_Colourspace(
    'DCI-P3+',
    DCI_P3_P_PRIMARIES,
    DCI_P3_WHITEPOINT,
    DCI_P3_ILLUMINANT,
    DCI_P3_P_TO_XYZ_MATRIX,
    XYZ_TO_DCI_P3_P_MATRIX,
    DCI_P3_OECF,
    DCI_P3_EOCF)
"""
*DCI-P3+* colourspace.

DCI_P3_P_COLOURSPACE : RGB_Colourspace
"""
