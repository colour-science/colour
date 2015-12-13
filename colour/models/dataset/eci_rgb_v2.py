#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECI RGB v2 Colourspace
======================

Defines the *ECI RGB v2* colourspace:

-   :attr:`ECI_RGB_V2_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  European Color Initiative. (2002). ECI RGB v2. Retrieved from
        http://www.eci.org/_media/downloads/icc_profiles_from_eci/ecirgbv20.zip
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS, lightness_1976, luminance_1976
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ECI_RGB_V2_PRIMARIES',
           'ECI_RGB_V_ILLUMINANT',
           'ECI_RGB_V2_WHITEPOINT',
           'ECI_RGB_V2_TO_XYZ_MATRIX',
           'XYZ_TO_ECI_RGB_V2_MATRIX',
           'ECI_RGB_V2_OECF',
           'ECI_RGB_V2_EOCF',
           'ECI_RGB_V2_COLOURSPACE']

ECI_RGB_V2_PRIMARIES = np.array(
    [[0.670103092783505220, 0.329896907216494840],
     [0.209905660377358470, 0.709905660377358360],
     [0.140061791967044270, 0.080329557157569509]])
"""
*ECI RGB v2* colourspace primaries.

ECI_RGB_V2_PRIMARIES : ndarray, (3, 2)
"""

ECI_RGB_V_ILLUMINANT = 'D50'
"""
*ECI RGB v2* colourspace whitepoint name as illuminant.

ECI_RGB_V_ILLUMINANT : unicode
"""

ECI_RGB_V2_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(ECI_RGB_V_ILLUMINANT)
"""
*ECI RGB v2* colourspace whitepoint.

ECI_RGB_V2_WHITEPOINT : tuple
"""

ECI_RGB_V2_TO_XYZ_MATRIX = normalised_primary_matrix(ECI_RGB_V2_PRIMARIES,
                                                     ECI_RGB_V2_WHITEPOINT)
"""
*ECI RGB v2* colourspace to *CIE XYZ* tristimulus values matrix.

ECI_RGB_V2_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ECI_RGB_V2_MATRIX = np.linalg.inv(ECI_RGB_V2_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *ECI RGB v2* colourspace matrix.

XYZ_TO_ECI_RGB_V2_MATRIX : array_like, (3, 3)
"""


def _eci_rgb_v2_OECF(value):
    """
    Defines the *ECI RGB v2* colourspace opto-electronic conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    return lightness_1976(value * 100) / 100


def _eci_rgb_v2_EOCF(value):
    """
    Defines the *ECI RGB v2* colourspace electro-optical conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    return luminance_1976(value * 100) / 100


ECI_RGB_V2_OECF = _eci_rgb_v2_OECF
"""
Opto-electronic conversion function of *ECI RGB v2* colourspace.

ECI_RGB_V2_OECF : object
"""

ECI_RGB_V2_EOCF = _eci_rgb_v2_EOCF
"""
Electro-optical conversion function of *ECI RGB v2* colourspace.

ECI_RGB_V2_EOCF : object
"""

ECI_RGB_V2_COLOURSPACE = RGB_Colourspace(
    'ECI RGB v2',
    ECI_RGB_V2_PRIMARIES,
    ECI_RGB_V2_WHITEPOINT,
    ECI_RGB_V_ILLUMINANT,
    ECI_RGB_V2_TO_XYZ_MATRIX,
    XYZ_TO_ECI_RGB_V2_MATRIX,
    ECI_RGB_V2_OECF,
    ECI_RGB_V2_EOCF)
"""
*ECI RGB v2* colourspace.

ECI_RGB_V2_COLOURSPACE : RGB_Colourspace
"""
