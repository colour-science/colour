#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pal/Secam RGB Colourspace
=========================

Defines the *Pal/Secam RGB* colourspace:

-   :attr:`PAL_SECAM_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (1998). CONVENTIONAL TELEVISION
        SYSTEMS. In Recommendation ITU-R BT.470-6 (pp. 1–36). Retrieved from
        http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.470-6-199811-S!!PDF-E.pdf
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

__all__ = ['PAL_SECAM_RGB_PRIMARIES',
           'PAL_SECAM_RGB_ILLUMINANT',
           'PAL_SECAM_RGB_WHITEPOINT',
           'PAL_SECAM_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_PAL_SECAM_RGB_MATRIX',
           'PAL_SECAM_RGB_OECF',
           'PAL_SECAM_RGB_EOCF',
           'PAL_SECAM_RGB_COLOURSPACE']

PAL_SECAM_RGB_PRIMARIES = np.array(
    [[0.64, 0.33],
     [0.29, 0.60],
     [0.15, 0.06]])
"""
*Pal/Secam RGB* colourspace primaries.

PAL_SECAM_RGB_PRIMARIES : ndarray, (3, 2)
"""

PAL_SECAM_RGB_ILLUMINANT = 'D65'
"""
*Pal/Secam RGB* colourspace whitepoint name as illuminant.

PAL_SECAM_RGB_ILLUMINANT : unicode
"""

PAL_SECAM_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(PAL_SECAM_RGB_ILLUMINANT)
"""
*Pal/Secam RGB* colourspace whitepoint.

PAL_SECAM_RGB_WHITEPOINT : tuple
"""

PAL_SECAM_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    PAL_SECAM_RGB_PRIMARIES, PAL_SECAM_RGB_WHITEPOINT)
"""
*Pal/Secam RGB* colourspace to *CIE XYZ* tristimulus values matrix.

PAL_SECAM_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_PAL_SECAM_RGB_MATRIX = np.linalg.inv(PAL_SECAM_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Pal/Secam RGB* colourspace matrix.

XYZ_TO_PAL_SECAM_RGB_MATRIX : array_like, (3, 3)
"""


def _pal_secam_rgb_OECF(value):
    """
    Defines the *Pal/Secam RGB* colourspace opto-electronic conversion
    function.

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

    return value ** (1 / 2.8)


def _pal_secam_rgb_EOCF(value):
    """
    Defines the *Pal/Secam RGB* colourspace electro-optical conversion
    function.

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

    return value ** 2.8


PAL_SECAM_RGB_OECF = _pal_secam_rgb_OECF
"""
Opto-electronic conversion function of *Pal/Secam RGB* colourspace.

PAL_SECAM_RGB_OECF : object
"""

PAL_SECAM_RGB_EOCF = (
    _pal_secam_rgb_EOCF)
"""
Electro-optical conversion function of *Pal/Secam RGB* colourspace.

PAL_SECAM_RGB_EOCF : object
"""

PAL_SECAM_RGB_COLOURSPACE = RGB_Colourspace(
    'Pal/Secam RGB',
    PAL_SECAM_RGB_PRIMARIES,
    PAL_SECAM_RGB_WHITEPOINT,
    PAL_SECAM_RGB_ILLUMINANT,
    PAL_SECAM_RGB_TO_XYZ_MATRIX,
    XYZ_TO_PAL_SECAM_RGB_MATRIX,
    PAL_SECAM_RGB_OECF,
    PAL_SECAM_RGB_EOCF)
"""
*Pal/Secam RGB* colourspace.

PAL_SECAM_RGB_COLOURSPACE : RGB_Colourspace
"""
