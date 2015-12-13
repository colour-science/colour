#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Beta RGB Colourspace
====================

Defines the *Beta RGB* colourspace:

-   :attr:`BETA_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Lindbloom, B. (2014). RGB Working Space Information. Retrieved April
        11, 2014, from http://www.brucelindbloom.com/WorkingSpaceInfo.html
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

__all__ = ['BETA_RGB_PRIMARIES',
           'BETA_RGB_ILLUMINANT',
           'BETA_RGB_WHITEPOINT',
           'BETA_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_BETA_RGB_MATRIX',
           'BETA_RGB_OECF',
           'BETA_RGB_EOCF',
           'BETA_RGB_COLOURSPACE']

BETA_RGB_PRIMARIES = np.array(
    [[0.6888, 0.3112],
     [0.1986, 0.7551],
     [0.1265, 0.0352]])
"""
*Beta RGB* colourspace primaries.

BETA_RGB_PRIMARIES : ndarray, (3, 2)
"""

BETA_RGB_ILLUMINANT = 'D50'
"""
*Beta RGB* colourspace whitepoint name as illuminant.

BETA_RGB_ILLUMINANT : unicode
"""

BETA_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(BETA_RGB_ILLUMINANT)
"""
*Beta RGB* colourspace whitepoint.

BETA_RGB_WHITEPOINT : tuple
"""

BETA_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(BETA_RGB_PRIMARIES,
                                                   BETA_RGB_WHITEPOINT)
"""
*Beta RGB* colourspace to *CIE XYZ* tristimulus values matrix.

BETA_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_BETA_RGB_MATRIX = np.linalg.inv(BETA_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Beta RGB* colourspace matrix.

XYZ_TO_BETA_RGB_MATRIX : array_like, (3, 3)
"""


def _beta_rgb_OECF(value):
    """
    Defines the *Beta RGB* colourspace opto-electronic conversion function.

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

    return value ** (1 / 2.2)


def _beta_rgb_EOCF(value):
    """
    Defines the *Beta RGB* colourspace electro-optical conversion function.

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

    return value ** 2.2


BETA_RGB_OECF = _beta_rgb_OECF
"""
Opto-electronic conversion function of *Beta RGB* colourspace.

BETA_RGB_OECF : object
"""

BETA_RGB_EOCF = _beta_rgb_EOCF
"""
Electro-optical conversion function of *Beta RGB* colourspace.

BETA_RGB_EOCF : object
"""

BETA_RGB_COLOURSPACE = RGB_Colourspace(
    'Beta RGB',
    BETA_RGB_PRIMARIES,
    BETA_RGB_WHITEPOINT,
    BETA_RGB_ILLUMINANT,
    BETA_RGB_TO_XYZ_MATRIX,
    XYZ_TO_BETA_RGB_MATRIX,
    BETA_RGB_OECF,
    BETA_RGB_EOCF)
"""
*Beta RGB* colourspace.

BETA_RGB_COLOURSPACE : RGB_Colourspace
"""
