#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Best RGB Colourspace
====================

Defines the *Best RGB* colourspace:

-   :attr:`BEST_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  HutchColor. (n.d.). BestRGB (4 K). Retrieved from
        http://www.hutchcolor.com/profiles/BestRGB.zip
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

__all__ = ['BEST_RGB_PRIMARIES',
           'BEST_RGB_ILLUMINANT',
           'BEST_RGB_WHITEPOINT',
           'BEST_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_BEST_RGB_MATRIX',
           'BEST_RGB_OECF',
           'BEST_RGB_EOCF',
           'BEST_RGB_COLOURSPACE']

BEST_RGB_PRIMARIES = np.array(
    [[0.735191637630662090, 0.264808362369337970],
     [0.215336134453781500, 0.774159663865546240],
     [0.130122950819672120, 0.034836065573770496]])
"""
*Best RGB* colourspace primaries.

BEST_RGB_PRIMARIES : ndarray, (3, 2)
"""

BEST_RGB_ILLUMINANT = 'D50'
"""
*Best RGB* colourspace whitepoint name as illuminant.

BEST_RGB_ILLUMINANT : unicode
"""

BEST_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(BEST_RGB_ILLUMINANT)
"""
*Best RGB* colourspace whitepoint.

BEST_RGB_WHITEPOINT : tuple
"""

BEST_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(BEST_RGB_PRIMARIES,
                                                   BEST_RGB_WHITEPOINT)
"""
*Best RGB* colourspace to *CIE XYZ* tristimulus values matrix.

BEST_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_BEST_RGB_MATRIX = np.linalg.inv(BEST_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Best RGB* colourspace matrix.

XYZ_TO_BEST_RGB_MATRIX : array_like, (3, 3)
"""


def _best_rgb_OECF(value):
    """
    Defines the *Best RGB* colourspace opto-electronic conversion function.

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


def _best_rgb_EOCF(value):
    """
    Defines the *Best RGB* colourspace electro-optical conversion function.

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


BEST_RGB_OECF = _best_rgb_OECF
"""
Opto-electronic conversion function of *Best RGB* colourspace.

BEST_RGB_OECF : object
"""

BEST_RGB_EOCF = _best_rgb_EOCF
"""
Electro-optical conversion function of *Best RGB* colourspace.

BEST_RGB_EOCF : object
"""

BEST_RGB_COLOURSPACE = RGB_Colourspace(
    'Best RGB',
    BEST_RGB_PRIMARIES,
    BEST_RGB_WHITEPOINT,
    BEST_RGB_ILLUMINANT,
    BEST_RGB_TO_XYZ_MATRIX,
    XYZ_TO_BEST_RGB_MATRIX,
    BEST_RGB_OECF,
    BEST_RGB_EOCF)
"""
*Best RGB* colourspace.

BEST_RGB_COLOURSPACE : RGB_Colourspace
"""
