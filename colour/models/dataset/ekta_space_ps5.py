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
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

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
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['EKTA_SPACE_PS_5_PRIMARIES',
           'EKTA_SPACE_PS_5_V_ILLUMINANT',
           'EKTA_SPACE_PS_5_WHITEPOINT',
           'EKTA_SPACE_PS_5_TO_XYZ_MATRIX',
           'XYZ_TO_EKTA_SPACE_PS_5_MATRIX',
           'EKTA_SPACE_PS_5_OECF',
           'EKTA_SPACE_PS_5_EOCF',
           'EKTA_SPACE_PS_5_COLOURSPACE']

EKTA_SPACE_PS_5_PRIMARIES = np.array(
    [[0.694736842105263100, 0.305263157894736850],
     [0.260000000000000010, 0.699999999999999960],
     [0.109728506787330320, 0.004524886877828055]])
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
*Ekta Space PS 5* colourspace to *CIE XYZ* tristimulus values matrix.

EKTA_SPACE_PS_5_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_EKTA_SPACE_PS_5_MATRIX = np.linalg.inv(EKTA_SPACE_PS_5_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Ekta Space PS 5* colourspace matrix.

XYZ_TO_EKTA_SPACE_PS_5_MATRIX : array_like, (3, 3)
"""


def _ekta_space_ps_5_OECF(value):
    """
    Defines the *Ekta Space PS 5* colourspace opto-electronic conversion
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

    return value ** (1 / 2.2)


def _ekta_space_ps_5_EOCF(value):
    """
    Defines the *Ekta Space PS 5* colourspace electro-optical conversion
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

    return value ** 2.2


EKTA_SPACE_PS_5_OECF = _ekta_space_ps_5_OECF
"""
Opto-electronic conversion function of *Ekta Space PS 5*
colourspace.

EKTA_SPACE_PS_5_OECF : object
"""

EKTA_SPACE_PS_5_EOCF = (
    _ekta_space_ps_5_EOCF)
"""
Electro-optical conversion function of *Ekta Space PS 5* colourspace to
linear.

EKTA_SPACE_PS_5_EOCF : object
"""

EKTA_SPACE_PS_5_COLOURSPACE = RGB_Colourspace(
    'Ekta Space PS 5',
    EKTA_SPACE_PS_5_PRIMARIES,
    EKTA_SPACE_PS_5_WHITEPOINT,
    EKTA_SPACE_PS_5_V_ILLUMINANT,
    EKTA_SPACE_PS_5_TO_XYZ_MATRIX,
    XYZ_TO_EKTA_SPACE_PS_5_MATRIX,
    EKTA_SPACE_PS_5_OECF,
    EKTA_SPACE_PS_5_EOCF)
"""
*Ekta Space PS 5* colourspace.

EKTA_SPACE_PS_5_COLOURSPACE : RGB_Colourspace
"""
