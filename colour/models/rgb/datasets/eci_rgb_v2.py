# -*- coding: utf-8 -*-
"""
ECI RGB v2 Colourspace
======================

Defines the *ECI RGB v2* colourspace:

-   :attr:`colour.models.ECI_RGB_V2_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`EuropeanColorInitiative2002a` : European Color Initiative. (2002).
    ECI RGB v2. Retrieved from
    http://www.eci.org/_media/downloads/icc_profiles_from_eci/ecirgbv20.zip
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import (ILLUMINANTS, lightness_CIE1976,
                                luminance_CIE1976)
from colour.models.rgb import RGB_Colourspace, normalised_primary_matrix
from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'ECI_RGB_V2_PRIMARIES', 'ECI_RGB_V_WHITEPOINT_NAME',
    'ECI_RGB_V2_WHITEPOINT', 'ECI_RGB_V2_TO_XYZ_MATRIX',
    'XYZ_TO_ECI_RGB_V2_MATRIX', 'ECI_RGB_V2_COLOURSPACE'
]

ECI_RGB_V2_PRIMARIES = np.array([
    [0.670103092783505, 0.329896907216495],
    [0.209905660377358, 0.709905660377358],
    [0.140061791967044, 0.080329557157570],
])
"""
*ECI RGB v2* colourspace primaries.

ECI_RGB_V2_PRIMARIES : ndarray, (3, 2)
"""

ECI_RGB_V_WHITEPOINT_NAME = 'D50'
"""
*ECI RGB v2* colourspace whitepoint name.

ECI_RGB_V_WHITEPOINT_NAME : unicode
"""

ECI_RGB_V2_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    ECI_RGB_V_WHITEPOINT_NAME])
"""
*ECI RGB v2* colourspace whitepoint.

ECI_RGB_V2_WHITEPOINT : ndarray
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


def _scale_domain_0_100_range_0_1(a, callable_):
    """
    Scales the input domain of given *luminance* :math:`Y` or *Lightness*
    :math:`L^*` array to [0, 100], call the given callable, and
    scales the output range to [0, 1].

    Parameters
    ----------
    a : numeric or array_like
        *Luminance* :math:`Y` or *Lightness* :math:`L^*` array.
    callable_ : callable
        *Luminance* :math:`Y` or *Lightness* :math:`L^*` computation
        definition, i.e., :func:`colour.colorimetry.lightness_CIE1976` or
        :func:`colour.colorimetry.luminance_CIE1976`. Reference white
        *luminance* :math:`Y_n` has implicit value of :math:`100\\ cd/m^2`.

    Returns
    -------
    numeric or ndarray
        Scaled *luminance* :math:`Y` or *Lightness* :math:`L^*` array.
    """

    a = as_float_array(a)

    return callable_(a * 100, Y_n=100) / 100


ECI_RGB_V2_COLOURSPACE = RGB_Colourspace(
    'ECI RGB v2',
    ECI_RGB_V2_PRIMARIES,
    ECI_RGB_V2_WHITEPOINT,
    ECI_RGB_V_WHITEPOINT_NAME,
    ECI_RGB_V2_TO_XYZ_MATRIX,
    XYZ_TO_ECI_RGB_V2_MATRIX,
    partial(_scale_domain_0_100_range_0_1, callable_=lightness_CIE1976),
    partial(_scale_domain_0_100_range_0_1, callable_=luminance_CIE1976),
)
ECI_RGB_V2_COLOURSPACE.__doc__ = """
*ECI RGB v2* colourspace.

References
----------
:cite:`EuropeanColorInitiative2002a`

ECI_RGB_V2_COLOURSPACE : RGB_Colourspace
"""
