# -*- coding: utf-8 -*-
"""
:math:`J_zA_zB_z` Colourspace
=============================

Defines the :math:`J_zA_zB_z` colourspace:

-   :func:`colour.XYZ_to_JzAzBz`
-   :func:`colour.JzAzBz_to_XYZ`

See Also
--------
`JzAzBz Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/jzazbz.ipynb>`_

References
----------
-   :cite:`Safdar2017` : Safdar, M., Cui, G., Kim, Y. J., & Luo, M. R. (2017).
    Perceptually uniform color space for image signals including high dynamic
    range and wide gamut. Optics Express, 25(13), 15131.
    doi:10.1364/OE.25.015131
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import oetf_ST2084, eotf_ST2084
from colour.models.rgb.transfer_functions.st_2084 import ST2084_CONSTANTS
from colour.utilities import Structure, dot_vector, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'JZAZBZ_CONSTANTS', 'JZAZBZ_XYZ_TO_LMS_MATRIX', 'JZAZBZ_LMS_TO_XYZ_MATRIX',
    'JZAZBZ_LMS_P_TO_IZAZBZ_MATRIX', 'JZAZBZ_IZAZBZ_TO_LMS_P_MATRIX',
    'XYZ_to_JzAzBz', 'JzAzBz_to_XYZ'
]

JZAZBZ_CONSTANTS = Structure(
    b=1.15, g=0.66, d=-0.56, d_0=1.6295499532821566 * 10 ** -11)
JZAZBZ_CONSTANTS.update(ST2084_CONSTANTS)
JZAZBZ_CONSTANTS.m_2 = 1.7 * 2523 / 2 ** 5
"""
Constants for :math:`J_zA_zB_z` colourspace and its variant of the perceptual
quantizer (PQ) from Dolby Laboratories.

Notes
-----
-   The :math:`m2` constant, i.e. the power factor has been re-optimized during
    the development of the :math:`J_zA_zB_z` colourspace.

JZAZBZ_CONSTANTS : Structure
"""

JZAZBZ_XYZ_TO_LMS_MATRIX = np.array([
    [0.41478972, 0.579999, 0.0146480],
    [-0.2015100, 1.120649, 0.0531008],
    [-0.0166008, 0.264800, 0.6684799],
])
"""
:math:`J_zA_zB_z` *CIE XYZ* tristimulus values to normalised cone responses
matrix.

JZAZBZ_XYZ_TO_LMS_MATRIX : array_like, (3, 3)
"""

JZAZBZ_LMS_TO_XYZ_MATRIX = np.linalg.inv(JZAZBZ_XYZ_TO_LMS_MATRIX)
"""
:math:`J_zA_zB_z` normalised cone responses to *CIE XYZ* tristimulus values
matrix.

JZAZBZ_LMS_TO_XYZ_MATRIX : array_like, (3, 3)
"""

JZAZBZ_LMS_P_TO_IZAZBZ_MATRIX = np.array([
    [0.500000, 0.500000, 0.000000],
    [3.524000, -4.066708, 0.542708],
    [0.199076, 1.096799, -1.295875],
])
"""
:math:`LMS_p` *SMPTE ST 2084:2014* encoded normalised cone responses to
:math:`I_zA_zB_z` intermediate colourspace matrix.

JZAZBZ_LMS_P_TO_IZAZBZ_MATRIX : array_like, (3, 3)
"""

JZAZBZ_IZAZBZ_TO_LMS_P_MATRIX = np.linalg.inv(JZAZBZ_LMS_P_TO_IZAZBZ_MATRIX)
"""
:math:`I_zA_zB_z` intermediate colourspace to :math:`LMS_p`
*SMPTE ST 2084:2014* encoded normalised cone responses matrix.

JZAZBZ_IZAZBZ_TO_LMS_P_MATRIX : array_like, (3, 3)
"""


def XYZ_to_JzAzBz(XYZ_D65, constants=JZAZBZ_CONSTANTS):
    """
    Converts from *CIE XYZ* tristimulus values to :math:`J_zA_zB_z`
    colourspace.

    Parameters
    ----------
    XYZ_D65 : array_like
        *CIE XYZ* tristimulus values under
        *CIE Standard Illuminant D Series D65*.
    constants : Structure, optional
        :math:`J_zA_zB_z` colourspace constants.

    Returns
    -------
    ndarray
        :math:`J_zA_zB_z` colourspace array where :math:`J_z` is Lightness,
        :math:`A_z` is redness-greenness and :math:`B_z` is
        yellowness-blueness.

    References
    ----------
    -   :cite:`Safdar2017`

    Examples
    --------
    >>> XYZ = np.array([0.96907232, 1.00000000, 1.12179215])
    >>> XYZ_to_JzAzBz(XYZ)  # doctest: +ELLIPSIS
    array([ 0.0176682...,  0.0006417..., -0.0005290...])
    """

    X_D65, Y_D65, Z_D65 = tsplit(XYZ_D65)

    X_p_D65 = constants.b * X_D65 - (constants.b - 1) * Z_D65
    Y_p_D65 = constants.g * Y_D65 - (constants.g - 1) * X_D65

    XYZ_p_D65 = tstack([X_p_D65, Y_p_D65, Z_D65])

    LMS = dot_vector(JZAZBZ_XYZ_TO_LMS_MATRIX, XYZ_p_D65)

    LMS_p = oetf_ST2084(LMS, 10000, constants)

    I_z, A_z, B_z = tsplit(dot_vector(JZAZBZ_LMS_P_TO_IZAZBZ_MATRIX, LMS_p))

    J_z = ((1 + constants.d) * I_z) / (1 + constants.d * I_z) - constants.d_0

    return tstack([J_z, A_z, B_z])


def JzAzBz_to_XYZ(JzAzBz, constants=JZAZBZ_CONSTANTS):
    """
    Converts from :math:`J_zA_zB_z` colourspace to *CIE XYZ* tristimulus
    values.

    Parameters
    ----------
    JzAzBz : array_like
        :math:`J_zA_zB_z` colourspace array  where :math:`J_z` is Lightness,
        :math:`A_z` is redness-greenness and :math:`B_z` is
        yellowness-blueness.
    constants : Structure, optional
        :math:`J_zA_zB_z` colourspace constants.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values under
        *CIE Standard Illuminant D Series D65*.

    References
    ----------
    -   :cite:`Safdar2017`

    Examples
    --------
    >>> JzAzBz = np.array([0.01766826, 0.00064174, -0.00052906])
    >>> JzAzBz_to_XYZ(JzAzBz)  # doctest: +ELLIPSIS
    array([ 0.9690723...,  1.0000000...,  1.1217920...])
    """

    J_z, A_z, B_z = tsplit(JzAzBz)

    I_z = ((J_z + constants.d_0) / (1 + constants.d - constants.d *
                                    (J_z + constants.d_0)))
    LMS_p = dot_vector(JZAZBZ_IZAZBZ_TO_LMS_P_MATRIX, tstack([I_z, A_z, B_z]))

    LMS = eotf_ST2084(LMS_p, 10000, constants)

    X_p_D65, Y_p_D65, Z_p_D65 = tsplit(
        dot_vector(JZAZBZ_LMS_TO_XYZ_MATRIX, LMS))

    X_D65 = (X_p_D65 + (constants.b - 1) * Z_p_D65) / constants.b
    Y_D65 = (Y_p_D65 + (constants.g - 1) * X_D65) / constants.g

    return tstack([X_D65, Y_D65, Z_p_D65])
