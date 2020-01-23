# -*- coding: utf-8 -*-
"""
Smits (1999) - Reflectance Recovery
===================================

Defines objects for reflectance recovery using *Smits (1999)* method.

See Also
--------
`Smits (1999) - Reflectance Recovery Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/recovery/smits1999.ipynb>`_

References
----------
-   :cite:`Smits1999a` : Smits, B. (1999). An RGB-to-Spectrum Conversion for
    Reflectances. Journal of Graphics Tools, 4(4), 11-22.
    doi:10.1080/10867651.1999.10487511
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import (XYZ_to_RGB, normalised_primary_matrix,
                           sRGB_COLOURSPACE)
from colour.recovery import SMITS_1999_SDS
from colour.utilities import to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SMITS1999_PRIMARIES', 'SMITS1999_WHITEPOINT',
    'SMITS1999_XYZ_TO_RGB_MATRIX', 'XYZ_to_RGB_Smits1999',
    'RGB_to_sd_Smits1999'
]

SMITS1999_PRIMARIES = sRGB_COLOURSPACE.primaries
"""
Current *Smits (1999)* method implementation colourspace primaries.

SMITS1999_PRIMARIES : ndarray, (3, 2)
"""

SMITS1999_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E'])
"""
Current *Smits (1999)* method implementation colourspace whitepoint.

SMITS1999_WHITEPOINT : ndarray
"""

SMITS1999_XYZ_TO_RGB_MATRIX = np.linalg.inv(
    normalised_primary_matrix(SMITS1999_PRIMARIES, SMITS1999_WHITEPOINT))
"""
Current *Smits (1999)* method implementation *RGB* colourspace to
*CIE XYZ* tristimulus values matrix.

SMITS1999_XYZ_TO_RGB_MATRIX : array_like, (3, 3)
"""


def XYZ_to_RGB_Smits1999(XYZ):
    """
    Convenient object to convert from *CIE XYZ* tristimulus values to *RGB*
    colourspace in conditions required by the current *Smits (1999)* method
    implementation.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        *RGB* colour array.

    Examples
    --------
    >>> XYZ = np.array([0.21781186, 0.12541048, 0.04697113])
    >>> XYZ_to_RGB_Smits1999(XYZ)  # doctest: +ELLIPSIS
    array([ 0.4063959...,  0.0275289...,  0.0398219...])
    """

    return XYZ_to_RGB(
        XYZ,
        SMITS1999_WHITEPOINT,
        SMITS1999_WHITEPOINT,
        SMITS1999_XYZ_TO_RGB_MATRIX,
        cctf_encoding=None)


def RGB_to_sd_Smits1999(RGB):
    """
    Recovers the spectral distribution of given *RGB* colourspace array using
    *Smits (1999)* method.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace array to recover the spectral distribution from.

    Returns
    -------
    SpectralDistribution
        Recovered spectral distribution.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Smits1999a`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> RGB = np.array([0.40639599, 0.02752894, 0.03982193])
    >>> with numpy_print_options(suppress=True):
    ...     RGB_to_sd_Smits1999(RGB)  # doctest: +ELLIPSIS
    SpectralDistribution([[ 380.        ,    0.0769192...],
                          [ 417.7778    ,    0.0587004...],
                          [ 455.5556    ,    0.0394319...],
                          [ 493.3333    ,    0.0302497...],
                          [ 531.1111    ,    0.0275069...],
                          [ 568.8889    ,    0.0280864...],
                          [ 606.6667    ,    0.3429898...],
                          [ 644.4444    ,    0.4118579...],
                          [ 682.2222    ,    0.4118579...],
                          [ 720.        ,    0.4118075...]],
                         interpolator=LinearInterpolator,
                         interpolator_args={},
                         extrapolator=Extrapolator,
                         extrapolator_args={...})
    """

    white_sd = SMITS_1999_SDS['white'].copy()
    cyan_sd = SMITS_1999_SDS['cyan'].copy()
    magenta_sd = SMITS_1999_SDS['magenta'].copy()
    yellow_sd = SMITS_1999_SDS['yellow'].copy()
    red_sd = SMITS_1999_SDS['red'].copy()
    green_sd = SMITS_1999_SDS['green'].copy()
    blue_sd = SMITS_1999_SDS['blue'].copy()

    R, G, B = to_domain_1(RGB)
    sd = white_sd.copy() * 0
    sd.name = 'Smits (1999) - {0}'.format(RGB)

    if R <= G and R <= B:
        sd += white_sd * R
        if G <= B:
            sd += cyan_sd * (G - R)
            sd += blue_sd * (B - G)
        else:
            sd += cyan_sd * (B - R)
            sd += green_sd * (G - B)
    elif G <= R and G <= B:
        sd += white_sd * G
        if R <= B:
            sd += magenta_sd * (R - G)
            sd += blue_sd * (B - R)
        else:
            sd += magenta_sd * (B - G)
            sd += red_sd * (R - B)
    else:
        sd += white_sd * B
        if R <= G:
            sd += yellow_sd * (R - B)
            sd += green_sd * (G - R)
        else:
            sd += yellow_sd * (G - B)
            sd += red_sd * (R - G)

    return sd
