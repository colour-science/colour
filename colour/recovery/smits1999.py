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
from colour.recovery import SMITS_1999_SPDS
from colour.utilities import to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'SMITS1999_PRIMARIES', 'SMITS1999_WHITEPOINT',
    'SMITS1999_XYZ_TO_RGB_MATRIX', 'XYZ_to_RGB_Smits1999',
    'RGB_to_spectral_Smits1999'
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

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are normalised to domain [0, 1].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_RGB_Smits1999(XYZ)  # doctest: +ELLIPSIS
    array([ 0.0214496...,  0.1315460...,  0.0928760...])
    """

    return XYZ_to_RGB(
        XYZ,
        SMITS1999_WHITEPOINT,
        SMITS1999_WHITEPOINT,
        SMITS1999_XYZ_TO_RGB_MATRIX,
        encoding_cctf=None)


def RGB_to_spectral_Smits1999(RGB):
    """
    Recovers the spectral power distribution of given *RGB* colourspace array
    using *Smits (1999)* method.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace array to recover the spectral power distribution
        from.

    Returns
    -------
    SpectralPowerDistribution
        Recovered spectral power distribution.

    References
    ----------
    -   :cite:`Smits1999a`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> RGB = np.array([0.02144962, 0.13154603, 0.09287601])
    >>> with numpy_print_options(suppress=True):
    ...     RGB_to_spectral_Smits1999(RGB)  # doctest: +ELLIPSIS
    SpectralPowerDistribution([[ 380.        ,    0.0908046...],
                               [ 417.7778    ,    0.0887761...],
                               [ 455.5556    ,    0.0939795...],
                               [ 493.3333    ,    0.1236033...],
                               [ 531.1111    ,    0.1315788...],
                               [ 568.8889    ,    0.1293411...],
                               [ 606.6667    ,    0.0392680...],
                               [ 644.4444    ,    0.0214496...],
                               [ 682.2222    ,    0.0214496...],
                               [ 720.        ,    0.0215463...]],
                              interpolator=CubicSplineInterpolator,
                              interpolator_args={},
                              extrapolator=Extrapolator,
                              extrapolator_args={...})
    """

    white_spd = SMITS_1999_SPDS['white'].copy()
    cyan_spd = SMITS_1999_SPDS['cyan'].copy()
    magenta_spd = SMITS_1999_SPDS['magenta'].copy()
    yellow_spd = SMITS_1999_SPDS['yellow'].copy()
    red_spd = SMITS_1999_SPDS['red'].copy()
    green_spd = SMITS_1999_SPDS['green'].copy()
    blue_spd = SMITS_1999_SPDS['blue'].copy()

    R, G, B = to_domain_1(RGB)
    spd = white_spd.copy() * 0
    spd.name = 'Smits (1999) - {0}'.format(RGB)

    if R <= G and R <= B:
        spd += white_spd * R
        if G <= B:
            spd += cyan_spd * (G - R)
            spd += blue_spd * (B - G)
        else:
            spd += cyan_spd * (B - R)
            spd += green_spd * (G - B)
    elif G <= R and G <= B:
        spd += white_spd * G
        if R <= B:
            spd += magenta_spd * (R - G)
            spd += blue_spd * (B - R)
        else:
            spd += magenta_spd * (B - G)
            spd += red_spd * (R - B)
    else:
        spd += white_spd * B
        if R <= G:
            spd += yellow_spd * (R - B)
            spd += green_spd * (G - R)
        else:
            spd += yellow_spd * (G - B)
            spd += red_spd * (R - G)

    return spd
