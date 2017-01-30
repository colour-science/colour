#!/usr/bin/env python
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
.. [1]  Smits, B. (1999). An RGB-to-Spectrum Conversion for Reflectances.
        Journal of Graphics Tools, 4(4), 11–22.
        doi:10.1080/10867651.1999.10487511
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS, zeros_spd
from colour.models import (
    XYZ_to_RGB,
    normalised_primary_matrix,
    sRGB_COLOURSPACE)
from colour.recovery import SMITS_1999_SPDS

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['SMITS1999_PRIMARIES',
           'SMITS1999_WHITEPOINT',
           'SMITS1999_XYZ_TO_RGB_MATRIX',
           'XYZ_to_RGB_Smits1999',
           'RGB_to_spectral_Smits1999']

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


def XYZ_to_RGB_Smits1999(XYZ, chromatic_adaptation_transform='Bradford'):
    """
    Convenient object to convert from *CIE XYZ* tristimulus values to *RGB*
    colourspace in conditions required by the current *Smits (1999)* method
    implementation.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* method.

    Returns
    -------
    ndarray
        *RGB* colour array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_RGB_Smits1999(XYZ)  # doctest: +ELLIPSIS
    array([ 0.0214496...,  0.1315460...,  0.0928760...])
    """

    return XYZ_to_RGB(XYZ,
                      SMITS1999_WHITEPOINT,
                      SMITS1999_WHITEPOINT,
                      SMITS1999_XYZ_TO_RGB_MATRIX,
                      chromatic_adaptation_transform,
                      encoding_cctf=None)


def RGB_to_spectral_Smits1999(RGB):
    """
    Recovers the spectral power distribution of given *RGB* colourspace array
    using *Smits (1999)* method.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace array.

    Returns
    -------
    SpectralPowerDistribution
        Recovered spectral power distribution.

    Examples
    --------
    >>> RGB = np.array([0.02144962, 0.13154603, 0.09287601])
    >>> print(RGB_to_spectral_Smits1999(RGB))  # doctest: +ELLIPSIS
    SpectralPowerDistribution('0 Constant', (380.0, 720.0, 37.7777777...))
    """

    white_spd = SMITS_1999_SPDS['white'].clone()
    cyan_spd = SMITS_1999_SPDS['cyan'].clone()
    magenta_spd = SMITS_1999_SPDS['magenta'].clone()
    yellow_spd = SMITS_1999_SPDS['yellow'].clone()
    red_spd = SMITS_1999_SPDS['red'].clone()
    green_spd = SMITS_1999_SPDS['green'].clone()
    blue_spd = SMITS_1999_SPDS['blue'].clone()

    R, G, B = np.ravel(RGB)
    spd = zeros_spd(SMITS_1999_SPDS['white'].shape)

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
