#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Illuminants
===========

Defines *CIE* illuminants computation related objects:

-   :func:`D_illuminant_relative_spd`
-   :func:`CIE_standard_illuminant_A_function`

See Also
--------
`Illuminants Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/illuminants.ipynb>`_
colour.colorimetry.dataset.illuminants.d_illuminants_s_spds,
colour.colorimetry.spectrum.SpectralPowerDistribution
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import (
    D_ILLUMINANTS_S_SPDS,
    SpectralPowerDistribution)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['D_illuminant_relative_spd',
           'CIE_standard_illuminant_A_function']


def D_illuminant_relative_spd(xy):
    """
    Returns the relative spectral power distribution of given
    *CIE Standard Illuminant D Series* using given *xy* chromaticity
    coordinates.

    References
    ----------
    .. [1]  Wyszecki, G., & Stiles, W. S. (2000). CIE Method of Calculating
            D-Illuminants. In Color Science: Concepts and Methods,
            Quantitative Data and Formulae (pp. 145–146). Wiley.
            ISBN:978-0471399186
    .. [2]  Lindbloom, B. (2007). Spectral Power Distribution of a
            CIE D-Illuminant. Retrieved April 05, 2014, from
            http://www.brucelindbloom.com/Eqn_DIlluminant.html

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.

    Returns
    -------
    SpectralPowerDistribution
        *CIE Standard Illuminant D Series* relative spectral power
        distribution.

    Examples
    --------
    >>> xy = np.array([0.34570, 0.35850])
    >>> print(D_illuminant_relative_spd(xy))
    SpectralPowerDistribution(\
'CIE Standard Illuminant D Series', (300.0, 830.0, 10.0))
    """

    M = 0.0241 + 0.2562 * xy[0] - 0.7341 * xy[1]
    M1 = (-1.3515 - 1.7703 * xy[0] + 5.9114 * xy[1]) / M
    M2 = (0.0300 - 31.4424 * xy[0] + 30.0717 * xy[1]) / M

    distribution = {}
    for i in D_ILLUMINANTS_S_SPDS['S0'].shape:
        S0 = D_ILLUMINANTS_S_SPDS['S0'][i]
        S1 = D_ILLUMINANTS_S_SPDS['S1'][i]
        S2 = D_ILLUMINANTS_S_SPDS['S2'][i]
        distribution[i] = S0 + M1 * S1 + M2 * S2

    return SpectralPowerDistribution(
        'CIE Standard Illuminant D Series', distribution)


def CIE_standard_illuminant_A_function(wl):
    """
    *CIE Standard Illuminant A* is intended to represent typical, domestic,
    tungsten-filament lighting. Its relative spectral power distribution is
    that of a Planckian radiator at a temperature of approximately 2856 K.
    CIE Standard Illuminant A should be used in all applications of
    colorimetry involving the use of incandescent lighting, unless there are
    specific reasons for using a different illuminant.

    Parameters
    ----------
    wl : array_like
        Wavelength to evaluate the function at.

    Returns
    -------
    ndarray
        *CIE Standard Illuminant A* value at given wavelength.

    References
    ----------
    .. [1]  CIE TC 1-48. (2004). 3.1 Recommendations concerning standard
            physical data of illuminants. In CIE 015:2004 Colorimetry, 3rd
            Edition (pp. 12–13). ISBN:978-3-901-90633-6

    Examples
    --------
    >>> wl = np.array([560, 580, 581.5])
    >>> CIE_standard_illuminant_A_function(wl)  # doctest: +ELLIPSIS
    array([ 100.        ,  114.4363383...,  115.5285063...])
    """

    wl = np.asarray(wl)

    return (100 * (560 / wl) ** 5 * (
        ((np.exp((1.435 * 10 ** 7) / (2848 * 560)) - 1) /
         (np.exp((1.435 * 10 ** 7) / (2848 * wl)) - 1))))
