# -*- coding: utf-8 -*-
"""
Illuminants
===========

Defines *CIE* illuminants computation related objects:

-   :func:`colour.D_illuminant_relative_spd`
-   :func:`colour.CIE_standard_illuminant_A_function`

See Also
--------
`Illuminants Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/illuminants.ipynb>`_

References
----------
-   :cite:`CIETC1-482004n` : CIE TC 1-48. (2004). 3.1 Recommendations
    concerning standard physical data of illuminants. In CIE 015:2004
    Colorimetry, 3rd Edition (pp. 12-13). ISBN:978-3-901-90633-6
-   :cite:`Lindbloom2007a` : Lindbloom, B. (2007). Spectral Power Distribution
    of a CIE D-Illuminant. Retrieved April 5, 2014, from
    http://www.brucelindbloom.com/Eqn_DIlluminant.html
-   :cite:`Wyszecki2000z` : Wyszecki, G., & Stiles, W. S. (2000). CIE Method of
    Calculating D-Illuminants. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (pp. 145-146). Wiley. ISBN:978-0471399186
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import (D_ILLUMINANTS_S_SPDS,
                                SpectralPowerDistribution)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['D_illuminant_relative_spd', 'CIE_standard_illuminant_A_function']


def D_illuminant_relative_spd(xy):
    """
    Returns the relative spectral power distribution of given
    *CIE Standard Illuminant D Series* using given *xy* chromaticity
    coordinates.

    References
    ----------
    -   :cite:`Lindbloom2007a`
    -   :cite:`Wyszecki2000z`

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
    >>> from colour.utilities import numpy_print_options
    >>> xy = np.array([0.34570, 0.35850])
    >>> with numpy_print_options(suppress=True):
    ...     D_illuminant_relative_spd(xy)  # doctest: +ELLIPSIS
    SpectralPowerDistribution([[ 300.        ,    0.0193039...],
                               [ 310.        ,    2.1265303...],
                               [ 320.        ,    7.9867359...],
                               [ 330.        ,   15.1666959...],
                               [ 340.        ,   18.3413202...],
                               [ 350.        ,   21.3757973...],
                               [ 360.        ,   24.2528862...],
                               [ 370.        ,   26.2782171...],
                               [ 380.        ,   24.7348842...],
                               [ 390.        ,   30.0518667...],
                               [ 400.        ,   49.458942 ...],
                               [ 410.        ,   56.6929605...],
                               [ 420.        ,   60.1981682...],
                               [ 430.        ,   57.9390276...],
                               [ 440.        ,   74.9047554...],
                               [ 450.        ,   87.3151258...],
                               [ 460.        ,   90.6691236...],
                               [ 470.        ,   91.4109985...],
                               [ 480.        ,   95.1362798...],
                               [ 490.        ,   91.9956940...],
                               [ 500.        ,   95.7488852...],
                               [ 510.        ,   96.6315995...],
                               [ 520.        ,   97.1308377...],
                               [ 530.        ,  102.0961518...],
                               [ 540.        ,  100.7580555...],
                               [ 550.        ,  102.3164095...],
                               [ 560.        ,  100.       ...],
                               [ 570.        ,   97.7339937...],
                               [ 580.        ,   98.9175842...],
                               [ 590.        ,   93.5440898...],
                               [ 600.        ,   97.7548532...],
                               [ 610.        ,   99.3559831...],
                               [ 620.        ,   99.1396431...],
                               [ 630.        ,   95.8275899...],
                               [ 640.        ,   99.0028159...],
                               [ 650.        ,   95.8307955...],
                               [ 660.        ,   98.3850717...],
                               [ 670.        ,  103.2245516...],
                               [ 680.        ,   99.3672578...],
                               [ 690.        ,   87.5676019...],
                               [ 700.        ,   91.8218781...],
                               [ 710.        ,   93.0772354...],
                               [ 720.        ,   77.0098456...],
                               [ 730.        ,   86.6795856...],
                               [ 740.        ,   92.7570922...],
                               [ 750.        ,   78.3784557...],
                               [ 760.        ,   57.8075859...],
                               [ 770.        ,   83.0873522...],
                               [ 780.        ,   78.4245724...],
                               [ 790.        ,   79.7098456...],
                               [ 800.        ,   73.5435857...],
                               [ 810.        ,   64.0424558...],
                               [ 820.        ,   70.9121958...],
                               [ 830.        ,   74.5862223...]],
                              interpolator=SpragueInterpolator,
                              interpolator_args={},
                              extrapolator=Extrapolator,
                              extrapolator_args={...})
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
        distribution, name='CIE Standard Illuminant D Series')


def CIE_standard_illuminant_A_function(wl):
    """
    *CIE Standard Illuminant A* is intended to represent typical, domestic,
    tungsten-filament lighting.

    Its relative spectral power distribution is that of a Planckian radiator
    at a temperature of approximately 2856 K. *CIE Standard Illuminant A*
    should be used in all applications of colorimetry involving the use of
    incandescent lighting, unless there are specific reasons for using
    a different illuminant.

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
    -   :cite:`CIETC1-482004n`

    Examples
    --------
    >>> wl = np.array([560, 580, 581.5])
    >>> CIE_standard_illuminant_A_function(wl)  # doctest: +ELLIPSIS
    array([ 100.        ,  114.4363383...,  115.5285063...])
    """

    wl = np.asarray(wl)

    return (100 * (560 / wl) ** 5 * (((np.exp(
        (1.435 * 10 ** 7) / (2848 * 560)) - 1) / (np.exp(
            (1.435 * 10 ** 7) / (2848 * wl)) - 1))))
