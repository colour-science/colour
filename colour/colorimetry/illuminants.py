#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Illuminants
===========

Defines *CIE* illuminants computation related objects.

See Also
--------
`Illuminants IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/colorimetry/illuminants.ipynb>`_
colour.colorimetry.dataset.illuminants.d_illuminants_s_spds,
colour.colorimetry.spectrum.SpectralPowerDistribution
"""

from __future__ import division, unicode_literals

from colour.colorimetry import (
    D_ILLUMINANTS_S_SPDS,
    SpectralPowerDistribution)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['D_illuminant_relative_spd']


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
    >>> import numpy as np
    >>> xy = np.array([0.34567, 0.35850])
    >>> D_illuminant_relative_spd(xy)  # doctest: +ELLIPSIS
    <colour.colorimetry.spectrum.SpectralPowerDistribution object at 0x...>
    """

    M = 0.0241 + 0.2562 * xy[0] - 0.7341 * xy[1]
    M1 = (-1.3515 - 1.7703 * xy[0] + 5.9114 * xy[1]) / M
    M2 = (0.0300 - 31.4424 * xy[0] + 30.0717 * xy[1]) / M

    distribution = {}
    for i in D_ILLUMINANTS_S_SPDS.get('S0').shape:
        S0 = D_ILLUMINANTS_S_SPDS.get('S0').get(i)
        S1 = D_ILLUMINANTS_S_SPDS.get('S1').get(i)
        S2 = D_ILLUMINANTS_S_SPDS.get('S2').get(i)
        distribution[i] = S0 + M1 * S1 + M2 * S2

    return SpectralPowerDistribution('CIE Standard Illuminant D Series',
                                     distribution)
