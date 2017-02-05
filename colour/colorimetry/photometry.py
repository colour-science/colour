#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Photometry
==========

Defines photometric quantities computation related objects.

See Also
--------
`Photometry Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/photometry.ipynb>`_

References
----------
.. [1]  Wikipedia. (n.d.). Luminosity function. Retrieved October 20, 2014,
        from https://en.wikipedia.org/wiki/Luminosity_function#Details
.. [2]  Wikipedia. (n.d.). Luminous Efficacy. Retrieved April 3, 2016, from
        https://en.wikipedia.org/wiki/Luminous_efficacy
.. [3]  Ohno, Y., & Davis, W. (2008). NIST CQS simulation 7.4. Retrieved from
        http://cie2.nist.gov/TC1-69/NIST CQS simulation 7.4.xls
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import PHOTOPIC_LEFS
from colour.constants import K_M

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['luminous_flux',
           'luminous_efficiency',
           'luminous_efficacy']


def luminous_flux(
        spd,
        lef=PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer'],
        K_m=K_M):
    """
    Returns the *luminous flux* for given spectral power distribution using
    given luminous efficiency function.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        test spectral power distribution
    lef : SpectralPowerDistribution, optional
        :math:`V(\lambda)` luminous efficiency function.
    K_m : numeric, optional
        :math:`lm\cdot W^{-1}` maximum photopic luminous efficiency

    Returns
    -------
    numeric
        Luminous flux.

    Examples
    --------
    >>> from colour import LIGHT_SOURCES_RELATIVE_SPDS
    >>> spd = LIGHT_SOURCES_RELATIVE_SPDS['Neodimium Incandescent']
    >>> luminous_flux(spd)  # doctest: +ELLIPSIS
    23807.6555273...
    """

    lef = lef.clone().align(spd.shape,
                            extrapolation_left=0,
                            extrapolation_right=0)
    spd = spd.clone() * lef

    flux = K_m * np.trapz(spd.values, spd.wavelengths)

    return flux


def luminous_efficiency(
        spd,
        lef=PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer']):
    """
    Returns the *luminous efficiency* of given spectral power distribution
    using given luminous efficiency function.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        test spectral power distribution
    lef : SpectralPowerDistribution, optional
        :math:`V(\lambda)` luminous efficiency function.

    Returns
    -------
    numeric
        Luminous efficiency.

    Examples
    --------
    >>> from colour import LIGHT_SOURCES_RELATIVE_SPDS
    >>> spd = LIGHT_SOURCES_RELATIVE_SPDS['Neodimium Incandescent']
    >>> luminous_efficiency(spd)  # doctest: +ELLIPSIS
    0.1994393...
    """

    lef = lef.clone().align(spd.shape,
                            extrapolation_left=0,
                            extrapolation_right=0)
    spd = spd.clone()

    efficiency = (np.trapz(lef.values * spd.values, spd.wavelengths) /
                  np.trapz(spd.values, spd.wavelengths))

    return efficiency


def luminous_efficacy(
        spd,
        lef=PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer']):
    """
    Returns the *luminous efficacy* in :math:`lm\cdot W^{-1}` of given spectral
    power distribution using given luminous efficiency function.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        test spectral power distribution
    lef : SpectralPowerDistribution, optional
        :math:`V(\lambda)` luminous efficiency function.

    Returns
    -------
    numeric
        Luminous efficacy in :math:`lm\cdot W^{-1}`.

    Examples
    --------
    >>> from colour import LIGHT_SOURCES_RELATIVE_SPDS
    >>> spd = LIGHT_SOURCES_RELATIVE_SPDS['Neodimium Incandescent']
    >>> luminous_efficacy(spd)  # doctest: +ELLIPSIS
    136.2170803...
    """

    efficacy = K_M * luminous_efficiency(spd, lef)

    return efficacy
