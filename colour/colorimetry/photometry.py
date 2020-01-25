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
-   :cite:`Wikipedia2005c` : Wikipedia. (2005). Luminous Efficacy. Retrieved
    April 3, 2016, from https://en.wikipedia.org/wiki/Luminous_efficacy
-   :cite:`Wikipedia2003b` : Wikipedia. (2003). Luminosity function. Retrieved
    October 20, 2014, from https://en.wikipedia.org/wiki/\
Luminosity_function#Details
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import PHOTOPIC_LEFS
from colour.constants import K_M

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['luminous_flux', 'luminous_efficiency', 'luminous_efficacy']


def luminous_flux(sd,
                  lef=PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer'],
                  K_m=K_M):
    """
    Returns the *luminous flux* for given spectral distribution using given
    luminous efficiency function.

    Parameters
    ----------
    sd : SpectralDistribution
        test spectral distribution
    lef : SpectralDistribution, optional
        :math:`V(\\lambda)` luminous efficiency function.
    K_m : numeric, optional
        :math:`lm\\cdot W^{-1}` maximum photopic luminous efficiency

    Returns
    -------
    numeric
        Luminous flux.

    References
    ----------
    :cite:`Wikipedia2003b`

    Examples
    --------
    >>> from colour import LIGHT_SOURCES_SDS
    >>> sd = LIGHT_SOURCES_SDS['Neodimium Incandescent']
    >>> luminous_flux(sd)  # doctest: +ELLIPSIS
    23807.6555273...
    """

    lef = lef.copy().align(
        sd.shape,
        extrapolator_args={
            'method': 'Constant',
            'left': 0,
            'right': 0
        })
    sd = sd.copy() * lef

    flux = K_m * np.trapz(sd.values, sd.wavelengths)

    return flux


def luminous_efficiency(
        sd, lef=PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer']):
    """
    Returns the *luminous efficiency* of given spectral distribution using
    given luminous efficiency function.

    Parameters
    ----------
    sd : SpectralDistribution
        test spectral distribution
    lef : SpectralDistribution, optional
        :math:`V(\\lambda)` luminous efficiency function.

    Returns
    -------
    numeric
        Luminous efficiency.

    References
    ----------
    :cite:`Wikipedia2003b`

    Examples
    --------
    >>> from colour import LIGHT_SOURCES_SDS
    >>> sd = LIGHT_SOURCES_SDS['Neodimium Incandescent']
    >>> luminous_efficiency(sd)  # doctest: +ELLIPSIS
    0.1994393...
    """

    lef = lef.copy().align(
        sd.shape,
        extrapolator_args={
            'method': 'Constant',
            'left': 0,
            'right': 0
        })
    sd = sd.copy()

    efficiency = (np.trapz(lef.values * sd.values, sd.wavelengths) / np.trapz(
        sd.values, sd.wavelengths))

    return efficiency


def luminous_efficacy(
        sd, lef=PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer']):
    """
    Returns the *luminous efficacy* in :math:`lm\\cdot W^{-1}` of given
    spectral distribution using given luminous efficiency function.

    Parameters
    ----------
    sd : SpectralDistribution
        test spectral distribution
    lef : SpectralDistribution, optional
        :math:`V(\\lambda)` luminous efficiency function.

    Returns
    -------
    numeric
        Luminous efficacy in :math:`lm\\cdot W^{-1}`.

    References
    ----------
    :cite:`Wikipedia2005c`

    Examples
    --------
    >>> from colour import LIGHT_SOURCES_SDS
    >>> sd = LIGHT_SOURCES_SDS['Neodimium Incandescent']
    >>> luminous_efficacy(sd)  # doctest: +ELLIPSIS
    136.2170803...
    """

    efficacy = K_M * luminous_efficiency(sd, lef)

    return efficacy
