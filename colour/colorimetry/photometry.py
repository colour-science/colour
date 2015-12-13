#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Photometry
==========

Defines photometric quantities computation related objects.

See Also
--------
`Photometry IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/colorimetry/photometry.ipynb>`_

References
----------
.. [1]  Wikipedia. (n.d.). Luminosity function. Retrieved October 20, 2014,
        from https://en.wikipedia.org/wiki/Luminosity_function#Details

.. [2]  Ohno, Y., & Davis, W. (2008). NIST CQS simulation 7.4. Retrieved from
        http://cie2.nist.gov/TC1-69/NIST CQS simulation 7.4.xls
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import PHOTOPIC_LEFS
from colour.constants import K_M

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['luminous_flux',
           'luminous_efficacy']


def luminous_flux(spd,
                  lef=PHOTOPIC_LEFS.get(
                      'CIE 1924 Photopic Standard Observer'),
                  K_m=K_M):
    """
    Returns the *luminous flux* for given spectral power distribution using
    the given luminous efficiency function.

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
        Luminous flux

    Examples
    --------
    >>> from colour import LIGHT_SOURCES_RELATIVE_SPDS
    >>> spd = LIGHT_SOURCES_RELATIVE_SPDS.get('Neodimium Incandescent')
    >>> luminous_flux(spd)  # doctest: +ELLIPSIS
    23807.6555273...
    """

    lef = lef.clone().align(spd.shape, left=0, right=0)
    spd = spd.clone() * lef

    flux = K_m * np.trapz(spd.values, spd.wavelengths)

    return flux


def luminous_efficacy(spd,
                      lef=PHOTOPIC_LEFS.get(
                          'CIE 1924 Photopic Standard Observer')):
    """
    Returns the *luminous efficacy* for given spectral power distribution using
    the given luminous efficiency function.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        test spectral power distribution
    lef : SpectralPowerDistribution, optional
        :math:`V(\lambda)` luminous efficiency function.

    Returns
    -------
    numeric
        Luminous efficacy

    Examples
    --------
    >>> from colour import LIGHT_SOURCES_RELATIVE_SPDS
    >>> spd = LIGHT_SOURCES_RELATIVE_SPDS.get('Neodimium Incandescent')
    >>> luminous_efficacy(spd)  # doctest: +ELLIPSIS
    0.1994393...
    """

    lef = lef.clone().align(spd.shape, left=0, right=0)
    spd = spd.clone()

    efficacy = (np.trapz(lef.values * spd.values, spd.wavelengths) /
                np.trapz(spd.values, spd.wavelengths))

    return efficacy
