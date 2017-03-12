#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Luminous Efficiency Functions Spectral Power Distributions
==========================================================

Defines luminous efficiency functions computation related objects.

See Also
--------
`Luminous Efficiency Functions Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/lefs.ipynb>`_
colour.colorimetry.dataset.lefs,
colour.colorimetry.spectrum.SpectralPowerDistribution

References
----------
.. [1]  Wikipedia. (n.d.). Mesopic weighting function. Retrieved June 20,
        2014, from http://en.wikipedia.org/wiki/\
Mesopic_vision#Mesopic_weighting_function
"""

from __future__ import division, unicode_literals

from colour.colorimetry import (
    PHOTOPIC_LEFS,
    SCOTOPIC_LEFS,
    SpectralPowerDistribution,
    SpectralShape)
from colour.colorimetry.dataset.lefs import MESOPIC_X_DATA
from colour.utilities import closest

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['mesopic_weighting_function',
           'mesopic_luminous_efficiency_function']


def mesopic_weighting_function(
        wavelength,
        Lp,
        source='Blue Heavy',
        method='MOVE',
        photopic_lef=PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer'],
        scotopic_lef=SCOTOPIC_LEFS['CIE 1951 Scotopic Standard Observer']):
    """
    Calculates the mesopic weighting function factor at given wavelength
    :math:`\lambda` using the photopic luminance :math:`L_p`.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` to calculate the mesopic weighting function
        factor.
    Lp : numeric
        Photopic luminance :math:`L_p`.
    source : unicode, optional
        **{'Blue Heavy', 'Red Heavy'}**,
        Light source colour temperature.
    method : unicode, optional
        **{'MOVE', 'LRC'}**,
        Method to calculate the weighting factor.
    photopic_lef : SpectralPowerDistribution, optional
        :math:`V(\lambda)` photopic luminous efficiency function.
    scotopic_lef : SpectralPowerDistribution, optional
        :math:`V^\prime(\lambda)` scotopic luminous efficiency function.

    Returns
    -------
    numeric or ndarray
        Mesopic weighting function factor.

    Examples
    --------
    >>> mesopic_weighting_function(500, 0.2)  # doctest: +ELLIPSIS
    0.7052200...
    """

    mesopic_x_luminance_values = sorted(MESOPIC_X_DATA.keys())
    index = mesopic_x_luminance_values.index(
        closest(mesopic_x_luminance_values, Lp))
    x = MESOPIC_X_DATA[mesopic_x_luminance_values[index]][source][method]

    Vm = ((1 - x) *
          scotopic_lef.get(wavelength) + x * photopic_lef.get(wavelength))

    return Vm


def mesopic_luminous_efficiency_function(
        Lp,
        source='Blue Heavy',
        method='MOVE',
        photopic_lef=PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer'],
        scotopic_lef=SCOTOPIC_LEFS['CIE 1951 Scotopic Standard Observer']):
    """
    Returns the mesopic luminous efficiency function :math:`V_m(\lambda)` for
    given photopic luminance :math:`L_p`.

    Parameters
    ----------
    Lp : numeric
        Photopic luminance :math:`L_p`.
    source : unicode, optional
        **{'Blue Heavy', 'Red Heavy'}**,
        Light source colour temperature.
    method : unicode, optional
        **{'MOVE', 'LRC'}**,
        Method to calculate the weighting factor.
    photopic_lef : SpectralPowerDistribution, optional
        :math:`V(\lambda)` photopic luminous efficiency function.
    scotopic_lef : SpectralPowerDistribution, optional
        :math:`V^\prime(\lambda)` scotopic luminous efficiency function.

    Returns
    -------
    SpectralPowerDistribution
        Mesopic luminous efficiency function :math:`V_m(\lambda)`.

    Examples
    --------
    >>> print(mesopic_luminous_efficiency_function(0.2))
    SpectralPowerDistribution(\
'0.2 Lp Mesopic Luminous Efficiency Function', (380.0, 780.0, 1.0))
    """

    photopic_lef_shape = photopic_lef.shape
    scotopic_lef_shape = scotopic_lef.shape
    shape = SpectralShape(
        max(photopic_lef_shape.start, scotopic_lef_shape.start),
        min(photopic_lef_shape.end, scotopic_lef_shape.end),
        max(photopic_lef_shape.interval, scotopic_lef_shape.interval))

    wavelengths = shape.range()

    spd_data = dict(zip(wavelengths,
                        mesopic_weighting_function(
                            wavelengths,
                            Lp,
                            source,
                            method,
                            photopic_lef,
                            scotopic_lef)))

    spd = SpectralPowerDistribution(
        '{0} Lp Mesopic Luminous Efficiency Function'.format(Lp),
        spd_data)

    return spd.normalise()
