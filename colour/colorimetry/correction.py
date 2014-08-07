# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectral Bandpass Dependence Correction
=======================================

Defines objects to perform spectral bandpass dependence correction.

The following correction methods are available:

-   :func:`bandpass_correction_stearns1988`: *Stearns and Stearns (1988)*
    spectral bandpass dependence correction method.
"""

from __future__ import unicode_literals

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["bandpass_correction_stearns1988",
           "BANDPASS_CORRECTION_METHODS",
           "bandpass_correction"]

ALPHA_STEARNS = 0.083


def bandpass_correction_stearns1988(spd):
    """
    Implements spectral bandpass dependence correction on given spectral power
    distribution using *Stearns and Stearns (1988)* method.

    References
    ----------
    .. [1]  **Stephen Westland, Caterina Ripamonti, Vien Cheung**,
            *Computational Colour Science Using MATLAB, 2nd Edition*,
            The Wiley-IS&T Series in Imaging Science and Technology,
            published July 2012, ISBN-13: 978-0-470-66569-5, Page 38.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Spectral bandpass dependence corrected spectral power distribution.

    Examples
    --------
    >>> spd = colour.SpectralPowerDistribution("Spd", {510: 49.6700, 520: 69.5900, 530: 81.7300, 540: 88.1900, 550: 86.0500})
    >>> corrected_spd = colour.bandpass_correction_stearns1988(spd)
    >>> print(corrected_spd.values)
    array([ 48.01664   ,  70.37296888,  82.13645358,  88.88480681,  85.87238   ])
    """

    values = spd.values
    values[0] = (1 + ALPHA_STEARNS) * values[0] - ALPHA_STEARNS * values[1]
    values[-1] = (1 + ALPHA_STEARNS) * values[-1] - ALPHA_STEARNS * values[-2]
    for i in range(1, len(values) - 1):
        values[i] = (-ALPHA_STEARNS * values[i - 1] +
                     (1. + 2. * ALPHA_STEARNS) *
                     values[i] - ALPHA_STEARNS * values[i + 1])

    for i, (wavelength, value) in enumerate(spd):
        spd[wavelength] = values[i]
    return spd


BANDPASS_CORRECTION_METHODS = {
    "Stearns 1988": bandpass_correction_stearns1988}
"""
Supported spectral bandpass dependence correction methods.

BANDPASS_CORRECTION_METHODS : dict
    ("Stearns 1988",)
"""


def bandpass_correction(spd, method="Stearns 1988"):
    """
    Implements spectral bandpass dependence correction on given spectral power
    distribution using given method.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    method : unicode
        ("Stearns 1988",)
        Correction method.

    Returns
    -------
    SpectralPowerDistribution
        Spectral bandpass dependence corrected spectral power distribution.
    """

    return BANDPASS_CORRECTION_METHODS.get(method)(spd)