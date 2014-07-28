# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**correction.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package spectral bandpass dependence correction related objects.

**Others:**

"""

from __future__ import unicode_literals

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["bandpass_correction_stearns",
           "bandpass_correction"]

ALPHA_STEARNS = 0.083


def bandpass_correction_stearns(spd):
    """
    Implements spectral bandpass dependence correction on given spectral power distribution \
    using *Stearns and Stearns (1988)* method.

    References:

    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, \
    2nd Edition*, Page 38.

    Usage::

        >>> spd = colour.SpectralPowerDistribution("Spd", {510: 49.6700, 520: 69.5900, 530: 81.7300, 540: 88.1900, 550: 86.0500})
        >>> corrected_spd = bandpass_correction_stearns(spd)
        >>> print(corrected_spd.values)
        [ 48.01664     70.37296888  82.13645358  88.88480681  85.87238   ]

    :param spd: Spectral power distribution.
    :type spd: SpectralPowerDistribution
    :return: Spectral bandpass dependence corrected spectral power distribution.
    :rtype: SpectralPowerDistribution
    """

    values = spd.values
    values[0] = (1 + ALPHA_STEARNS) * values[0] - ALPHA_STEARNS * values[1]
    values[-1] = (1 + ALPHA_STEARNS) * values[-1] - ALPHA_STEARNS * values[-2]
    for i in range(1, len(values) - 1):
        values[i] = -ALPHA_STEARNS * values[i - 1] + (1. + 2. * ALPHA_STEARNS) * values[i] - ALPHA_STEARNS * values[
            i + 1]

    for i, (wavelength, value) in enumerate(spd):
        spd[wavelength] = values[i]
    return spd


def bandpass_correction(spd, method="Stearns"):
    """
    Implements spectral bandpass dependence correction on given spectral power distribution using given method.

    :param spd: Spectral power distribution.
    :type spd: SpectralPowerDistribution
    :param method: Correction method.
    :type method: unicode ("Stearns",)
    :return: Spectral bandpass dependence corrected spectral power distribution.
    :rtype: SpectralPowerDistribution
    """

    if method == "Stearns":
        return bandpass_correction_stearns(spd)