#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectral Bandpass Dependence Correction
=======================================

Defines objects to perform spectral bandpass dependence correction.

The following correction methods are available:

-   :func:`bandpass_correction_Stearns1988`: *Stearns and Stearns (1988)⁠⁠*
    spectral bandpass dependence correction method.

See Also
--------
`Spectral Bandpass Dependence Correction Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/correction.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['bandpass_correction_Stearns1988',
           'BANDPASS_CORRECTION_METHODS',
           'bandpass_correction']

ALPHA_STEARNS = 0.083


def bandpass_correction_Stearns1988(spd):
    """
    Implements spectral bandpass dependence correction on given spectral power
    distribution using *Stearns and Stearns (1988)* method.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Spectral bandpass dependence corrected spectral power distribution.

    References
    ----------
    .. [1]  Westland, S., Ripamonti, C., & Cheung, V. (2012). Correction for
            Spectral Bandpass. In Computational Colour Science Using MATLAB
            (2nd ed., p. 38). ISBN:978-0-470-66569-5
    .. [2]  Stearns, E. I., & Stearns, R. E. (1988). An example of a method
            for correcting radiance data for Bandpass error. Color Research &
            Application, 13(4), 257–259. doi:10.1002/col.5080130410

    Examples
    --------
    >>> from colour import SpectralPowerDistribution
    >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    >>> spd = SpectralPowerDistribution('Spd', data)
    >>> corrected_spd = bandpass_correction_Stearns1988(spd)
    >>> corrected_spd.values  # doctest: +ELLIPSIS
    array([ 48.01664   ,  70.3729688...,  82.0919506...,  88.72618   ])
    """

    values = spd.values
    values[0] = (1 + ALPHA_STEARNS) * values[0] - ALPHA_STEARNS * values[1]
    values[-1] = (1 + ALPHA_STEARNS) * values[-1] - ALPHA_STEARNS * values[-2]
    for i in range(1, len(values) - 1):
        values[i] = (-ALPHA_STEARNS * values[i - 1] +
                     (1 + 2 * ALPHA_STEARNS) *
                     values[i] - ALPHA_STEARNS * values[i + 1])

    for i, (wavelength, _value) in enumerate(spd):
        spd[wavelength] = values[i]
    return spd


BANDPASS_CORRECTION_METHODS = CaseInsensitiveMapping(
    {'Stearns 1988': bandpass_correction_Stearns1988})
"""
Supported spectral bandpass dependence correction methods.

BANDPASS_CORRECTION_METHODS : CaseInsensitiveMapping
    **{'Stearns 1988', }**
"""


def bandpass_correction(spd, method='Stearns 1988'):
    """
    Implements spectral bandpass dependence correction on given spectral power
    distribution using given method.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    method : unicode, optional
        ('Stearns 1988', )
        Correction method.

    Returns
    -------
    SpectralPowerDistribution
        Spectral bandpass dependence corrected spectral power distribution.
    """

    return BANDPASS_CORRECTION_METHODS.get(method)(spd)
