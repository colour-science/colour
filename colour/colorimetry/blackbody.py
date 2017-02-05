#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blackbody - Planckian Radiator
==============================

Defines objects to compute the spectral radiance of a planckian radiator and
its spectral power distribution.

See Also
--------
`Blackbody Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/blackbody.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import (
    DEFAULT_SPECTRAL_SHAPE,
    SpectralPowerDistribution)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['C1',
           'C2',
           'N',
           'planck_law',
           'blackbody_spectral_radiance',
           'blackbody_spd']

C1 = 3.741771e-16  # 2 * math.pi * PLANCK_CONSTANT * LIGHT_SPEED ** 2
C2 = 1.4388e-2  # PLANCK_CONSTANT * LIGHT_SPEED / BOLTZMANN_CONSTANT
N = 1


def planck_law(wavelength, temperature, c1=C1, c2=C2, n=N):
    """
    Returns the spectral radiance of a blackbody at thermodynamic temperature
    :math:`T[K]` in a medium having index of refraction :math:`n`.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength in meters.
    temperature : numeric or array_like
        Temperature :math:`T[K]` in kelvin degrees.
    c1 : numeric or array_like, optional
        The official value of :math:`c1` is provided by the Committee on Data
        for Science and Technology (CODATA) and is
        :math:`c1=3,741771x10.16\ W/m_2` *(Mohr and Taylor, 2000)*.
    c2 : numeric or array_like, optional
        Since :math:`T` is measured on the International Temperature Scale,
        the value of :math:`c2` used in colorimetry should follow that adopted
        in the current International Temperature Scale (ITS-90)
        *(Preston-Thomas, 1990; Mielenz et aI., 1991)*, namely
        :math:`c2=1,4388x10.2\ m/K`.
    n : numeric or array_like, optional
        Medium index of refraction. For dry air at 15°C and 101 325 Pa,
        containing 0,03 percent by volume of carbon dioxide, it is
        approximately 1,00028 throughout the visible region although
        *CIE 15:2004* recommends using :math:`n=1`.

    Returns
    -------
    numeric or ndarray
        Radiance in *watts per steradian per square metre*.

    Notes
    -----
    -   The following form implementation is expressed in term of wavelength.
    -   The SI unit of radiance is *watts per steradian per square metre*.

    References
    ----------
    .. [1]  CIE TC 1-48. (2004). APPENDIX E. INFORMATION ON THE USE OF
            PLANCK’S EQUATION FOR STANDARD AIR. In CIE 015:2004 Colorimetry,
            3rd Edition (pp. 77–82). ISBN:978-3-901-90633-6

    Examples
    --------
    >>> # Doctests ellipsis for Python 2.x compatibility.
    >>> planck_law(500 * 1e-9, 5500)  # doctest: +ELLIPSIS
    20472701909806.5...
    """

    l = np.asarray(wavelength)
    t = np.asarray(temperature)

    p = (((c1 * n ** -2 * l ** -5) / np.pi) *
         (np.exp(c2 / (n * l * t)) - 1) ** -1)

    return p


blackbody_spectral_radiance = planck_law


def blackbody_spd(temperature,
                  shape=DEFAULT_SPECTRAL_SHAPE,
                  c1=C1,
                  c2=C2,
                  n=N):
    """
    Returns the spectral power distribution of the planckian radiator for given
    temperature :math:`T[K]`.

    Parameters
    ----------
    temperature : numeric
        Temperature :math:`T[K]` in kelvin degrees.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution of the
        planckian radiator.
    c1 : numeric, optional
        The official value of :math:`c1` is provided by the Committee on Data
        for Science and Technology (CODATA) and is
        :math:`c1=3,741771x10.16\ W/m_2` *(Mohr and Taylor, 2000)*.
    c2 : numeric, optional
        Since :math:`T` is measured on the International Temperature Scale,
        the value of :math:`c2` used in colorimetry should follow that adopted
        in the current International Temperature Scale (ITS-90)
        *(Preston-Thomas, 1990; Mielenz et aI., 1991)*, namely
        :math:`c2=1,4388x10.2\ m/K`.
    n : numeric, optional
        Medium index of refraction. For dry air at 15°C and 101 325 Pa,
        containing 0,03 percent by volume of carbon dioxide, it is
        approximately 1,00028 throughout the visible region although
        *CIE 15:2004* recommends using :math:`n=1`.

    Returns
    -------
    SpectralPowerDistribution
        Blackbody spectral power distribution.

    Examples
    --------
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
    >>> print(blackbody_spd(5000, cmfs.shape))
    SpectralPowerDistribution('5000K Blackbody', (360.0, 830.0, 1.0))
    """

    wavelengths = shape.range()
    return SpectralPowerDistribution(
        name='{0}K Blackbody'.format(temperature),
        data=dict(
            zip(wavelengths,
                planck_law(
                    wavelengths * 1e-9, temperature, c1, c2, n))))
