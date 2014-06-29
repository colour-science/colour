# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**blackbody.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *blackbody* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import math
import numpy
import warnings

import colour.computation.spectrum
import colour.utilities.decorators
import colour.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LIGHT_SPEED_CONSTANT",
           "PLANCK_CONSTANT",
           "BOLTZMANN_CONSTANT",
           "E_CONSTANT",
           "C1_CONSTANT",
           "C2_CONSTANT",
           "N_CONSTANT",
           "planck_law",
           "blackbody_spectral_radiance",
           "blackbody_spectral_power_distribution"]

LOGGER = colour.utilities.verbose.install_logger()

LIGHT_SPEED_CONSTANT = 299792458
PLANCK_CONSTANT = 6.62607e-34
BOLTZMANN_CONSTANT = 1.38065e-23
E_CONSTANT = math.exp(1)
C1_CONSTANT = 3.741771e-16  # 2 * math.pi * PLANCK_CONSTANT * LIGHT_SPEED_CONSTANT ** 2
C2_CONSTANT = 1.4388e-2  # PLANCK_CONSTANT * LIGHT_SPEED_CONSTANT / BOLTZMANN_CONSTANT
N_CONSTANT = 1.


@colour.utilities.decorators.memoize(None)
def planck_law(wavelength, temperature, c1=C1_CONSTANT, c2=C2_CONSTANT, n=N_CONSTANT):
    """
    Returns the spectral radiance of a blackbody at thermodynamic temperature *T [K]* in a medium having index of refraction *n*.
    The following form implementation is expressed in term of wavelength. The SI unit of radiance is watts per steradian per square metre.

    References:

    -  `CIE 015:2004 Colorimetry, 3rd edition: Appendix E. Information on the Use of Planck's Equation for Standard Air. <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf>`_

    Usage::

        >>> planck_law(500 * 1e-9, 5500)
        5.50833496314e+13

    :param wavelength: Wavelength in meters.
    :type wavelength: float
    :param temperature: Temperature in kelvins.
    :type temperature: float
    :param c1: The official value of c1 is provided by the Committee on Data for Science and Technology (CODATA), and is c1 = 3,741771 x 10.16 W / m2 (Mohr and Taylor, 2000).
    :type c1: float
    :param c2: Since T is measured on the International Temperature Scale, the value of C2 used in colorimetry should follow that adopted in the current International Temperature Scale (ITS-90) (Preston-Thomas, 1990; Mielenz et aI., 1991), namely C2= 1,4388 x 10.2 m / K.
    :type c2: float
    :param n: Medium index of refraction. For dry air at 15°C and 101 325 Pa, containing 0,03 percent by volume of carbon dioxide, it is approximately 1,00028 throughout the visible region although CIE 15:2004 recommends using n = 1.
    :type n: float
    :return: Radiance.
    :rtype: float
    """

    t = temperature
    l = wavelength

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return ((c1 * n ** -2 * l ** -5) / math.pi) * (math.exp(c2 / (n * l * t)) - 1) ** -1
    except (OverflowError, RuntimeWarning) as error:
        return 0.0


blackbody_spectral_radiance = planck_law


def blackbody_spectral_power_distribution(temperature,
                                          start=None,
                                          end=None,
                                          steps=None,
                                          c1=C1_CONSTANT,
                                          c2=C2_CONSTANT,
                                          n=N_CONSTANT):
    """
    Returns the spectral power distribution of the *blackbody* for given temperature.

    Usage::

        >>> cmfs = colour.STANDARD_OBSERVERS_CMFS.get("CIE 1931 2 Degree Standard Observer")
        >>> blackbody_spectral_power_distribution(5000, *cmfs.shape)
        <colour.computation.spectrum.SpectralPowerDistribution at 0x10616fe90>

    :param temperature: Temperature in kelvins.
    :type temperature: float
    :param start: Wavelengths range start in nm.
    :type start: float
    :param end: Wavelengths range end in nm.
    :type end: float
    :param steps: Wavelengths range steps.
    :type steps: float
    :param c1: The official value of c1 is provided by the Committee on Data for Science and Technology (CODATA), and is c1 = 3,741771 x 10.16 W / m2 (Mohr and Taylor, 2000).
    :type c1: float
    :param c2: Since T is measured on the International Temperature Scale, the value of C2 used in colorimetry should follow that adopted in the current International Temperature Scale (ITS-90) (Preston-Thomas, 1990; Mielenz et aI., 1991), namely C2= 1,4388 x 10.2 m / K.
    :type c2: float
    :param n: Medium index of refraction. For dry air at 15°C and 101 325 Pa, containing 0,03 percent by volume of carbon dioxide, it is approximately 1,00028 throughout the visible region although CIE 15:2004 recommends using n = 1.
    :type n: float
    :return: Blackbody spectral power distribution.
    :rtype: SpectralPowerDistribution
    """

    return colour.computation.spectrum.SpectralPowerDistribution(name="{0}K Blackbody".format(temperature),
                                                            data=dict((wavelength,
                                                                      blackbody_spectral_radiance(wavelength * 1e-9,
                                                                                                  temperature,
                                                                                                  c1,
                                                                                                  c2,
                                                                                                  n))
                                                                     for wavelength in
                                                                     numpy.arange(start, end + steps, steps)))
