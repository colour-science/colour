#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**blackbody.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *blackbody* data and manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import math
import numpy
import warnings

import color.spectral
import color.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
		   "LIGHT_SPEED_CONSTANT",
		   "PLANCK_CONSTANT",
		   "BOLTZMANN_CONSTANT",
		   "E_CONSTANT",
		   "planck_law",
		   "blackbody_spectral_radiance",
		   "blackbody_spectral_power_distribution"]

LOGGER = color.verbose.install_logger()

LIGHT_SPEED_CONSTANT = 299792458
PLANCK_CONSTANT = 6.62607e-34
BOLTZMANN_CONSTANT = 1.38065e-23
E_CONSTANT = math.exp(1)

def planck_law(wavelength, temperature):
	"""
	Returns electromagnetic radiation emitted by a *blackbody* in thermal equilibrium at a definite temperature.
	The following form implementation is expressed in term of wavelength. The SI unit of radiance is watts per steradian per square metre.

	Reference: http://en.wikipedia.org/wiki/Planck's_law

	Usage::

		>>> planck_law(500 * 1e-9, 5500)
		5.50833496314e+13

	:param wavelength: Wavelength in meters.
	:type wavelength: float
	:param temperature: Temperature in kelvins.
	:type temperature: float
	:return: Radiance.
	:rtype: float
	"""

	t = temperature
	l = wavelength
	c = LIGHT_SPEED_CONSTANT
	h = PLANCK_CONSTANT
	k = BOLTZMANN_CONSTANT
	e = E_CONSTANT

	try:
		with warnings.catch_warnings():
			warnings.simplefilter("error")
			return ((2 * h * c ** 2) / l ** 5) * \
				   (1 / (e ** (((h * c) / (l * k * t))) - 1))
	except (OverflowError, RuntimeWarning) as error:
		return 0.0

blackbody_spectral_radiance = planck_law

def blackbody_spectral_power_distribution(temperature, start=None, end=None, steps=None):
	"""
	Returns the spectral power distribution of the *blackbody* for given temperature.

	:param temperature: Temperature in kelvins.
	:type temperature: float
	:param start: Wavelengths range start in nm.
	:type start: float
	:param end: Wavelengths range end in nm.
	:type end: float
	:param steps: Wavelengths range steps.
	:type steps: float
	:return: Blackbody spectral power distribution.
	:rtype: SpectralPowerDistribution
	"""

	return color.spectral.SpectralPowerDistribution(name="Blackbody",
													spd=dict(
														(wavelength,
														 blackbody_spectral_radiance(wavelength * 1e-9, temperature)) \
														for wavelength in numpy.arange(start, end + steps, steps)))
