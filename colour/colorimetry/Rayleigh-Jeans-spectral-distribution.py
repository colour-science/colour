# -*- coding: utf-8 -*-
"""
Rayleigh-Jeans spectral distribution
==============================

Defines the objects to compute the Rayleigh–Jeans law approximation to the spectral 
radiance of electromagnetic radiation as a function of wavelength from a black body at a 
given temperature through classical arguments.

References
----------
-  
-
-
 
"""

import colour
import numpy as np



__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2022 - Colour Developers'
__license__ = 'New BSD License - *--License here--*'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'



__all__ = [
    'rayleigh_jeans_law',
    'sd_rayleigh_jeans'
]


def rayleigh_jeans_law(wavelength, temperature):
    """
    
    Returns
    -------
    Spectral radiance, the power emitted per unit emitting area, 
    per steradian, per unit wavelength.
    
    
    Parameters
    ----------
    where "c" is the speed of light; 
    "k_B" is the Boltzmann constant;
    and "T" is the temperature in kelvins.
    
    Notes
    -----
    -   The following form implementation is expressed in term of wavelength.
    -   The SI unit of radiance is *watts per steradian per square metre*
        (:math:`W/sr/m^2`).
    
    
    """
    
    λ = colour.utilities.as_float_array(wavelength)
    T = colour.utilities.as_float_array(temperature)

    c = colour.constants.CONSTANT_LIGHT_SPEED
    k_B = colour.constants.CONSTANT_BOLTZMANN

    B = (2 * c * k_B * T) / (λ ** 4)
    
    return B







def sd_rayleigh_jeans(temperature,shape=colour.SPECTRAL_SHAPE_DEFAULT):
    
    """
    Returns the spectral distribution of the Spectral radiance for given
    temperature :math:`T[K]` with values in
    *watts per steradian per square metre per nanometer* (:math:`W/sr/m^2/nm`).

    Parameters
    ----------
    temperature : numeric
        Temperature :math:`T[K]` in kelvin degrees.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution of the
        planckian radiator.
    
    Returns
    -------
    SpectralDistribution
        Rayleigh-Jeans spectral distribution with values in
        *watts per steradian per square metre per nanometer*
        (:math:`W/sr/m^2/nm`).

    Examples
    --------
    
    
    
    --------
    """
    
    wavelengths = shape.range()
    
    return colour.SpectralDistribution(
        rayleigh_jeans_law(wavelengths * 1e-9, temperature) * 1e-9,
        wavelengths,
        name='{0}K Rayleigh-Jeans'.format(temperature))
