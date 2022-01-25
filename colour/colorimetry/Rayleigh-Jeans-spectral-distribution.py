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
"""
import colour
import numpy as np




__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2022 - Colour Developers'
__license__ = 'New BSD License - '
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'rayleigh_jeans_law',
    'sd_rayleigh_jeans'
]



def rayleigh_jeans_law(wavelength, temperature):
    λ = colour.utilities.as_float_array(wavelength)
    T = colour.utilities.as_float_array(temperature)

    c = colour.constants.CONSTANT_LIGHT_SPEED
    k_B = colour.constants.CONSTANT_BOLTZMANN

    B = (2 * c * k_B * T) / (λ ** 4)
    
    return B







def sd_rayleigh_jeans(temperature,shape=colour.SPECTRAL_SHAPE_DEFAULT):
    
    wavelengths = shape.range()
    
    return colour.SpectralDistribution(
        rayleigh_jeans_law(wavelengths * 1e-9, temperature) * 1e-9,
        wavelengths,
        name='{0}K Rayleigh-Jeans'.format(temperature))
