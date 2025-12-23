"""
Demonstrate reflectance recovery computations using *Gaussian* basis method.

This module provides examples of reflectance recovery from RGB colourspace
values using the *Gaussian* basis method with *Smits (1999)* decomposition.
"""

import numpy as np

import colour
from colour.recovery.gaussian import XYZ_to_RGB_Gaussian
from colour.utilities import message_box

message_box('"Gaussian" - Reflectance Recovery Computations')

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
RGB = XYZ_to_RGB_Gaussian(XYZ)
message_box(
    f'Recovering reflectance using "Gaussian" method from given "XYZ" '
    f"tristimulus values:\n\n\tXYZ: {XYZ}\n\tRGB: {RGB}"
)
sd = colour.recovery.RGB_to_sd_Gaussian(RGB)
print(sd)

print("\n")

message_box("Round-trip XYZ recovery:")
XYZ_recovered = colour.sd_to_XYZ(sd) / 100
print(f"Original XYZ:  {XYZ}")
print(f"Recovered XYZ: {XYZ_recovered}")
print(f"Delta: {XYZ_recovered - XYZ}")

print("\n")

message_box("Comparison with Smits (1999) method:")
sd_smits = colour.recovery.RGB_to_sd_Smits1999(
    colour.recovery.smits1999.XYZ_to_RGB_Smits1999(XYZ)
)
XYZ_smits = colour.sd_to_XYZ(sd_smits.align(colour.SPECTRAL_SHAPE_DEFAULT)) / 100
print(f"Gaussian error:    {np.sqrt(np.sum((XYZ_recovered - XYZ) ** 2)):.6f}")
print(f"Smits 1999 error:  {np.sqrt(np.sum((XYZ_smits - XYZ) ** 2)):.6f}")

print("\n")

message_box("Gaussian basis spectra parameters:")
print(f"Peak wavelengths: {colour.recovery.PEAK_WAVELENGTHS_GAUSSIAN_BASIS}")
print(f"FWHM: {colour.recovery.FWHM_GAUSSIAN_BASIS}")
