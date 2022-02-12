"""Showcases *Rayleigh Optical Depth* computations examples."""

import colour
from colour.utilities import message_box

message_box('"Rayleigh" Optical Depth Computations')

message_box(
    f'Creating a "Rayleigh" spectral distribution with default spectral '
    f"shape:\n\n\t{colour.SPECTRAL_SHAPE_DEFAULT}"
)
sd_rayleigh = colour.sd_rayleigh_scattering()
print(sd_rayleigh[555])

print("\n")

wavelength = 555 * 10e-8
message_box(
    f"Computing the scattering cross section per molecule at given wavelength "
    f"in cm:\n\n\tWavelength: {wavelength}cm"
)
print(colour.phenomena.scattering_cross_section(wavelength))

print("\n")

message_box(
    f'Computing the "Rayleigh" optical depth as function of wavelength in '
    f"cm:\n\n\tWavelength: {wavelength}cm"
)
print(colour.phenomena.rayleigh_optical_depth(wavelength))
