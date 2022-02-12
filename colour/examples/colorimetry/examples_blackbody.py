"""Showcases blackbody / planckian radiator computations."""

import colour
from colour.utilities import message_box

message_box("Blackbody / Planckian Radiator Computations")

message_box(
    "Computing the spectral distribution of a blackbody at temperature 5500K"
    'degrees and converting to "CIE XYZ" tristimulus values.'
)
cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
sd_blackbody = colour.sd_blackbody(5500, cmfs.shape)
print(sd_blackbody)
XYZ = colour.sd_to_XYZ(sd_blackbody, cmfs)
print(XYZ)

print("\n")

message_box(
    "Computing the spectral radiance of a blackbody at wavelength 500nm and "
    "temperature 5500K degrees."
)
print(colour.colorimetry.blackbody_spectral_radiance(500 * 1e-9, 5500))
print(colour.colorimetry.planck_law(500 * 1e-9, 5500))
