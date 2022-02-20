"""
Showcases reflectance recovery computations using *Mallett et al. (2019)*
method.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Mallett et al. (2019)" - Reflectance Recovery Computations')

illuminant = colour.SDS_ILLUMINANTS["D65"]

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
RGB = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)
message_box(
    f'Recovering reflectance using "Mallett et al. (2019)" method from given '
    f'"XYZ" tristimulus values:\n\n\tXYZ: {XYZ}'
)
sd = colour.XYZ_to_sd(XYZ, method="Mallett 2019")
print(sd)
print(colour.recovery.RGB_to_sd_Mallett2019(RGB))
print(colour.sd_to_XYZ(sd, illuminant=illuminant) / 100)

print("\n")

message_box(
    'Generating the "Mallett et al. (2019)" basis functions for the '
    "*Pal/Secam* colourspace:"
)
cmfs = (
    colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    .copy()
    .align(colour.SpectralShape(360, 780, 10))
)
illuminant = colour.SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)

print(
    colour.recovery.spectral_primary_decomposition_Mallett2019(
        colour.models.RGB_COLOURSPACE_PAL_SECAM,
        cmfs,
        illuminant,
        optimisation_kwargs={"options": {"ftol": 1e-5}},
    )
)
