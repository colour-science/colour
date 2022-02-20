"""Showcases reflectance recovery computations using *Meng et al. (2015)* method."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Meng et al. (2015)" - Reflectance Recovery Computations')

illuminant = colour.SDS_ILLUMINANTS["D65"]

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box(
    f'Recovering reflectance using "Meng et al. (2015)" method from given '
    f'"XYZ" tristimulus values:\n\n\tXYZ: {XYZ}'
)
sd = colour.XYZ_to_sd(XYZ, method="Meng 2015")
print(sd)
print(colour.recovery.XYZ_to_sd_Meng2015(XYZ))
print(colour.sd_to_XYZ(sd, illuminant=illuminant) / 100)
