"""Showcases reflectance recovery computations using *Smits (1999)* method."""

import numpy as np

import colour
from colour.recovery.smits1999 import XYZ_to_RGB_Smits1999
from colour.utilities import message_box

message_box('"Smits (1999)" - Reflectance Recovery Computations')

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
RGB = XYZ_to_RGB_Smits1999(XYZ)
message_box(
    f'Recovering reflectance using "Smits (1999)" method from given "RGB" '
    f"colourspace array:\n\n\tRGB: {RGB}"
)
sd = colour.XYZ_to_sd(XYZ, method="Smits 1999")
print(sd)
print(colour.recovery.RGB_to_sd_Smits1999(XYZ))
print(colour.sd_to_XYZ(sd.align(colour.SPECTRAL_SHAPE_DEFAULT)) / 100)

print("\n")

message_box(
    'An analysis of "Smits (1999)" method is available at the '
    "following url : "
    "http://nbviewer.jupyter.org/github/colour-science/colour-website/"
    "blob/master/ipython/about_reflectance_recovery.ipynb"
)
