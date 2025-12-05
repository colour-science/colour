"""
Demonstrate spectral uniformity computations.

This module provides examples of spectral uniformity (or flatness)
calculations for test colour samples.
"""

import colour
from colour.quality.cfi2017 import load_TCS_CIE2017
from colour.quality.datasets import SDS_TCS
from colour.utilities import message_box

message_box("Spectral Uniformity (or Flatness) Computations")

message_box('Computing the spectral uniformity of the "CRI" test colour samples.')

print(colour.spectral_uniformity(list(SDS_TCS["CIE 1995"].values())))

print("\n")

message_box('Computing the spectral uniformity of the "CFI" test colour samples.')

print(colour.spectral_uniformity(load_TCS_CIE2017(colour.SPECTRAL_SHAPE_DEFAULT)))
