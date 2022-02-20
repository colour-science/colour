"""Showcases spectral uniformity computations."""

import colour
from colour.quality.cfi2017 import load_TCS_CIE2017
from colour.quality.datasets import SDS_TCS
from colour.utilities import message_box

message_box("Spectral Uniformity (or Flatness) Computations")

message_box(
    'Computing the spectral uniformity of the "CRI" test colour samples.'
)

print(colour.spectral_uniformity(SDS_TCS.values()))

print("\n")

message_box(
    'Computing the spectral uniformity of the "CFI" test colour samples.'
)

print(
    colour.spectral_uniformity(load_TCS_CIE2017(colour.SPECTRAL_SHAPE_DEFAULT))
)
