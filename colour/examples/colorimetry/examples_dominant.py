"""Showcases dominant wavelength and purity of a colour computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box("Dominant Wavelength and Purity")

xy = np.array([0.54369557, 0.32107944])
xy_n = np.array([0.31270, 0.32900])
cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
message_box(
    f'Computing the "dominant wavelength" for colour stimulus "xy" and '
    f'achromatic stimulus "xy_n" chromaticity coordinates:\n\n'
    f"\txy: {xy}\n"
    f"\txy_n: {xy_n}"
)

print(colour.dominant_wavelength(xy, xy_n, cmfs))

print("\n")

xy = np.array([0.35000, 0.25000])
message_box(
    f'Computing the "dominant wavelength" for colour stimulus "xy" and '
    f'achromatic stimulus "xy_n" chromaticity coordinates:\n\n'
    f"\txy: {xy}\n"
    f"\txy_n: {xy_n}\n\n"
    f'In this case the "complementary dominant wavelength" indicated by a '
    f"negative sign is returned because the first intersection is located on "
    f"the line of purples."
)

print(colour.dominant_wavelength(xy, xy_n, cmfs))

print("\n")

xy = np.array([0.54369557, 0.32107944])
message_box(
    f'Computing the "complementary wavelength" for colour stimulus "xy" and '
    f'achromatic stimulus "xy_n" chromaticity coordinates:\n\n'
    f"\txy: {xy}\n"
    f"\txy_n: {xy_n}\n\n"
    f'In this case the "dominant wavelength" indicated by a negative sign is '
    f"returned because the first intersection is located on the line of purples."
)

print(colour.complementary_wavelength(xy, xy_n, cmfs))

print("\n")

xy = np.array([0.35000, 0.25000])
message_box(
    f'Computing the "complementary wavelength" for colour stimulus "xy" and '
    f'achromatic stimulus "xy_n" chromaticity coordinates:\n\n'
    f"\txy: {xy}\n"
    f"\txy_n: {xy_n}"
)

print(colour.complementary_wavelength(xy, xy_n, cmfs))

print("\n")

xy = np.array([0.54369557, 0.32107944])
message_box(
    f'Computing the "excitation purity" for colour stimulus "xy" and '
    f'achromatic stimulus "xy_n" chromaticity coordinates:\n\n'
    f"\txy: {xy}\n"
    f"\txy_n: {xy_n}"
)

print(colour.excitation_purity(xy, xy_n, cmfs))

print("\n")

xy = np.array([0.54369557, 0.32107944])
message_box(
    f'Computing the "colorimetric purity" for colour stimulus "xy" and '
    f'achromatic stimulus "xy_n" chromaticity coordinates:\n\n'
    f"\txy: {xy}\n"
    f"\txy_n: {xy_n}"
)

print(colour.colorimetric_purity(xy, xy_n, cmfs))
