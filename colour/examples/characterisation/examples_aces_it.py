"""
Showcases *Academy Color Encoding System* *Input Transform* related
computations.
"""

import os

import colour
from colour.utilities import message_box

message_box('"ACES" "Input Transform" Computations')

message_box(
    'Computing "ACES" relative exposure values for some colour rendition '
    "chart spectral distributions:\n\n"
    '\t("dark skin", "blue sky")'
)
print(
    colour.sd_to_aces_relative_exposure_values(
        colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"]
    )
)
print(
    colour.sd_to_aces_relative_exposure_values(
        colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]["blue sky"]
    )
)

print("\n")

message_box(
    'Computing "ACES" relative exposure values for various ideal '
    "reflectors:\n\n"
    '\t("18%", "100%")'
)
wavelengths = colour.characterisation.MSDS_ACES_RICD.wavelengths
gray_reflector = colour.SpectralDistribution(
    dict(zip(wavelengths, [0.18] * len(wavelengths))), name="18%"
)
print(repr(colour.sd_to_aces_relative_exposure_values(gray_reflector)))

perfect_reflector = colour.SpectralDistribution(
    dict(zip(wavelengths, [1.0] * len(wavelengths))), name="100%"
)
print(colour.sd_to_aces_relative_exposure_values(perfect_reflector))

print("\n")

message_box(
    'Computing an "ACES" input device transform for a "CANON EOS 5DMark II" '
    "and and *CIE Illuminant D Series* *D55*:"
)

path = os.path.join(
    colour.characterisation.aces_it.RESOURCES_DIRECTORY_RAWTOACES,
    "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
)
sensitivities = colour.colorimetry.sds_and_msds_to_msds(
    colour.io.read_sds_from_csv_file(path).values()
)
illuminant = colour.SDS_ILLUMINANTS["D55"]

print(colour.matrix_idt(sensitivities, illuminant))
