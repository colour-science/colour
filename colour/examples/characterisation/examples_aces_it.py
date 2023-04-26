"""
Showcases *Academy Color Encoding System* *Input Transform* related
computations.
"""

import os

import colour
import numpy as np
from colour.hints import cast
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
    colour.characterisation.aces_it.ROOT_RESOURCES_RAWTOACES,
    "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
)
sensitivities = colour.colorimetry.sds_and_msds_to_msds(
    list(colour.io.read_sds_from_csv_file(path).values())
)
illuminant = colour.SDS_ILLUMINANTS["D55"]

print(
    colour.matrix_idt(
        cast(colour.characterisation.RGB_CameraSensitivities, sensitivities),
        illuminant,
    )
)

message_box(
    'Optimising in "Oklab" colourspace using "Finlayson et al. (2015)" '
    "root-polynomials colour correction:"
)

M, RGB_w = colour.matrix_idt(  # pyright: ignore
    cast(colour.characterisation.RGB_CameraSensitivities, sensitivities),
    illuminant,
    optimisation_factory=colour.characterisation.optimisation_factory_Oklab_18,
)

print((M, RGB_w))

RGB = np.random.random((48, 3))

print(
    colour.utilities.vector_dot(  # pyright: ignore
        colour.models.RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ,
        np.transpose(
            np.dot(
                M,
                np.transpose(
                    colour.characterisation.polynomial_expansion_Finlayson2015(
                        RGB, 2, True
                    )
                ),
            )
        ),
    )
)
