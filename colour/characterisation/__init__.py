import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from colour.hints import Any

from .cameras import RGB_CameraSensitivities
from .displays import RGB_DisplayPrimaries
from .datasets import *  # noqa
from . import datasets
from .aces_it import (
    sd_to_aces_relative_exposure_values,
    sd_to_ACES2065_1,
    read_training_data_rawtoaces_v1,
    generate_illuminants_rawtoaces_v1,
    white_balance_multipliers,
    best_illuminant,
    normalise_illuminant,
    training_data_sds_to_RGB,
    training_data_sds_to_XYZ,
    optimisation_factory_rawtoaces_v1,
    optimisation_factory_Jzazbz,
    matrix_idt,
    camera_RGB_to_ACES2065_1,
)
from .correction import (
    matrix_augmented_Cheung2004,
    polynomial_expansion_Finlayson2015,
    polynomial_expansion_Vandermonde,
    POLYNOMIAL_EXPANSION_METHODS,
    polynomial_expansion,
    matrix_colour_correction_Cheung2004,
    matrix_colour_correction_Finlayson2015,
    matrix_colour_correction_Vandermonde,
    MATRIX_COLOUR_CORRECTION_METHODS,
    matrix_colour_correction,
    colour_correction_Cheung2004,
    colour_correction_Finlayson2015,
    colour_correction_Vandermonde,
    COLOUR_CORRECTION_METHODS,
    colour_correction,
)

__all__ = [
    "RGB_CameraSensitivities",
]
__all__ += [
    "RGB_DisplayPrimaries",
]
__all__ += datasets.__all__
__all__ += [
    "sd_to_aces_relative_exposure_values",
    "sd_to_ACES2065_1",
    "read_training_data_rawtoaces_v1",
    "generate_illuminants_rawtoaces_v1",
    "white_balance_multipliers",
    "best_illuminant",
    "normalise_illuminant",
    "training_data_sds_to_RGB",
    "training_data_sds_to_XYZ",
    "optimisation_factory_rawtoaces_v1",
    "optimisation_factory_Jzazbz",
    "matrix_idt",
    "camera_RGB_to_ACES2065_1",
]
__all__ += [
    "matrix_augmented_Cheung2004",
    "polynomial_expansion_Finlayson2015",
    "polynomial_expansion_Vandermonde",
    "POLYNOMIAL_EXPANSION_METHODS",
    "polynomial_expansion",
    "matrix_colour_correction_Cheung2004",
    "matrix_colour_correction_Finlayson2015",
    "matrix_colour_correction_Vandermonde",
    "MATRIX_COLOUR_CORRECTION_METHODS",
    "matrix_colour_correction",
    "colour_correction_Cheung2004",
    "colour_correction_Finlayson2015",
    "colour_correction_Vandermonde",
    "COLOUR_CORRECTION_METHODS",
    "colour_correction",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class characterisation(ModuleAPI):
    """Define a class acting like the *characterisation* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


# v0.4.0
API_CHANGES = {
    "ObjectRenamed": [
        [
            "colour.characterisation.optimisation_factory_JzAzBz",
            "colour.characterisation.optimisation_factory_Jzazbz",
        ],
    ]
}
"""Defines the *colour.characterisation* sub-package API changes."""

if not is_documentation_building():
    sys.modules[
        "colour.characterisation"
    ] = characterisation(  # type: ignore[assignment]
        sys.modules["colour.characterisation"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
