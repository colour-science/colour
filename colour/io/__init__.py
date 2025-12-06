import sys

from colour.hints import Any
from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from . import luts
from .image import (
    MAPPING_BIT_DEPTH,
    READ_IMAGE_METHODS,
    WRITE_IMAGE_METHODS,
    Image_Specification_Attribute,
    as_3_channels_image,
    convert_bit_depth,
    image_specification_OpenImageIO,
    read_image,
    read_image_Imageio,
    read_image_OpenImageIO,
    write_image,
    write_image_Imageio,
    write_image_OpenImageIO,
)
from .luts import *  # noqa: F403

# isort: split

from .ctl import (
    ctl_render,
    process_image_ctl,
    template_ctl_transform_float,
    template_ctl_transform_float3,
)
from .fichet2021 import (
    ComponentsFichet2021,
    Specification_Fichet2021,
    read_spectral_image_Fichet2021,
    sd_to_spectrum_attribute_Fichet2021,
    spectrum_attribute_to_sd_Fichet2021,
    write_spectral_image_Fichet2021,
)
from .ocio import process_image_OpenColorIO
from .tabular import (
    read_sds_from_csv_file,
    read_spectral_data_from_csv_file,
    write_sds_to_csv_file,
)
from .tm2714 import Header_IESTM2714, SpectralDistribution_IESTM2714
from .uprtek_sekonic import (
    SpectralDistribution_Sekonic,
    SpectralDistribution_UPRTek,
)
from .xrite import read_sds_from_xrite_file

__all__ = luts.__all__
__all__ += [
    "MAPPING_BIT_DEPTH",
    "READ_IMAGE_METHODS",
    "WRITE_IMAGE_METHODS",
    "Image_Specification_Attribute",
    "as_3_channels_image",
    "convert_bit_depth",
    "image_specification_OpenImageIO",
    "read_image",
    "read_image_Imageio",
    "read_image_OpenImageIO",
    "write_image",
    "write_image_Imageio",
    "write_image_OpenImageIO",
]
__all__ += [
    "ctl_render",
    "process_image_ctl",
    "template_ctl_transform_float",
    "template_ctl_transform_float3",
]
__all__ += [
    "ComponentsFichet2021",
    "Specification_Fichet2021",
    "read_spectral_image_Fichet2021",
    "sd_to_spectrum_attribute_Fichet2021",
    "spectrum_attribute_to_sd_Fichet2021",
    "write_spectral_image_Fichet2021",
]
__all__ += [
    "process_image_OpenColorIO",
]
__all__ += [
    "read_sds_from_csv_file",
    "read_spectral_data_from_csv_file",
    "write_sds_to_csv_file",
]
__all__ += [
    "Header_IESTM2714",
    "SpectralDistribution_IESTM2714",
]
__all__ += [
    "SpectralDistribution_Sekonic",
    "SpectralDistribution_UPRTek",
]
__all__ += [
    "read_sds_from_xrite_file",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class io(ModuleAPI):
    """Define a class acting like the *io* module."""

    def __getattr__(self, attribute: str) -> Any:
        """Return the value from the attribute with the specified name."""

        return super().__getattr__(attribute)


# v0.4.5
API_CHANGES = {
    "ObjectRenamed": [
        [
            "colour.io.ImageAttribute_Specification",
            "colour.io.Image_Specification_Attribute",
        ],
    ]
}

"""Defines the *colour.io* sub-package API changes."""

if not is_documentation_building():
    sys.modules["colour.io"] = io(  # pyright: ignore
        sys.modules["colour.io"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
