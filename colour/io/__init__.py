import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from colour.hints import Any

from .luts import *  # noqa: F403
from . import luts
from .image import (
    Image_Specification_Attribute,
    image_specification_OpenImageIO,
    convert_bit_depth,
)
from .image import read_image_OpenImageIO, write_image_OpenImageIO
from .image import read_image_Imageio, write_image_Imageio
from .image import READ_IMAGE_METHODS, WRITE_IMAGE_METHODS
from .image import read_image, write_image
from .image import as_3_channels_image
from .ctl import (
    ctl_render,
    process_image_ctl,
    template_ctl_transform_float,
    template_ctl_transform_float3,
)
from .ocio import process_image_OpenColorIO
from .tabular import (
    read_spectral_data_from_csv_file,
    read_sds_from_csv_file,
    write_sds_to_csv_file,
)
from .tm2714 import Header_IESTM2714, SpectralDistribution_IESTM2714
from .uprtek_sekonic import (
    SpectralDistribution_UPRTek,
    SpectralDistribution_Sekonic,
)
from .xrite import read_sds_from_xrite_file

__all__ = []
__all__ += luts.__all__
__all__ += [
    "Image_Specification_Attribute",
    "image_specification_OpenImageIO",
    "convert_bit_depth",
]
__all__ += [
    "read_image_OpenImageIO",
    "write_image_OpenImageIO",
]
__all__ += [
    "read_image_Imageio",
    "write_image_Imageio",
]
__all__ += [
    "READ_IMAGE_METHODS",
    "WRITE_IMAGE_METHODS",
]
__all__ += [
    "read_image",
    "write_image",
]
__all__ += [
    "ctl_render",
    "process_image_ctl",
    "template_ctl_transform_float",
    "template_ctl_transform_float3",
]
__all__ += [
    "as_3_channels_image",
]
__all__ += [
    "process_image_OpenColorIO",
]
__all__ += [
    "read_spectral_data_from_csv_file",
    "read_sds_from_csv_file",
    "write_sds_to_csv_file",
]
__all__ += [
    "SpectralDistribution_UPRTek",
    "SpectralDistribution_Sekonic",
]
__all__ += [
    "Header_IESTM2714",
    "SpectralDistribution_IESTM2714",
]
__all__ += [
    "read_sds_from_xrite_file",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class io(ModuleAPI):
    """Define a class acting like the *io* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

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
