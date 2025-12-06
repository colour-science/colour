"""
Demonstrate *Fichet, Pacanowski and Wilkie (2021)* OpenEXR Layout for Spectral Images.

This module provides examples of reading and writing spectral images
using the OpenEXR layout format.
"""

import os
import tempfile

import colour
from colour.utilities import is_imageio_installed, message_box

if is_imageio_installed():
    ROOT_RESOURCES = os.path.join(
        os.path.dirname(__file__), "..", "..", "io", "tests", "resources"
    )

    message_box(
        '"Fichet, Pacanowski and Wilkie (2021)" Spectral Image Reading and Writing'
    )

    message_box("Reading a spectral image.")
    path = os.path.join(ROOT_RESOURCES, "Ohta1997.exr")
    components, specification = colour.read_spectral_image_Fichet2021(
        path, additional_data=True
    )
    print(components)
    print(specification)

    print("\n")

    message_box("Writing a spectral image.")
    _descriptor, path = tempfile.mkstemp(suffix=".exr")
    colour.write_spectral_image_Fichet2021(components, path)
    components = colour.read_spectral_image_Fichet2021(path)
    print(components)
