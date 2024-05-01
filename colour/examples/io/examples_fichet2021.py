"""
Showcases *Fichet, Pacanowski and Wilkie (2021)*
*OpenEXR Layout for Spectral Images* related examples.
"""

import os
import tempfile

import colour
from colour.utilities import message_box

ROOT_RESOURCES = os.path.join(
    os.path.dirname(__file__), "..", "..", "io", "tests", "resources"
)

message_box('"Fichet, Pacanowski and Wilkie (2021)" Spectral Image Reading and Writing')

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
colour.write_spectral_image_Fichet2021(components, path)  # pyright: ignore
components = colour.read_spectral_image_Fichet2021(path)
print(components)
