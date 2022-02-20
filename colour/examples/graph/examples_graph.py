"""Showcases *Automatic Colour Conversion Graph* computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box("Automatic Colour Conversion Graph")

message_box(
    'Converting a "ColorChecker" "dark skin" sample spectral distribution to '
    '"Output-Referred" "sRGB" colourspace.'
)

sd_dark_skin = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"]
print(colour.convert(sd_dark_skin, "Spectral Distribution", "sRGB"))
print(
    colour.XYZ_to_sRGB(
        colour.sd_to_XYZ(
            sd_dark_skin, illuminant=colour.SDS_ILLUMINANTS["D65"]
        )
        / 100
    )
)

print("\n")

RGB = np.array([0.45675795, 0.30986982, 0.24861924])
message_box(
    f'Converting to the "CAM16-UCS" colourspace from given "Output-Referred" '
    f'"sRGB" colourspace values:\n\n\t{RGB}'
)
print(colour.convert(RGB, "Output-Referred RGB", "CAM16UCS"))
specification = colour.XYZ_to_CAM16(
    colour.sRGB_to_XYZ(RGB) * 100,
    XYZ_w=colour.xy_to_XYZ(
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    )
    * 100,
    L_A=64 / np.pi * 0.2,
    Y_b=20,
)
print(
    colour.JMh_CAM16_to_CAM16UCS(
        colour.utilities.tstack(
            [
                specification.J,
                specification.M,
                specification.h,
            ]
        )
    )
    / 100
)

print("\n")

Jpapbp = np.array([0.39994811, 0.09206558, 0.0812752])
message_box(
    f'Converting to the "Output-Referred" "sRGB" colourspace from given '
    f'"CAM16-UCS" colourspace colourspace values:\n\n\t{RGB}'
)
print(
    colour.convert(
        Jpapbp,
        "CAM16UCS",
        "sRGB",
        verbose_kwargs={"describe": "Extended", "width": 75},
    )
)
J, M, h = colour.utilities.tsplit(colour.CAM16UCS_to_JMh_CAM16(Jpapbp * 100))
specification = colour.CAM_Specification_CAM16(J=J, M=M, h=h)
print(
    colour.XYZ_to_sRGB(
        colour.CAM16_to_XYZ(
            specification,
            XYZ_w=colour.xy_to_XYZ(
                colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
                    "D65"
                ]
            )
            * 100,
            L_A=64 / np.pi * 0.2,
            Y_b=20,
        )
        / 100
    )
)
