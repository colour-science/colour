"""Showcases Color Transformation Language (CTL) related examples."""

import os
import tempfile

import colour
from colour.utilities import message_box

ROOT_RESOURCES = os.path.join(
    os.path.dirname(__file__), "..", "..", "io", "tests", "resources"
)

message_box("Color Transformation Language (CTL)")

message_box(
    'Using a "CTL" string and the "float" template to transform an image.'
)

ctl_adjust_exposure_float = colour.io.template_ctl_transform_float(
    "rIn * pow(2, exposure)",
    "gIn * pow(2, exposure)",
    "bIn * pow(2, exposure)",
    description="Adjust Exposure",
    parameters=["input float exposure = 0.0"],
)
print(ctl_adjust_exposure_float)
path_input = os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
_descriptor, path_output = tempfile.mkstemp(suffix=".exr")
colour.io.ctl_render(
    path_input,
    path_output,
    {ctl_adjust_exposure_float: ["-param1 exposure 3.0"]},
    "-verbose",
    "-force",
)
print(colour.read_image(path_output)[:3])

print("\n")

message_box(
    'Using a "CTL" "float" template based transform file to transform an image.'
)

path_input = os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
colour.io.ctl_render(
    path_input,
    path_output,
    {
        os.path.join(ROOT_RESOURCES, "Adjust_Exposure_Float.ctl"): [
            "-param1 exposure 3.0"
        ]
    },
    "-verbose",
    "-force",
)
print(colour.read_image(path_output)[:3])

print("\n")

message_box(
    'Using a "CTL" "float" template based transform file to transform an array.'
)

print(os.path.join(ROOT_RESOURCES, "Adjust_Exposure_Float.ctl"))
a = colour.utilities.full((4, 2, 3), 0.18)  # pyright: ignore
print(
    colour.io.process_image_ctl(
        a,
        {
            os.path.join(ROOT_RESOURCES, "Adjust_Exposure_Float.ctl"): [
                "-param1 exposure 3.0"
            ]
        },
        "-verbose",
        "-force",
    )
)

print("\n")

message_box(
    'Using a "CTL" string and the "float3" template to transform an image.'
)

ctl_adjust_exposure_float3 = colour.io.template_ctl_transform_float3(
    "adjust_exposure(rgbIn, exposure)",
    description="Adjust Exposure",
    header="""
float[3] adjust_exposure(float rgbIn[3], float exposureIn)
{
    float rgbOut[3];

    float exposure = pow(2, exposureIn);

    rgbOut[0] = rgbIn[0] * exposure;
    rgbOut[1] = rgbIn[1] * exposure;
    rgbOut[2] = rgbIn[2] * exposure;

    return rgbOut;
}\n"""[
        1:
    ],
    parameters=["input float exposure = 0.0"],
)
print(ctl_adjust_exposure_float3)
path_input = os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
colour.io.ctl_render(
    path_input,
    path_output,
    {ctl_adjust_exposure_float3: ["-param1 exposure 3.0"]},
    "-verbose",
    "-force",
)
print(colour.read_image(path_output)[:3])

print("\n")

message_box(
    'Using a "CTL" "float3" template based transform file to transform an image.'
)

path_input = os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
_descriptor, path_output = tempfile.mkstemp(suffix=".exr")
colour.io.ctl_render(
    path_input,
    path_output,
    {
        os.path.join(ROOT_RESOURCES, "Adjust_Exposure_Float3.ctl"): [
            "-param1 exposure 3.0"
        ]
    },
    "-verbose",
    "-force",
)
print(colour.read_image(path_output)[:3])
os.remove(path_output)

print("\n")

message_box(
    'Using a "CTL" "float3" template based transform file to transform an array.'
)

print(
    colour.io.process_image_ctl(
        a,
        {
            os.path.join(ROOT_RESOURCES, "Adjust_Exposure_Float3.ctl"): [
                "-param1 exposure 3.0"
            ]
        },
        "-verbose",
        "-force",
    )
)

print("\n")

ROOT_ACES_DEV_TRANSFORMS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "..",
    "ampas",
    "aces-dev",
    "transforms",
    "ctl",
)

if os.path.exists(ROOT_ACES_DEV_TRANSFORMS):
    message_box(
        'Running the "aces-dev" "RRT" "CTL" function to transform an array.'
    )

    CTL_MODULE_PATH = (
        f"{ROOT_ACES_DEV_TRANSFORMS}:"
        f"{ROOT_ACES_DEV_TRANSFORMS}/lib:"
        f"{ROOT_ACES_DEV_TRANSFORMS}/utilities"
    )

    print(
        colour.io.process_image_ctl(
            a,
            [f"{ROOT_ACES_DEV_TRANSFORMS}/rrt/RRT.ctl"],
            env=dict(
                os.environ,
                CTL_MODULE_PATH=CTL_MODULE_PATH,
            ),
        )
    )

    print("\n")

    message_box(
        'Running the "aces-dev" "Y_2_linCV" "CTL" function to transform an array.'
    )

    def format_imports(imports):
        """Format given imports."""
        return [f'import "{i}";' for i in imports]

    ctl_Y_2_linCV_float = colour.io.template_ctl_transform_float(
        "Y_2_linCV(rIn, CINEMA_WHITE, CINEMA_BLACK)",
        "Y_2_linCV(gIn, CINEMA_WHITE, CINEMA_BLACK)",
        "Y_2_linCV(bIn, CINEMA_WHITE, CINEMA_BLACK)",
        imports=format_imports(
            [
                "ACESlib.Utilities",
                "ACESlib.Transform_Common",
                "ACESlib.ODT_Common",
            ]
        ),
    )
    print(ctl_Y_2_linCV_float)
    print(
        colour.io.process_image_ctl(
            a,
            [ctl_Y_2_linCV_float],
            env=dict(
                os.environ,
                CTL_MODULE_PATH=CTL_MODULE_PATH,
            ),
        )
    )

    print("\n")

    message_box(
        'Running the "aces-dev" "darkSurround_to_dimSurround" "CTL" function '
        "to transform an array."
    )
    ctl_darkSurround_to_dimSurround_float3 = (
        colour.io.template_ctl_transform_float3(
            "darkSurround_to_dimSurround(rgbIn)",
            imports=format_imports(
                [
                    "ACESlib.Utilities",
                    "ACESlib.Transform_Common",
                    "ACESlib.ODT_Common",
                ]
            ),
        )
    )
    print(ctl_darkSurround_to_dimSurround_float3)
    print(
        colour.io.process_image_ctl(
            a,
            [ctl_darkSurround_to_dimSurround_float3],
            env=dict(
                os.environ,
                CTL_MODULE_PATH=CTL_MODULE_PATH,
            ),
        )
    )
