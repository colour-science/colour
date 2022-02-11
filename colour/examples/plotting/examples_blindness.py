"""Showcases corresponding colour blindness plotting examples."""

import numpy as np
import os

import colour
from colour.plotting import (
    colour_style,
    plot_cvd_simulation_Machado2009,
    plot_image,
)
from colour.utilities.verbose import message_box

RESOURCES_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources"
)

colour_style()

ISHIHARA_CBT_3_IMAGE = colour.cctf_decoding(
    colour.read_image(
        os.path.join(
            RESOURCES_DIRECTORY, "Ishihara_Colour_Blindness_Test_Plate_3.png"
        )
    ),
    function="sRGB",
)

message_box("Colour Blindness Plots")

message_box('Displaying "Ishihara Colour Blindness Test - Plate 3".')
plot_image(
    colour.cctf_encoding(ISHIHARA_CBT_3_IMAGE),
    text_kwargs={"text": "Normal Trichromat", "color": "black"},
)

print("\n")

message_box(
    'Simulating average "Protanomaly" on '
    '"Ishihara Colour Blindness Test - Plate 3" with Machado (2010) model and '
    "pre-computed matrix."
)
plot_cvd_simulation_Machado2009(
    ISHIHARA_CBT_3_IMAGE,
    "Protanomaly",
    0.5,
    text_kwargs={"text": "Protanomaly - 50%", "color": "black"},
)

print("\n")

M_a = colour.matrix_anomalous_trichromacy_Machado2009(
    colour.MSDS_CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"],
    colour.MSDS_DISPLAY_PRIMARIES["Typical CRT Brainard 1997"],
    np.array([10, 0, 0]),
)
message_box(
    'Simulating average "Protanomaly" on '
    '"Ishihara Colour Blindness Test - Plate 3" with Machado (2010) model '
    'using "Stockman & Sharpe 2 Degree Cone Fundamentals" and '
    '"Typical CRT Brainard 1997" "RGB" display primaries.'
)
plot_cvd_simulation_Machado2009(
    ISHIHARA_CBT_3_IMAGE,
    M_a=M_a,
    text_kwargs={"text": "Average Protanomaly - 10nm", "color": "black"},
)

print("\n")

M_a = colour.matrix_anomalous_trichromacy_Machado2009(
    colour.MSDS_CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"],
    colour.MSDS_DISPLAY_PRIMARIES["Typical CRT Brainard 1997"],
    np.array([20, 0, 0]),
)
message_box(
    'Simulating "Protanopia" on '
    '"Ishihara Colour Blindness Test - Plate 3" with Machado (2010) model '
    'using "Stockman & Sharpe 2 Degree Cone Fundamentals" and '
    '"Typical CRT Brainard 1997" "RGB" display primaries.'
)
plot_cvd_simulation_Machado2009(
    ISHIHARA_CBT_3_IMAGE,
    M_a=M_a,
    text_kwargs={"text": "Protanopia - 20nm", "color": "black"},
)
