"""
Demonstrate *Machado et al. (2009)* colour vision deficiency simulation.

This module demonstrates the computation and application of colour vision
deficiency matrices using *Machado et al. (2009)* method for simulating
protanomaly, deuteranomaly, and tritanomaly conditions.
"""

import numpy as np

import colour
from colour.utilities.verbose import message_box

message_box("Simulation of CVD - Machado (2009)")

M_a = colour.matrix_anomalous_trichromacy_Machado2009(
    colour.MSDS_CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"],
    colour.MSDS_DISPLAY_PRIMARIES["Typical CRT Brainard 1997"],
    np.array([10, 0, 0]),
)
message_box(
    f'Compute a "Protanomaly" matrix using '
    f'"Stockman & Sharpe 2 Degree Cone Fundamentals" and '
    f'"Typical CRT Brainard 1997" "RGB" display primaries for a 10nm shift:\n\n{M_a}'
)

print("\n")

M_a = colour.matrix_cvd_Machado2009("Protanomaly", 0.5)
message_box(
    f'Retrieve a "Protanomaly" pre-computed matrix for a 50% severity:\n\n{M_a}'
)

print("\n")

M_a = colour.matrix_anomalous_trichromacy_Machado2009(
    colour.MSDS_CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"],
    colour.MSDS_DISPLAY_PRIMARIES["Typical CRT Brainard 1997"],
    np.array([0, 10, 0]),
)
message_box(
    f'Compute a "Deuteranomaly" matrix using '
    f'"Stockman & Sharpe 2 Degree Cone Fundamentals" and '
    f'"Typical CRT Brainard 1997" "RGB" display primaries for a 10nm shift:\n\n{M_a}'
)

print("\n")

M_a = colour.matrix_cvd_Machado2009("Deuteranomaly", 0.5)
message_box(
    f'Retrieve a "Deuteranomaly" pre-computed matrix for a 50% severity:\n\n{M_a}'
)

print("\n")

M_a = colour.matrix_anomalous_trichromacy_Machado2009(
    colour.MSDS_CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"],
    colour.MSDS_DISPLAY_PRIMARIES["Typical CRT Brainard 1997"],
    np.array([0, 0, 27]),
)
message_box(
    f'Compute a "Tritanomaly" matrix using '
    f'"Stockman & Sharpe 2 Degree Cone Fundamentals" and '
    f'"Typical CRT Brainard 1997" "RGB" display primaries for a 27nm shift:\n\n{M_a}'
)

print("\n")

M_a = colour.matrix_cvd_Machado2009("Tritanomaly", 0.5)
message_box(
    f'Retrieve a "Tritanomaly" pre-computed matrix for a 50% severity:\n\n{M_a}'
)
