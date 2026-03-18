"""
Demonstrate *CIE Standard General Sky* computations.

This module demonstrates *CIE Standard General Sky* luminance distribution
computations as defined in *CIE S 011/E:2003*, reproducing Figures 2, 3 and 4
from the standard.
"""

import matplotlib.pyplot as plt
import numpy as np

import colour
from colour.plotting import colour_style, render
from colour.utilities import message_box

message_box("CIE Standard General Sky Computations")

colour_style()

# ==============================================================================
# Figure 2: Standard Gradation Function Groups
# ==============================================================================
message_box(
    'Reproducing "Figure 2" from the standard:\n\n'
    "\tStandard gradation function groups (I-VI).\n\n"
    "The gradation function phi(Z) is plotted as phi(Z)/phi(0) for each\n"
    "gradation group, showing relative luminance from zenith to horizon."
)

GRADATION_GROUPS = {
    1: {"a": 4.0, "b": -0.70, "label": "I"},
    2: {"a": 1.1, "b": -0.8, "label": "II"},
    3: {"a": 0, "b": -1.0, "label": "III"},
    4: {"a": -1.0, "b": -0.55, "label": "IV"},
    5: {"a": -1.0, "b": -0.32, "label": "V"},
    6: {"a": -1.0, "b": -0.15, "label": "VI"},
}

Z_degrees = np.linspace(0, 90, 500)
Z_radians = np.radians(Z_degrees)

figure, axes = plt.subplots(1, 1, figsize=(8, 6))

for group in GRADATION_GROUPS.values():
    a, b = group["a"], group["b"]

    phi = colour.phenomena.sky_luminance_gradation_CIE2003(Z_radians, a, b)
    phi_0 = colour.phenomena.sky_luminance_gradation_CIE2003(0, a, b)

    axes.plot(Z_degrees, phi / phi_0, label=group["label"])

axes.set_xlabel("Zenith angle Z in degrees")
axes.set_ylabel(r"Relative gradation $\varphi(Z)/\varphi(0)$")
axes.set_title("Figure 2. Standard gradation function groups.")
axes.set_xlim(0, 90)
axes.set_ylim(0, 8)
axes.legend()
axes.grid(True, alpha=0.3)

render(show=True)

# ==============================================================================
# Figure 3: Standard Indicatrix Function Groups
# ==============================================================================
message_box(
    'Reproducing "Figure 3" from the standard:\n\n'
    "\tStandard indicatrix function groups (1-6).\n\n"
    "The scattering indicatrix f(chi) is plotted for each indicatrix group,\n"
    "showing relative luminance as function of angular distance from the sun."
)

INDICATRIX_GROUPS = {
    1: {"c": 0, "d": -1.0, "e": 0, "label": "1"},
    2: {"c": 2, "d": -1.5, "e": 0.15, "label": "2"},
    3: {"c": 5, "d": -2.5, "e": 0.30, "label": "3"},
    4: {"c": 10, "d": -3.0, "e": 0.45, "label": "4"},
    5: {"c": 16, "d": -3.0, "e": 0.30, "label": "5"},
    6: {"c": 24, "d": -2.8, "e": 0.15, "label": "6"},
}

chi_degrees = np.linspace(0, 180, 500)
chi_radians = np.radians(chi_degrees)

figure, axes = plt.subplots(1, 1, figsize=(8, 6))

for group in INDICATRIX_GROUPS.values():
    c, d, e = group["c"], group["d"], group["e"]

    f = colour.phenomena.sky_scattering_indicatrix_CIE2003(chi_radians, c, d, e)

    axes.plot(chi_degrees, f, label=group["label"])

axes.set_xlabel(r"Scattering angle $\chi$ in degrees")
axes.set_ylabel(r"Relative indicatrix $f(\chi)$")
axes.set_title("Figure 3. Standard indicatrix function groups.")
axes.set_xlim(0, 180)
axes.set_ylim(0.8, 20)
axes.set_yscale("log")
axes.set_yticks([0.8, 1, 2, 3, 4, 5, 10, 20])
axes.set_yticklabels(["0.8", "1", "2", "3", "4", "5", "10", "20"])
axes.legend()
axes.grid(True, alpha=0.3)

render(show=True)

# ==============================================================================
# Figure 4: Difference Between Sky Types 1 and 16
# ==============================================================================
message_box(
    'Reproducing "Figure 4" from the standard:\n\n'
    "\tDifference between Sky Types 1 and 16.\n\n"
    "Comparison of the CIE General Sky Type 1 and the Traditional Overcast\n"
    "Sky (Type 16) luminance distributions as function of elevation angle."
)

Z_degrees = np.linspace(0, 90, 500)
Z_radians = np.radians(Z_degrees)

# Type 1 relative luminance (azimuthally uniform with indicatrix group 1,
# sun position is irrelevant).
L_type1 = colour.phenomena.sky_luminance_distribution_CIE2003(
    1, Z_radians, 0, np.radians(45), 0
)

# Type 16 (Traditional Overcast Sky).
L_type16 = colour.phenomena.sky_luminance_distribution_overcast_CIE2003(Z_radians)

# Relative difference.
relative_difference = (L_type16 - L_type1) / L_type1

figure, axes_left = plt.subplots(1, 1, figsize=(8, 6))
axes_right = axes_left.twinx()

axes_left.plot(Z_degrees, L_type1, "k-", label="CIE General Sky: Sky Type 1")
axes_left.plot(
    Z_degrees,
    L_type16,
    "k--",
    label="Traditional Overcast Sky: Sky Type 16",
)
axes_right.plot(Z_degrees, relative_difference, "k:", label="Relative Difference")

axes_left.set_xlabel(r"Angular distance of a sky element and the zenith [deg]")
axes_left.set_ylabel("Relative luminance")
axes_right.set_ylabel("Relative Difference")
axes_left.set_xlim(0, 90)
axes_left.set_ylim(0, 1.0)
axes_right.set_ylim(-0.30, 0.30)
axes_left.set_title("Figure 4. Difference between Sky Types 1 and 16.")

lines_left, labels_left = axes_left.get_legend_handles_labels()
lines_right, labels_right = axes_right.get_legend_handles_labels()
axes_left.legend(
    lines_left + lines_right, labels_left + labels_right, fontsize=8, loc="lower left"
)

axes_left.grid(True, alpha=0.3)

render(show=True)
