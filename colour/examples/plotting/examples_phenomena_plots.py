"""Showcases optical phenomena plotting examples."""

from colour.phenomena import sd_rayleigh_scattering
from colour.plotting import (
    colour_style,
    plot_multi_sds,
    plot_single_sd_rayleigh_scattering,
    plot_the_blue_sky,
)
from colour.utilities import message_box

message_box("Optical Phenomena Plots")

colour_style()

message_box('Plotting a single "Rayleigh" scattering spectral "distribution."')
plot_single_sd_rayleigh_scattering()

print("\n")

message_box(
    'Comparing multiple "Rayleigh" scattering spectral distributions with '
    "different CO_2 concentrations."
)
name_template = "Rayleigh Scattering - CO2: {0} ppm"
sds_rayleigh = []
for ppm in (0, 50, 300):
    sd_rayleigh = sd_rayleigh_scattering(CO2_concentration=ppm)
    sd_rayleigh.name = name_template.format(ppm)
    sds_rayleigh.append(sd_rayleigh)
plot_multi_sds(
    sds_rayleigh,
    title=(
        "Rayleigh Optical Depth - " 'Comparing "C02" Concentration Influence'
    ),
    y_label="Optical Depth",
)

print("\n")

message_box('Plotting "The Blue Sky".')
plot_the_blue_sky()
