"""
Demonstrate optical phenomena plotting.

This module provides examples of plotting Rayleigh scattering,
CIE Standard General Sky luminance distributions, Prague Sky Model
radiance distributions, and other optical phenomena.
"""

import os

from colour.phenomena import sd_rayleigh_scattering
from colour.phenomena.sky.wilkie2021 import (
    PATH_PRAGUE_SKY_MODEL_DATASET_GROUND,
    SkyDataset_Wilkie2021,
)
from colour.plotting import (
    colour_style,
    plot_multi_sds,
    plot_single_sd_rayleigh_scattering,
    plot_sky_colour_Wilkie2021,
    plot_sky_luminance_distribution_CIE2003,
    plot_sky_radiance_Wilkie2021,
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
    title='Rayleigh Optical Depth - Comparing "C02" Concentration Influence',
    y_label="Optical Depth",
)

print("\n")

message_box('Plotting "The Blue Sky".')
plot_the_blue_sky()

print("\n")

message_box(
    'Plotting "CIE Standard General Sky" Type 12 (Clear Sky) using polar projection.'
)
plot_sky_luminance_distribution_CIE2003(12, method="Polar")

print("\n")

message_box(
    'Plotting "CIE Standard General Sky" Type 12 (Clear Sky) '
    "using latitude-longitude projection."
)
plot_sky_luminance_distribution_CIE2003(12, method="Latlong")

print("\n")

if os.path.isfile(PATH_PRAGUE_SKY_MODEL_DATASET_GROUND):
    message_box('Plotting "Prague Sky Model" radiance using polar projection.')
    dataset = SkyDataset_Wilkie2021(PATH_PRAGUE_SKY_MODEL_DATASET_GROUND)
    plot_sky_radiance_Wilkie2021(dataset, method="Polar")

    print("\n")

    message_box(
        'Plotting "Prague Sky Model" radiance using latitude-longitude projection.'
    )
    plot_sky_radiance_Wilkie2021(dataset, method="Latlong")

    print("\n")

    message_box('Plotting "Prague Sky Model" true colour sky with sun contribution.')
    plot_sky_colour_Wilkie2021(dataset)
