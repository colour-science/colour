# -*- coding: utf-8 -*-
"""
Showcases optical phenomena plotting examples.
"""

from colour.phenomena import sd_rayleigh_scattering
from colour.plotting import (colour_style, plot_multi_sds,
                             plot_single_sd_rayleigh_scattering,
                             plot_the_blue_sky)
from colour.utilities import message_box

message_box('Optical Phenomenons Plots')

colour_style()

message_box(('Plotting a single "Rayleigh" scattering spectral '
             'distribution.'))
plot_single_sd_rayleigh_scattering()

print('\n')

message_box(('Comparing multiple "Rayleigh" scattering spectral '
             'distributions with different CO_2 concentrations.'))
name_template = 'Rayleigh Scattering - CO2: {0} ppm'
rayleigh_sds = []
for ppm in (0, 50, 300):
    rayleigh_sd = sd_rayleigh_scattering(CO2_concentration=ppm)
    rayleigh_sd.name = name_template.format(ppm)
    rayleigh_sds.append(rayleigh_sd)
plot_multi_sds(
    rayleigh_sds,
    title=('Rayleigh Optical Depth - '
           'Comparing "C02" Concentration Influence'),
    y_label='Optical Depth')

print('\n')

message_box('Plotting "The Blue Sky".')
plot_the_blue_sky()
