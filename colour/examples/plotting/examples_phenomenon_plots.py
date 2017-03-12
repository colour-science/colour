#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases optical phenomenons plotting examples.
"""

from colour.phenomenons import rayleigh_scattering_spd
from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Optical Phenomenons Plots')

colour_plotting_defaults()

message_box(('Plotting a single "Rayleigh" scattering spectral power '
             'distribution.'))
single_rayleigh_scattering_spd_plot()

print('\n')

message_box(('Comparing multiple "Rayleigh" scattering spectral power '
             'distributions with different CO_2 concentrations.'))
name_template = 'Rayleigh Scattering - CO2: {0} ppm'
rayleigh_spds = []
for ppm in (0, 50, 300):
    rayleigh_spd = rayleigh_scattering_spd(CO2_concentration=ppm)
    rayleigh_spd.name = name_template.format(ppm)
    rayleigh_spds.append(rayleigh_spd)
multi_spd_plot(rayleigh_spds,
               title=('Rayleigh Optical Depth - '
                      'Comparing "C02" Concentration Influence'),
               y_label='Optical Depth',
               legend_location='upper right')

print('\n')

message_box('Plotting "The Blue Sky".')
the_blue_sky_plot()
