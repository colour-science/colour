#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases optical phenomenon plotting examples.
"""

import colour
from colour.phenomenon import rayleigh_scattering_spd
from colour.plotting import *

# Plotting a single *Rayleigh Scattering* spectral power distribution.
single_rayleigh_scattering_spd_plot()

# Comparing multiple *Rayleigh Scattering* spectral power distributions with
# different :math:`C0_2` concentrations.
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

# Plotting *The Blue Sky*.
the_blue_sky_plot()
