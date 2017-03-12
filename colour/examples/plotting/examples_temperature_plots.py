#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour temperature and correlated colour temperature plotting
examples.
"""

from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Colour Temperature and Correlated Colour Temperature Plots')

colour_plotting_defaults()

message_box('Plotting planckian locus in "CIE 1931 Chromaticity Diagram".')
planckian_locus_CIE_1931_chromaticity_diagram_plot(['A', 'B', 'C'])

print('\n')

message_box('Plotting planckian locus in "CIE 1960 UCS Chromaticity Diagram".')
planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot(['A', 'B', 'C'])
