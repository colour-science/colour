# -*- coding: utf-8 -*-
"""
Showcases colour temperature and correlated colour temperature plotting
examples.
"""

from colour.plotting import (
    colour_plotting_defaults,
    planckian_locus_chromaticity_diagram_plot_CIE1931,
    planckian_locus_chromaticity_diagram_plot_CIE1960UCS)
from colour.utilities import message_box

message_box('Colour Temperature and Correlated Colour Temperature Plots')

colour_plotting_defaults()

message_box('Plotting planckian locus in "CIE 1931 Chromaticity Diagram".')
planckian_locus_chromaticity_diagram_plot_CIE1931(['A', 'B', 'C'])

print('\n')

message_box('Plotting planckian locus in "CIE 1960 UCS Chromaticity Diagram".')
planckian_locus_chromaticity_diagram_plot_CIE1960UCS(['A', 'B', 'C'])
