# -*- coding: utf-8 -*-
"""
Showcases colour temperature and correlated colour temperature plotting
examples.
"""

from colour.plotting import (
    colour_style, plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS)
from colour.utilities import message_box

message_box('Colour Temperature and Correlated Colour Temperature Plots')

colour_style()

message_box('Plotting planckian locus in "CIE 1931 Chromaticity Diagram".')
plot_planckian_locus_in_chromaticity_diagram_CIE1931(['A', 'B', 'C'])

print('\n')

message_box('Plotting planckian locus in "CIE 1960 UCS Chromaticity Diagram".')
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(['A', 'B', 'C'])
