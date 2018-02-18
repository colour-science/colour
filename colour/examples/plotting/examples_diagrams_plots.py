# -*- coding: utf-8 -*-
"""
Showcases *CIE* chromaticity diagrams plotting examples.
"""

from colour import ILLUMINANTS_RELATIVE_SPDS
from colour.plotting import (
    colour_plotting_defaults, chromaticity_diagram_plot_CIE1931,
    chromaticity_diagram_plot_CIE1960UCS, chromaticity_diagram_plot_CIE1976UCS,
    spds_chromaticity_diagram_plot_CIE1931,
    spds_chromaticity_diagram_plot_CIE1960UCS,
    spds_chromaticity_diagram_plot_CIE1976UCS)
from colour.utilities import message_box

message_box('"CIE" Chromaticity Diagrams Plots')

colour_plotting_defaults()

message_box('Plotting "CIE 1931 Chromaticity Diagram".')
chromaticity_diagram_plot_CIE1931()

print('\n')

message_box('Plotting "CIE 1960 UCS Chromaticity Diagram".')
chromaticity_diagram_plot_CIE1960UCS()

print('\n')

message_box('Plotting "CIE 1976 UCS Chromaticity Diagram".')
chromaticity_diagram_plot_CIE1976UCS()

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" relative spectral power '
             'distribution chromaticity coordinates in '
             '"CIE 1931 Chromaticity Diagram".'))
A = ILLUMINANTS_RELATIVE_SPDS['A']
D65 = ILLUMINANTS_RELATIVE_SPDS['D65']
spds_chromaticity_diagram_plot_CIE1931((A, D65))

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" relative spectral power '
             'distribution chromaticity coordinates in '
             '"CIE 1960 UCS Chromaticity Diagram".'))
spds_chromaticity_diagram_plot_CIE1960UCS((A, D65))

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" relative spectral power '
             'distribution chromaticity coordinates in '
             '"CIE 1976 UCS Chromaticity Diagram".'))
spds_chromaticity_diagram_plot_CIE1976UCS((A, D65))
