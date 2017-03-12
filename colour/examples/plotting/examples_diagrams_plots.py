#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *CIE* chromaticity diagrams plotting examples.
"""

from colour import ILLUMINANTS_RELATIVE_SPDS
from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('"CIE" Chromaticity Diagrams Plots')

colour_plotting_defaults()

message_box('Plotting "CIE 1931 Chromaticity Diagram".')
CIE_1931_chromaticity_diagram_plot()

print('\n')

message_box('Plotting "CIE 1960 UCS Chromaticity Diagram".')
CIE_1960_UCS_chromaticity_diagram_plot()

print('\n')

message_box('Plotting "CIE 1976 UCS Chromaticity Diagram".')
CIE_1976_UCS_chromaticity_diagram_plot()

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" relative spectral power '
             'distribution chromaticity coordinates in '
             '"CIE 1931 Chromaticity Diagram".'))
A = ILLUMINANTS_RELATIVE_SPDS['A']
D65 = ILLUMINANTS_RELATIVE_SPDS['D65']
spds_CIE_1931_chromaticity_diagram_plot((A, D65))

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" relative spectral power '
             'distribution chromaticity coordinates in '
             '"CIE 1960 UCS Chromaticity Diagram".'))
spds_CIE_1960_UCS_chromaticity_diagram_plot((A, D65))

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" relative spectral power '
             'distribution chromaticity coordinates in '
             '"CIE 1976 UCS Chromaticity Diagram".'))
spds_CIE_1976_UCS_chromaticity_diagram_plot((A, D65))
