# -*- coding: utf-8 -*-
"""
Showcases *CIE* chromaticity diagrams plotting examples.
"""

from colour import ILLUMINANTS_SDS
from colour.plotting import (colour_style, plot_chromaticity_diagram_CIE1931,
                             plot_chromaticity_diagram_CIE1960UCS,
                             plot_chromaticity_diagram_CIE1976UCS,
                             plot_sds_in_chromaticity_diagram_CIE1931,
                             plot_sds_in_chromaticity_diagram_CIE1960UCS,
                             plot_sds_in_chromaticity_diagram_CIE1976UCS)
from colour.utilities import message_box

message_box('"CIE" Chromaticity Diagrams Plots')

colour_style()

message_box('Plotting "CIE 1931 Chromaticity Diagram".')
plot_chromaticity_diagram_CIE1931()

print('\n')

message_box('Plotting "CIE 1960 UCS Chromaticity Diagram".')
plot_chromaticity_diagram_CIE1960UCS()

print('\n')

message_box('Plotting "CIE 1976 UCS Chromaticity Diagram".')
plot_chromaticity_diagram_CIE1976UCS()

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" spectral '
             'distribution chromaticity coordinates in '
             '"CIE 1931 Chromaticity Diagram".'))
A = ILLUMINANTS_SDS['A']
D65 = ILLUMINANTS_SDS['D65']
plot_sds_in_chromaticity_diagram_CIE1931((A, D65))

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" spectral '
             'distribution chromaticity coordinates in '
             '"CIE 1960 UCS Chromaticity Diagram".'))
plot_sds_in_chromaticity_diagram_CIE1960UCS((A, D65))

print('\n')

message_box(('Plotting "CIE Standard Illuminant A" and '
             '"CIE Standard Illuminant D65" spectral '
             'distribution chromaticity coordinates in '
             '"CIE 1976 UCS Chromaticity Diagram".'))
plot_sds_in_chromaticity_diagram_CIE1976UCS((A, D65))
