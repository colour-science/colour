#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *CIE* chromaticity diagrams plotting examples.
"""

from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('"CIE" Chromaticity Diagrams Plots')

message_box('Plotting "CIE 1931 Chromaticity Diagram".')
CIE_1931_chromaticity_diagram_plot()

print('\n')

message_box('Plotting "CIE 1960 UCS Chromaticity Diagram".')
CIE_1960_UCS_chromaticity_diagram_plot()

print('\n')

message_box('Plotting "CIE 1976 UCS Chromaticity Diagram".')
CIE_1976_UCS_chromaticity_diagram_plot()
