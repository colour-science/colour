#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour temperature and correlated colour temperature plotting
examples.
"""

from colour.plotting import *

# Plotting planckian locus in *CIE 1931 Chromaticity Diagram*.
planckian_locus_CIE_1931_chromaticity_diagram_plot(['A', 'B', 'C'])

# Plotting planckian locus in *CIE 1960 UCS Chromaticity Diagram*.
planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot(['A', 'B', 'C'])