#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour quality plotting examples.
"""

import colour
from colour.plotting import *

# Plotting various illuminants *colour rendering index*.
colour_rendering_index_bars_plot(colour.ILLUMINANTS_RELATIVE_SPDS.get('F2'))
colour_rendering_index_bars_plot(colour.ILLUMINANTS_RELATIVE_SPDS.get('F10'))
colour_rendering_index_bars_plot(colour.ILLUMINANTS_RELATIVE_SPDS.get('D50'))
