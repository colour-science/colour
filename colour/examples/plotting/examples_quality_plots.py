#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour quality plotting examples.
"""

import colour
from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Colour Quality Plots')

message_box(('Plotting various illuminants and light sources '
             '"colour rendering index".'))
colour_rendering_index_bars_plot(
    colour.ILLUMINANTS_RELATIVE_SPDS.get('F2'))
colour_rendering_index_bars_plot(
    colour.LIGHT_SOURCES_RELATIVE_SPDS.get('F32T8/TL841 (Triphosphor)'))

print('\n')

message_box(('Plotting various illuminants and light sources '
             '"colour quality scale".'))
colour_quality_scale_bars_plot(
    colour.ILLUMINANTS_RELATIVE_SPDS.get('F2'))
colour_quality_scale_bars_plot(
    colour.LIGHT_SOURCES_RELATIVE_SPDS.get('F32T8/TL841 (Triphosphor)'))
