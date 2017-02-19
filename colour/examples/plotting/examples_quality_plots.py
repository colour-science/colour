#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour quality plotting examples.
"""

import colour
from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Colour Quality Plots')

colour_plotting_defaults()

message_box('Plotting "F2" illuminant "Colour Rendering Index (CRI)".')
single_spd_colour_rendering_index_bars_plot(
    colour.ILLUMINANTS_RELATIVE_SPDS['F2'])

print('\n')

message_box(('Plotting various illuminants and light sources '
             '"Colour Rendering Index (CRI)".'))
multi_spd_colour_rendering_index_bars_plot((
    colour.ILLUMINANTS_RELATIVE_SPDS['F2'],
    colour.LIGHT_SOURCES_RELATIVE_SPDS['F32T8/TL841 (Triphosphor)'],
    colour.LIGHT_SOURCES_RELATIVE_SPDS['Kinoton 75P']))

print('\n')

message_box('Plotting "F2" illuminant "Colour Quality Scale (CQS)".')
single_spd_colour_quality_scale_bars_plot(
    colour.ILLUMINANTS_RELATIVE_SPDS['F2'])

print('\n')

message_box(('Plotting various illuminants and light sources '
             '"Colour Quality Scale (CQS)".'))
multi_spd_colour_quality_scale_bars_plot((
    colour.ILLUMINANTS_RELATIVE_SPDS['F2'],
    colour.LIGHT_SOURCES_RELATIVE_SPDS['F32T8/TL841 (Triphosphor)'],
    colour.LIGHT_SOURCES_RELATIVE_SPDS['Kinoton 75P']))
