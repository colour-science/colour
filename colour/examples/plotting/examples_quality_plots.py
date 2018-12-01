# -*- coding: utf-8 -*-
"""
Showcases colour quality plotting examples.
"""

import colour
from colour.plotting import (colour_style,
                             multi_spd_colour_quality_scale_bars_plot,
                             multi_spd_colour_rendering_index_bars_plot,
                             single_spd_colour_quality_scale_bars_plot,
                             plot_single_spd_colour_rendering_index_bars)
from colour.utilities import message_box

message_box('Colour Quality Plots')

colour_style()

message_box('Plotting "F2" illuminant "Colour Rendering Index (CRI)".')
plot_single_spd_colour_rendering_index_bars(colour.ILLUMINANTS_SPDS['F2'])

print('\n')

message_box(('Plotting various illuminants and light sources '
             '"Colour Rendering Index (CRI)".'))
multi_spd_colour_rendering_index_bars_plot(
    (colour.ILLUMINANTS_SPDS['F2'],
     colour.LIGHT_SOURCES_SPDS['F32T8/TL841 (Triphosphor)'],
     colour.LIGHT_SOURCES_SPDS['Kinoton 75P']))

print('\n')

message_box('Plotting "F2" illuminant "Colour Quality Scale (CQS)".')
single_spd_colour_quality_scale_bars_plot(colour.ILLUMINANTS_SPDS['F2'])

print('\n')

message_box(('Plotting various illuminants and light sources '
             '"Colour Quality Scale (CQS)".'))
multi_spd_colour_quality_scale_bars_plot(
    (colour.ILLUMINANTS_SPDS['F2'],
     colour.LIGHT_SOURCES_SPDS['F32T8/TL841 (Triphosphor)'],
     colour.LIGHT_SOURCES_SPDS['Kinoton 75P']))
