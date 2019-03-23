# -*- coding: utf-8 -*-
"""
Showcases common plotting examples.
"""

from colour.plotting import (ColourSwatch, colour_style,
                             plot_multi_colour_swatches,
                             plot_single_colour_swatch)
from colour.utilities import message_box

message_box('Common Plots')

colour_style()

message_box('Plotting a single colour.')
plot_single_colour_swatch(
    ColourSwatch(
        'Neutral 5 (.70 D)', RGB=(0.32315746, 0.32983556, 0.33640183)),
    text_size=32)

print('\n')

message_box('Plotting multiple colours.')
plot_multi_colour_swatches(
    (ColourSwatch('Dark Skin', RGB=(0.45293517, 0.31732158, 0.26414773)),
     ColourSwatch('Light Skin', RGB=(0.77875824, 0.57726450, 0.50453169))),
    text_size=32)
