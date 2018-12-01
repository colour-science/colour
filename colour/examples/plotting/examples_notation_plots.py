# -*- coding: utf-8 -*-
"""
Showcases colour notation systems plotting examples.
"""

from colour.plotting import (colour_style, multi_munsell_value_function_plot,
                             plot_single_munsell_value_function)
from colour.utilities import message_box

message_box('Colour Notation Systems Plots')

colour_style()

message_box('Plotting a single "Munsell" value function.')
plot_single_munsell_value_function('Ladd 1955')

print('\n')

message_box('Plotting multiple "Munsell" value functions.')
multi_munsell_value_function_plot(['Ladd 1955', 'Saunderson 1944'])
