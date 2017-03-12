#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour notation systems plotting examples.
"""

from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Colour Notation Systems Plots')

colour_plotting_defaults()

message_box('Plotting a single "Munsell" value function.')
single_munsell_value_function_plot('Ladd 1955')

print('\n')

message_box('Plotting multiple "Munsell" value functions.')
multi_munsell_value_function_plot(['Ladd 1955',
                                   'Saunderson 1944'])
