#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases corresponding chromaticities prediction plotting examples.
"""

from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Corresponding Chromaticities Prediction Plots')

colour_plotting_defaults()

message_box('Plotting corresponding chromaticities prediction with '
            '"Von Kries" chromatic adaptation model for "Breneman (1987)" '
            'experiment number "2" using "Bianco" CAT.')
corresponding_chromaticities_prediction_plot(2, 'Von Kries', 'Bianco')

print('\n')

message_box('Plotting corresponding chromaticities prediction with '
            '"CMCCAT200" chromatic adaptation model for "Breneman (1987)" '
            'experiment number "4" using "Bianco" CAT.')
corresponding_chromaticities_prediction_plot(4, 'CMCCAT2000')
