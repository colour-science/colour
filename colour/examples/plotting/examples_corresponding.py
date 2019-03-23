# -*- coding: utf-8 -*-
"""
Showcases corresponding chromaticities prediction plotting examples.
"""

from colour.plotting import (colour_style,
                             plot_corresponding_chromaticities_prediction)
from colour.utilities import message_box

message_box('Corresponding Chromaticities Prediction Plots')

colour_style()

message_box('Plotting corresponding chromaticities prediction with '
            '"Von Kries" chromatic adaptation model for "Breneman (1987)" '
            'experiment number "2" using "Bianco" CAT.')
plot_corresponding_chromaticities_prediction(2, 'Von Kries', 'Bianco')

print('\n')

message_box('Plotting corresponding chromaticities prediction with '
            '"CMCCAT200" chromatic adaptation model for "Breneman (1987)" '
            'experiment number "4" using "Bianco" CAT.')
plot_corresponding_chromaticities_prediction(4, 'CMCCAT2000')
