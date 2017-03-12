#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases characterisation plotting examples.
"""

from pprint import pprint

import colour
from colour.characterisation.dataset.colour_checkers.spds import (
    COLOURCHECKER_INDEXES_TO_NAMES_MAPPING)
from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Characterisation Plots')

colour_plotting_defaults()

message_box('Plotting colour rendition charts.')
pprint(sorted(colour.COLOURCHECKERS.keys()))
colour_checker_plot('ColorChecker 1976')
colour_checker_plot('BabelColor Average', text_display=False)
colour_checker_plot('ColorChecker 1976', text_display=False)
colour_checker_plot('ColorChecker 2005', text_display=False)

print('\n')

message_box(('Plotting "BabelColor Average" colour rendition charts spectral '
             'power distributions.'))
multi_spd_plot([colour.COLOURCHECKERS_SPDS['BabelColor Average'][value]
                for key, value in
                sorted(COLOURCHECKER_INDEXES_TO_NAMES_MAPPING.items())],
               use_spds_colours=True,
               title=('BabelColor Average - '
                      'Relative Spectral Power Distributions'))
