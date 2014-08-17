#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases characterization plotting examples.
"""

import colour
from colour.characterization.dataset.colour_checkers.spds import (
    COLOURCHECKER_INDEXES_TO_NAMES_MAPPING)
from colour.plotting import *

# Plotting colour checkers.
print(sorted(colour.COLOURCHECKERS.keys()))
colour_checker_plot('ColorChecker 1976')
colour_checker_plot('BabelColor Average', text_display=False)
colour_checker_plot('ColorChecker 1976', text_display=False)
colour_checker_plot('ColorChecker 2005', text_display=False)

# Plotting multiple *ColorChecker* relative spectral power distributions.
multi_spd_plot([colour.COLOURCHECKERS_SPDS.get('BabelColor Average').get(value)
                for key, value in
                sorted(COLOURCHECKER_INDEXES_TO_NAMES_MAPPING.items())],
               use_spds_colours=True,
               title='BabelColor Average - Relative Spectral Power Distributions')