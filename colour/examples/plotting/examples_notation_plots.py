#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour notation system plotting examples.
"""

from colour.plotting import *

# Plotting a single *Munsell* value function.
single_munsell_value_function_plot('Munsell Value Ladd 1955')

# Plotting multiple *Munsell* value functions.
multi_munsell_value_function_plot(['Munsell Value Ladd 1955',
                                   'Munsell Value Saunderson 1944'])