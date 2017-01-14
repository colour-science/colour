#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases RGB colourspace volume computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('RGB Colourspace Volume Computations')

message_box('Computing "ProPhoto RGB" RGB colourspace limits.')
limits = colour.RGB_colourspace_limits(colour.PROPHOTO_RGB_COLOURSPACE)
print(limits)

print('\n')

samples = 10e4
message_box(('Computing "ProPhoto RGB" RGB colourspace volume using '
             '{0} samples.'.format(samples)))
print(colour.RGB_colourspace_volume_MonteCarlo(
    colour.PROPHOTO_RGB_COLOURSPACE,
    samples=samples,
    limits=limits * 1.1))

print('\n')

message_box(('Computing "ProPhoto RGB" RGB colourspace coverage of Pointer\'s '
             'Gamut using {0} samples.'.format(samples)))
print(colour.RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
    colour.PROPHOTO_RGB_COLOURSPACE,
    samples=samples))
