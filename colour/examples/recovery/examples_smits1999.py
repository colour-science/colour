#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases reflectance recovery computations using *Smits (1999)* method.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Smits (1999)" - Reflectance Recovery Computations')

RGB = (0.35505307, 0.47995567, 0.61088035)
message_box(('Recovering reflectance using "Smits (1999)" method from '
             'given "RGB" colourspace array:\n'
             '\n\tRGB: {0}'.format(RGB)))
print(colour.RGB_to_spectral_Smits1999(RGB))

print('\n')

message_box((
    'An analysis of "Smits (1999)" method is available at the '
    'following url : '
    'http://nbviewer.jupyter.org/github/colour-science/colour-website/'
    'blob/master/ipython/about_reflectance_recovery.ipynb'))
