#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases reflectance recovery computations using *Brian Smits (1999)* method.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Brian Smits (1999)" - Reflectance Recovery Computations')

RGB = [0.35505307, 0.47995567, 0.61088035]
message_box(('Recovering reflectance using "Brian Smits (1999)" method from '
             'given "RGB" colourspace matrix:\n'
             '\n\tRGB: {0}'.format(RGB)))
print(colour.RGB_to_spectral_smits1999(RGB))

print('\n')

message_box(('An analysis of "Brian Smits (1999)" method is available at the '
             'following url : '
             'http://nbviewer.ipython.org/github/colour-science/colour-website/blob/master/ipython/about_reflectance_recovery.ipynb'))  # noqa