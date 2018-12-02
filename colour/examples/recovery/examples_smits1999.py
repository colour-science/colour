# -*- coding: utf-8 -*-
"""
Showcases reflectance recovery computations using *Smits (1999)* method.
"""

import numpy as np

import colour
from colour.recovery.smits1999 import XYZ_to_RGB_Smits1999
from colour.utilities import message_box

message_box('"Smits (1999)" - Reflectance Recovery Computations')

XYZ = np.array([1.14176346, 1.00000000, 0.49815206])
RGB = XYZ_to_RGB_Smits1999(XYZ)
message_box(('Recovering reflectance using "Smits (1999)" method from '
             'given "RGB" colourspace array:\n'
             '\n\tRGB: {0}'.format(RGB)))
print(colour.XYZ_to_sd(XYZ, method='Smits 1999'))
print(colour.recovery.RGB_to_sd_Smits1999(RGB))

print('\n')

message_box(
    ('An analysis of "Smits (1999)" method is available at the '
     'following url : '
     'http://nbviewer.jupyter.org/github/colour-science/colour-website/'
     'blob/master/ipython/about_reflectance_recovery.ipynb'))
