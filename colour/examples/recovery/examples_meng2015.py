#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Showcases reflectance recovery computations using *Meng et al. (2015)* method.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Meng et al. (2015)" - Reflectance Recovery Computations')

XYZ = (1.14176346, 1.00000000, 0.49815206)
message_box(('Recovering reflectance using "Meng et al. (2015)" method from '
             'given "XYZ" tristimulus values:\n'
             '\n\tXYZ: {0}'.format(XYZ)))
print(colour.XYZ_to_spectral_Meng2015(XYZ))
