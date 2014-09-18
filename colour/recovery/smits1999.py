#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smits (1999) - Reflectance Recovery
===================================

Defines objects for reflectance recovery using *Brian Smits (1999)* method.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import zeros_spd
from colour.recovery import SMITS_1999_SPDS

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_to_spd_smits1999']


def RGB_to_spd_smits1999(RGB):
    white_spd = SMITS_1999_SPDS.get('white').clone()
    cyan_spd = SMITS_1999_SPDS.get('cyan').clone()
    magenta_spd = SMITS_1999_SPDS.get('magenta').clone()
    yellow_spd = SMITS_1999_SPDS.get('yellow').clone()
    red_spd = SMITS_1999_SPDS.get('red').clone()
    green_spd = SMITS_1999_SPDS.get('green').clone()
    blue_spd = SMITS_1999_SPDS.get('blue').clone()

    R, G, B = np.ravel(RGB)
    spd = zeros_spd(SMITS_1999_SPDS.get('white').shape)

    if R <= G and R <= B:
        spd += white_spd * R
        if G <= B:
            spd += cyan_spd * (G - R)
            spd += blue_spd * (B - G)
        else:
            spd += cyan_spd * (B - R)
            spd += green_spd * (G - B)
    elif G <= R and G <= B:
        spd += white_spd * G
        if R <= B:
            spd += magenta_spd * (R - G)
            spd += blue_spd * (B - R)
        else:
            spd += magenta_spd * (B - G)
            spd += red_spd * (R - B)
    else:
        spd += white_spd * B
        if R <= G:
            spd += yellow_spd * (R - B)
            spd += green_spd * (G - R)
        else:
            spd += yellow_spd * (G - B)
            spd += red_spd * (R - G)
    return spd
