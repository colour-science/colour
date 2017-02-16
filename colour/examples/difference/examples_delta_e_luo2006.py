#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Delta E* colour difference computation objects based on
*Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, and *CAM02-UCS* colourspaces.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Delta E - Luo et al. (2006)" Computations')

Jpapbp_1 = (54.90433134, -0.08450395, -0.06854831)
Jpapbp_2 = (54.90433134, -0.08442362, -0.06848314)
message_box(('Computing "Delta E" with "Luo et al. (2006)" "CAM02-LCD" method '
             'from given "J\'a\'b\'" arrays:\n'
             '\n\t{0}\n\t{1}'.format(Jpapbp_1, Jpapbp_2)))
print(colour.delta_E_CAM02LCD(Jpapbp_1, Jpapbp_2))

print('\n')

message_box(('Computing "Delta E" with "Luo et al. (2006)" "CAM02-SCD" method '
             'from given "J\'a\'b\'" arrays:\n'
             '\n\t{0}\n\t{1}'.format(Jpapbp_1, Jpapbp_2)))
print(colour.delta_E_CAM02SCD(Jpapbp_1, Jpapbp_2))

print('\n')

message_box(('Computing "Delta E" with "Luo et al. (2006)" "CAM02-UCS" method '
             'from given "J\'a\'b\'" arrays:\n'
             '\n\t{0}\n\t{1}'.format(Jpapbp_1, Jpapbp_2)))
print(colour.delta_E_CAM02UCS(Jpapbp_1, Jpapbp_2))
