# -*- coding: utf-8 -*-
"""
Showcases reflectance recovery computations using *Otsu et al. (2018)* method.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Otsu et al. (2018)" - Reflectance Recovery Computations')

illuminant = colour.SDS_ILLUMINANTS['D65']

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box('Recovering reflectance using "Otsu et al. (2018)" method from '
            'given "XYZ" tristimulus values:\n'
            '\n\tXYZ: {0}'.format(XYZ))
sd = colour.XYZ_to_sd(XYZ, method='Otsu 2018')
print(sd)
print(colour.recovery.XYZ_to_sd_Otsu2018(XYZ))
print(colour.sd_to_XYZ(sd, illuminant=illuminant) / 100)

print('\n')

message_box('Generating a spectral dataset according to "Otsu et al. (2018)"'
            'method :')
XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
reflectances = [
    reflectance.copy().align(colour.recovery.SPECTRAL_SHAPE_OTSU2018).values
    for reflectance in colour.SDS_COLOURCHECKERS['ColorChecker N Ohta']
    .values()
]
node_tree = colour.recovery.NodeTree_Otsu2018(reflectances)
node_tree.optimise()
dataset = node_tree.to_dataset()
print(colour.recovery.XYZ_to_sd_Otsu2018(XYZ, dataset=dataset))
