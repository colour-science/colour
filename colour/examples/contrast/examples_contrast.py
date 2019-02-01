# -*- coding: utf-8 -*-
"""
Showcases contrast sensitivity computations.
"""

from pprint import pprint

import numpy as np

import colour
from colour.utilities import message_box
from colour.plotting import colour_style, plot_single_function

message_box('Contrast Sensitivity Computations')

colour_style()

message_box(('Computing the contrast sensitivity for a spatial frequency "u" '
             'of 4, an angular size "X_0" of 60 and a retinal illuminance "E" '
             'of 65 using "Barten (1999)" method.'))
pprint(colour.function_contrast_sensitivity(u=4, X_0=60, E=65))
pprint(
    colour.contrast.function_contrast_sensitivity_Barten1999(
        u=4, X_0=60, E=65))

print('\n')

message_box(
    ('Computing the minimum detectable contrast with the assumed '
     'conditions for UHDTV applications using "Barten (1999)" method.'))

L = np.linspace(0.01, 100, 10000)
X_0 = Y_0 = 60
d = colour.contrast.barten1999.pupil_diameter_Barten1999(L, X_0, Y_0)
sigma = colour.contrast.barten1999.sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
E = colour.contrast.barten1999.retinal_illuminance_Barten1999(L, d)
u = X_0 / 15

pprint(1 / colour.function_contrast_sensitivity(
    u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0) * 2 * (1 / 1.27))
pprint(1 / colour.contrast.function_contrast_sensitivity_Barten1999(
    u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0) * 2 * (1 / 1.27))

plot_single_function(
    lambda x: 1 / colour.contrast.function_contrast_sensitivity_Barten1999(
        u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0) * 2 * (1 / 1.27),
    samples=L,
    log_x=10,
    log_y=10,
    bounding_box=[0.1, 100, 0.001, 0.1],
    **{
        'title':
            'Examples of HVS Minimum Detectable Contrast Characteristics',
        'x_label':
            'Luminance ($cd/m^2$)',
        'y_label':
            'Minimum Detectable Contrast',
        'axes.grid.which':
            'both'
    })
