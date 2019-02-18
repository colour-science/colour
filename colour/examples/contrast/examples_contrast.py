# -*- coding: utf-8 -*-
"""
Showcases contrast sensitivity computations.
"""

from pprint import pprint

import numpy as np
from scipy.optimize import fmin

import colour
from colour.utilities import as_float, message_box
from colour.plotting import colour_style, plot_single_function

message_box('Contrast Sensitivity Computations')

colour_style()

message_box(('Computing the contrast sensitivity for a spatial frequency "u" '
             'of 4, an angular size "X_0" of 60 and a retinal illuminance "E" '
             'of 65 using "Barten (1999)" method.'))
pprint(colour.contrast_sensitivity_function(u=4, X_0=60, E=65))
pprint(
    colour.contrast.contrast_sensitivity_function_Barten1999(
        u=4, X_0=60, E=65))

print('\n')

message_box(('Computing the minimum detectable contrast with the assumed '
             'conditions for UHDTV applications as given in "ITU-R BT.2246-4"'
             '"Figure 31" and using "Barten (1999)" method.'))

settings_BT2246 = {
    'k': 3.0,
    'T': 0.1,
    'X_max': 12,
    'N_max': 15,
    'n': 0.03,
    'p': 1.2274 * 10 ** 6,
    'phi_0': 3 * 10 ** -8,
    'u_0': 7,
}


def maximise_spatial_frequency(L):
    """
    Maximises the spatial frequency :math:`u` for given luminance value.

    Parameters
    ----------
    L : numeric or array_like
        Luminance value at which to maximize the spatial frequency :math:`u`.

    Returns
    -------
    numeric or ndarray
        Maximised spatial frequency :math:`u`.
    """

    maximised_spatial_frequency = []
    for L_v in L:
        X_0 = 60
        d = colour.contrast.pupil_diameter_Barten1999(L_v, X_0)
        sigma = colour.contrast.sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
        E = colour.contrast.retinal_illuminance_Barten1999(L_v, d, True)
        maximised_spatial_frequency.append(
            fmin(lambda x: (
                    -colour.contrast.contrast_sensitivity_function_Barten1999(
                        u=x,
                        sigma=sigma,
                        X_0=X_0,
                        E=E,
                        **settings_BT2246)
                ), 0, disp=False)[0])

    return as_float(np.array(maximised_spatial_frequency))


L = np.logspace(np.log10(0.01), np.log10(100), 100)
X_0 = Y_0 = 60
d = colour.contrast.barten1999.pupil_diameter_Barten1999(L, X_0, Y_0)
sigma = colour.contrast.barten1999.sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
E = colour.contrast.barten1999.retinal_illuminance_Barten1999(L, d)
u = maximise_spatial_frequency(L)

pprint(1 / colour.contrast_sensitivity_function(
    u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0, **settings_BT2246) * 2 *
       (1 / 1.27))
pprint(1 / colour.contrast.contrast_sensitivity_function_Barten1999(
    u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0, **settings_BT2246) * 2 *
       (1 / 1.27))

plot_single_function(
    lambda x: (
        1 / colour.contrast.contrast_sensitivity_function_Barten1999(
            u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0, **settings_BT2246)
        * 2 * (1 / 1.27)),
    samples=L,
    log_x=10,
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
