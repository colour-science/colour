#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Showcases Machado (2009) simulation of colour vision deficiency.
"""

from __future__ import division, unicode_literals

import numpy as np

import colour
from colour.utilities.verbose import message_box

message_box('Simulation of CVD - Machado (2009)')

M_a = colour.anomalous_trichromacy_matrix_Machado2009(
    colour.LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals'),
    colour.DISPLAYS_RGB_PRIMARIES['Typical CRT Brainard 1997'],
    np.array([10, 0, 0]))
message_box(('Computing a "Protanomaly" matrix using '
             '"Stockman & Sharpe 2 Degree Cone Fundamentals" and '
             '"Typical CRT Brainard 1997" "RGB" display primaries for a 10nm '
             'shift:\n\n'
             '{0}'.format(M_a)))

print('\n')

M_a = colour.cvd_matrix_Machado2009('Protanomaly', 0.5)
message_box(('Retrieving a "Protanomaly" pre-computed matrix for a 50% '
             'severity:\n\n'
             '{0}'.format(M_a)))

print('\n')

M_a = colour.anomalous_trichromacy_matrix_Machado2009(
    colour.LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals'),
    colour.DISPLAYS_RGB_PRIMARIES['Typical CRT Brainard 1997'],
    np.array([0, 10, 0]))
message_box(('Computing a "Deuteranomaly" matrix using '
             '"Stockman & Sharpe 2 Degree Cone Fundamentals" and '
             '"Typical CRT Brainard 1997" "RGB" display primaries for a 10nm '
             'shift:\n\n'
             '{0}'.format(M_a)))

print('\n')

M_a = colour.cvd_matrix_Machado2009('Deuteranomaly', 0.5)
message_box(('Retrieving a "Deuteranomaly" pre-computed matrix for a 50% '
             'severity:\n\n'
             '{0}'.format(M_a)))

print('\n')

M_a = colour.anomalous_trichromacy_matrix_Machado2009(
    colour.LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals'),
    colour.DISPLAYS_RGB_PRIMARIES['Typical CRT Brainard 1997'],
    np.array([0, 0, 27]))
message_box(('Computing a "Tritanomaly" matrix using '
             '"Stockman & Sharpe 2 Degree Cone Fundamentals" and '
             '"Typical CRT Brainard 1997" "RGB" display primaries for a 27nm '
             'shift:\n\n'
             '{0}'.format(M_a)))

print('\n')

M_a = colour.cvd_matrix_Machado2009('Tritanomaly', 0.5)
message_box(('Retrieving a "Tritanomaly" pre-computed matrix for a 50% '
             'severity:\n\n'
             '{0}'.format(M_a)))
