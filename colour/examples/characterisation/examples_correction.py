# -*- coding: utf-8 -*-
"""
Showcases colour correction computations.
"""

import numpy as np
import colour
from colour.utilities import message_box

message_box('Colour Correction Computations')

M_T = np.array([
    [0.17290600, 0.08205715, 0.05711951],
    [0.56807350, 0.29250361, 0.21942000],
    [0.10437166, 0.19656122, 0.32946697],
    [0.10089156, 0.14839029, 0.05324779],
    [0.22306044, 0.21697008, 0.43151034],
    [0.10718115, 0.51351274, 0.41399101],
    [0.74644090, 0.20020391, 0.03077999],
    [0.05948985, 0.10659048, 0.39884205],
    [0.56735781, 0.08485298, 0.11940265],
    [0.11178198, 0.04285385, 0.14161263],
    [0.34254479, 0.50627811, 0.05571580],
    [0.79268226, 0.35803827, 0.02544159],
    [0.01865226, 0.05139666, 0.28876921],
    [0.05440562, 0.29876767, 0.07183236],
    [0.45631278, 0.03075616, 0.04089930],
    [0.85385852, 0.56503529, 0.01470045],
    [0.53537579, 0.09006281, 0.30467248],
    [-0.03661893, 0.24753827, 0.39810356],
    [0.91186287, 0.91497635, 0.89391370],
    [0.57979860, 0.59203200, 0.59346914],
    [0.35499180, 0.36538033, 0.36757315],
    [0.19011528, 0.19180135, 0.19309001],
    [0.08525591, 0.08890588, 0.09252104],
    [0.03039192, 0.03118624, 0.03278316],
])

M_R = np.array([
    [0.15579559, 0.09715755, 0.07514556],
    [0.39113140, 0.25943419, 0.21266708],
    [0.12824821, 0.18463570, 0.31508023],
    [0.12028974, 0.13455659, 0.07408400],
    [0.19368988, 0.21158946, 0.37955964],
    [0.19957425, 0.36085439, 0.40678123],
    [0.48896605, 0.20691688, 0.05816533],
    [0.09775522, 0.16710693, 0.47147724],
    [0.39358649, 0.12233400, 0.10526425],
    [0.10780332, 0.07258529, 0.16151473],
    [0.27502671, 0.34705454, 0.09728099],
    [0.43980441, 0.26880559, 0.05430533],
    [0.05887212, 0.11126272, 0.38552469],
    [0.12705825, 0.25787860, 0.13566464],
    [0.35612929, 0.07933258, 0.05118732],
    [0.48131976, 0.42082843, 0.07120612],
    [0.34665585, 0.15170714, 0.24969804],
    [0.08261116, 0.24588716, 0.48707733],
    [0.66054904, 0.65941137, 0.66376412],
    [0.48051509, 0.47870296, 0.48230082],
    [0.33045354, 0.32904184, 0.33228886],
    [0.18001305, 0.17978567, 0.18004416],
    [0.10283975, 0.10424680, 0.10384975],
    [0.04742204, 0.04772203, 0.04914226],
])

message_box(
    ('Computing the colour correction matrix correcting a "M_T" "ColorChecker"'
     'colour rendition chart to a "M_R" one using '
     '"Cheung, Westland, Connah and Ripamonti (2004)" method '
     'with 3 terms polynomial.'))

print(colour.characterisation.colour_correction_matrix_Cheung2004(M_T, M_R))
print(colour.colour_correction_matrix(M_T, M_R, method='Cheung 2004'))

print('\n')

message_box(
    ('Computing the colour correction matrix correcting a "M_T" "ColorChecker"'
     'colour rendition chart to a "M_R" one using '
     '"Cheung, Westland, Connah and Ripamonti (2004)" method '
     'with 7 terms polynomial.'))

print(colour.characterisation.colour_correction_matrix_Cheung2004(M_T, M_R, 7))
print(colour.colour_correction_matrix(M_T, M_R, method='Cheung 2004', terms=7))

print('\n')

message_box(
    ('Computing the colour correction matrix correcting a "M_T" "ColorChecker"'
     'colour rendition chart to a "M_R" one using '
     '"Finlayson, MacKiewicz and Hurlbert (2015)" method '
     'with polynomial of degree 1.'))

print(colour.characterisation.colour_correction_matrix_Finlayson2015(M_T, M_R))
print(colour.colour_correction_matrix(M_T, M_R, method='Finlayson 2015'))

print('\n')

message_box(
    ('Computing the colour correction matrix correcting a "M_T" "ColorChecker"'
     'colour rendition chart to a "M_R" one using '
     '"Finlayson, MacKiewicz and Hurlbert (2015)" method '
     'with polynomial of degree 3.'))

print(
    colour.characterisation.colour_correction_matrix_Finlayson2015(
        M_T, M_R, 3))
print(
    colour.colour_correction_matrix(
        M_T, M_R, method='Finlayson 2015', degree=3))

print('\n')

message_box(
    ('Computing the colour correction matrix correcting a "M_T" "ColorChecker"'
     'colour rendition chart to a "M_R" one using '
     '"Vandermonde" method with polynomial of degree 1.'))

print(colour.characterisation.colour_correction_matrix_Vandermonde(M_T, M_R))
print(colour.colour_correction_matrix(M_T, M_R, method='Vandermonde'))

print('\n')

message_box(
    ('Computing the colour correction matrix correcting a "M_T" "ColorChecker"'
     'colour rendition chart to a "M_R" one using '
     '"Vandermonde" method with polynomial of degree 3.'))

print(
    colour.characterisation.colour_correction_matrix_Vandermonde(M_T, M_R, 3))
print(
    colour.colour_correction_matrix(M_T, M_R, method='Vandermonde', degree=3))

print('\n')

RGB = np.array([0.17224810, 0.09170660, 0.06416938])

message_box(
    ('Colour correct given "RGB" colourspace array with matrix mapping a '
     '"M_T" "ColorChecker" colour rendition chart to a "M_R" one using '
     '"Cheung, Westland, Connah and Ripamonti (2004)" method with 3 terms '
     'polynomial.'))

print(colour.characterisation.colour_correction_Cheung2004(RGB, M_T, M_R))
print(colour.colour_correction(RGB, M_T, M_R, method='Cheung 2004'))

print('\n')

message_box(
    ('Colour correct given "RGB" colourspace array with matrix mapping a '
     '"M_T" "ColorChecker" colour rendition chart to a "M_R" one using '
     '"Cheung, Westland, Connah and Ripamonti (2004)" method with 7 terms '
     'polynomial.'))

print(colour.characterisation.colour_correction_Cheung2004(RGB, M_T, M_R, 7))
print(colour.colour_correction(RGB, M_T, M_R, method='Cheung 2004', terms=7))

print('\n')

message_box(
    ('Colour correct given "RGB" colourspace array with matrix mapping a '
     '"M_T" "ColorChecker" colour rendition chart to a "M_R" one using '
     '"Finlayson, MacKiewicz and Hurlbert (2015)" method with polynomial of '
     'degree 1.'))

print(colour.characterisation.colour_correction_Finlayson2015(RGB, M_T, M_R))
print(colour.colour_correction(RGB, M_T, M_R, method='Finlayson 2015'))

print('\n')

message_box(
    ('Colour correct given "RGB" colourspace array with matrix mapping a '
     '"M_T" "ColorChecker" colour rendition chart to a "M_R" one using '
     '"Finlayson, MacKiewicz and Hurlbert (2015)" method with polynomial of '
     'degree 3.'))

print(
    colour.characterisation.colour_correction_Finlayson2015(RGB, M_T, M_R, 3))
print(
    colour.colour_correction(RGB, M_T, M_R, method='Finlayson 2015', degree=3))

print('\n')

message_box(
    ('Colour correct given "RGB" colourspace array with matrix mapping a '
     '"M_T" "ColorChecker" colour rendition chart to a "M_R" one using '
     '"Vandermonde" method with polynomial of degree 1.'))

print(colour.characterisation.colour_correction_Vandermonde(RGB, M_T, M_R))
print(colour.colour_correction(RGB, M_T, M_R, method='Vandermonde'))

print('\n')

message_box(
    ('Colour correct given "RGB" colourspace array with matrix mapping a '
     '"M_T" "ColorChecker" colour rendition chart to a "M_R" one using '
     '"Vandermonde" method with polynomial of degree 3.'))
print(colour.characterisation.colour_correction_Vandermonde(RGB, M_T, M_R, 3))
print(colour.colour_correction(RGB, M_T, M_R, method='Vandermonde', degree=3))
