# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package colour *difference* related examples.
"""

from numpy import array
import colour

# Retrieving *Delta E CIE 1976* colour.colorimetry.difference, *CIE Lab*
# colourspace colors are expected as input.
print(colour.delta_E_CIE_1976(
    array([100., 21.57210357, 272.2281935]).reshape((3, 1)),
    array([100., 426.67945353, 72.39590835]).reshape((3, 1))))
# Using simplified syntax form.
print(colour.delta_E_CIE_1976([100., 21.57210357, 272.2281935],
                              [100., 426.67945353, 72.39590835]))

# Retrieving *Delta E CIE 1994* colour.colorimetry.difference, *CIE Lab*
# colourspace colors are expected as input.
print(colour.delta_E_CIE_1994(
    array([100., 21.57210357, 272.2281935]).reshape((3, 1)),
    array([100., 426.67945353, 72.39590835]).reshape((3, 1))))

# Retrieving *Delta E CIE 1994* colour.colorimetry.difference for
# *graphics arts* applications.
print(colour.delta_E_CIE_1994(
    array([100., 21.57210357, 272.2281935]).reshape((3, 1)),
    array([100., 426.67945353, 72.39590835]).reshape((3, 1)),
    textiles=False))

# Retrieving *Delta E CIE 2000* colour.colorimetry.difference, *CIE Lab*
# colourspace colors are expected as input.
print(colour.delta_E_CIE_2000(
    array([100., 21.57210357, 272.2281935]).reshape((3, 1)),
    array([100., 426.67945353, 72.39590835]).reshape((3, 1))))

# Retrieving *Delta E CMC* colour.colorimetry.difference, *CIE Lab*
# colourspace colors are expected as input.
print(    colour.delta_E_CMC(
    array([100., 21.57210357, 272.2281935]).reshape((3, 1)),
    array([100., 426.67945353, 72.39590835]).reshape((3, 1))))

# Retrieving *Delta E CMC* colour.colorimetry.difference with imperceptibility
# threshold.
print(    colour.delta_E_CMC(
    array([100., 21.57210357, 272.2281935]).reshape((3, 1)),
    array([100., 426.67945353, 72.39590835]).reshape((3, 1)),
    l=1.))
