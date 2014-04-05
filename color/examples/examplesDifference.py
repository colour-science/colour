#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package color *difference* related examples.
"""

from numpy import matrix
import color

# Retrieving *Delta E CIE 1976* color difference, *CIE Lab* colorspace colors are expected as input.
print(color.delta_E_CIE_1976(matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
							 matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))))
# Using simplified syntax form.
print(color.delta_E_CIE_1976([100., 21.57210357, 272.2281935], [100., 426.67945353, 72.39590835]))

# Retrieving *Delta E CIE 1994* color difference, *CIE Lab* colorspace colors are expected as input.
print(color.delta_E_CIE_1994(matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
							 matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))))

# Retrieving *Delta E CIE 1994* color difference for *graphics arts* applications.
print(color.delta_E_CIE_1994(matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
							 matrix([100., 426.67945353, 72.39590835]).reshape((3, 1)),
							 textiles=False))

# Retrieving *Delta E CIE 2000* color difference, *CIE Lab* colorspace colors are expected as input.
print(color.delta_E_CIE_2000(matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
							 matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))))

# Retrieving *Delta E CMC* color difference, *CIE Lab* colorspace colors are expected as input.
print(color.delta_E_CMC(matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
						matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))))

# Retrieving *Delta E CMC* color difference with imperceptibility threshold.
print(color.delta_E_CMC(matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
						matrix([100., 426.67945353, 72.39590835]).reshape((3, 1)),
						l=1.))
