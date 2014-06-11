#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *derivation* related examples.
"""

from numpy import matrix
import color

# Retrieving *normalized primary matrix* from *ACES RGB* colorspace to *CIE XYZ* colorspace.
print(color.get_normalized_primary_matrix(matrix([[0.73470, 0.26530],
                                                  [0.00000, 1.00000],
                                                  [0.00010, -0.07700]]),
                                          (0.32168, 0.33767)))

# Retrieving *normalized primary matrix* from *ACES RGB* colorspace to *CIE XYZ* colorspace.
print(color.get_normalized_primary_matrix(color.ACES_RGB_COLORSPACE.primaries, color.ACES_RGB_COLORSPACE.whitepoint))

# Retrieving *normalized primary matrix* from *CIE XYZ* colorspace to *ACES RGB* colorspace.
print(color.get_normalized_primary_matrix(color.ACES_RGB_COLORSPACE.primaries,
                                          color.ACES_RGB_COLORSPACE.whitepoint).getI())
