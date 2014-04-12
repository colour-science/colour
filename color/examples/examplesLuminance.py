#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package *luminance* related examples.
"""

from numpy import matrix
import color

# Retrieving luminance of given *RGB* components.
print color.getLuminance(matrix([56., 16., 100.]).reshape((3, 1)),
						 color.sRGB_COLORSPACE.primaries,
						 color.sRGB_COLORSPACE.whitepoint)