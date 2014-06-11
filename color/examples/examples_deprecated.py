#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *deprecated* objects related examples.
"""

import color.deprecated

# Converting from *RGB* colorspace to *HSV* colorspace.
print(color.deprecated.RGB_to_HSV([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]))

# Converting from *HSV* colorspace to *RGB* colorspace.
print(color.deprecated.HSV_to_RGB([0.27867384, 0.744, 0.98039216]))

# Converting from *RGB* colorspace to *HSL* colorspace.
print(color.deprecated.RGB_to_HSL([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]))

# Converting from *HSL* colorspace to *RGB* colorspace.
print(color.deprecated.HSL_to_RGB([0.27867384, 0.94897959, 0.61568627]))

# Converting from *RGB* colorspace to *CMY* colorspace.
print(color.deprecated.RGB_to_CMY([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]))

# Converting from *CMY* colorspace to *RGB* colorspace.
print(color.deprecated.CMY_to_RGB([0.50980392, 0.01960784, 0.74901961]))

# Converting from *CMY* colorspace to *CMYK* colorspace.
print(color.deprecated.CMY_to_CMYK([0.50980392, 0.01960784, 0.74901961]))

# Converting from *CMYK* colorspace to *CMY* colorspace.
print(color.deprecated.CMYK_to_CMY([0.5, 0., 0.744, 0.01960784]))

# Converting from *RGB* colorspace to hex triplet representation.
print(color.deprecated.RGB_to_HEX([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]))

# Converting from hex triplet representation to *RGB* colorspace.
print(color.deprecated.HEX_to_RGB("#7dfa40"))