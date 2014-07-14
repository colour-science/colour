# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package *deprecated* objects related examples.
"""

import colour.computation.colourspaces.deprecated

# Converting from *RGB* colourspace to *HSV* colourspace.
print(colour.computation.colourspaces.deprecated.RGB_to_HSV([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]))

# Converting from *HSV* colourspace to *RGB* colourspace.
print(colour.computation.colourspaces.deprecated.HSV_to_RGB([0.27867384, 0.744, 0.98039216]))

# Converting from *RGB* colourspace to *HSL* colourspace.
print(colour.computation.colourspaces.deprecated.RGB_to_HSL([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]))

# Converting from *HSL* colourspace to *RGB* colourspace.
print(colour.computation.colourspaces.deprecated.HSL_to_RGB([0.27867384, 0.94897959, 0.61568627]))

# Converting from *RGB* colourspace to *CMY* colourspace.
print(colour.computation.colourspaces.deprecated.RGB_to_CMY([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]))

# Converting from *CMY* colourspace to *RGB* colourspace.
print(colour.computation.colourspaces.deprecated.CMY_to_RGB([0.50980392, 0.01960784, 0.74901961]))

# Converting from *CMY* colourspace to *CMYK* colourspace.
print(colour.computation.colourspaces.deprecated.CMY_to_CMYK([0.50980392, 0.01960784, 0.74901961]))

# Converting from *CMYK* colourspace to *CMY* colourspace.
print(colour.computation.colourspaces.deprecated.CMYK_to_CMY([0.5, 0., 0.744, 0.01960784]))

# Converting from *RGB* colourspace to hex triplet representation.
print(colour.computation.colourspaces.deprecated.RGB_to_HEX([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]))

# Converting from hex triplet representation to *RGB* colourspace.
print(colour.computation.colourspaces.deprecated.HEX_to_RGB("#7dfa40"))