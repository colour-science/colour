#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package *colorspaces* related examples.
"""

from numpy import matrix
import color

# Displaying :attr:`color.colorspaces.COLORSPACES` data.
colorspace = color.COLORSPACES["ACES RGB"]

print("Name: '{0}'".format(colorspace.name))
print("Primaries: '{0}'".format(colorspace.primaries))
print("Normalized primary matrix to 'CIE XYZ': '{0}'".format(colorspace.toXYZ))
print("Normalized primary matrix from 'CIE XYZ': '{0}'".format(colorspace.fromXYZ))
print("Transfer function: '{0}'".format(colorspace.transferFunction))
print("Inverse transfer function: '{0}'".format(colorspace.inverseTransferFunction))

# Calculating *ACES RGB* to *sRGB* transformation matrix.
print("'ACES RGB' colorspace to 'sRGB' colorspace matrix:")
cat = color.getChromaticAdaptationMatrix(color.xy_to_XYZ(color.ACES_RGB_WHITEPOINT),
										 color.xy_to_XYZ(color.sRGB_WHITEPOINT))
print color.XYZ_TO_sRGB_MATRIX * (cat * color.ACES_RGB_TO_XYZ_MATRIX)
