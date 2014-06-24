#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *colorspaces* related examples.
"""

import pprint
import color

# Displaying :attr:`color.COLORSPACES` data.
pprint.pprint(sorted(color.COLORSPACES.keys()))

colorspace = color.COLORSPACES["ACES RGB"]
print("Name: '{0}'".format(colorspace.name))
print("Primaries: '{0}'".format(colorspace.primaries))
print("Normalized primary matrix to 'CIE XYZ': '{0}'".format(colorspace.to_XYZ))
print("Normalized primary matrix from 'CIE XYZ': '{0}'".format(colorspace.from_XYZ))
print("Transfer function: '{0}'".format(colorspace.transfer_function))
print("Inverse transfer function: '{0}'".format(colorspace.inverse_transfer_function))

# Calculating *ACES RGB* to *sRGB* transformation matrix.
print("'ACES RGB' colorspace to 'sRGB' colorspace matrix:")
cat = color.get_chromatic_adaptation_matrix(color.xy_to_XYZ(color.COLORSPACES["ACES RGB"].whitepoint),
                                            color.xy_to_XYZ(color.COLORSPACES["sRGB"].whitepoint))
print color.COLORSPACES["sRGB"].from_XYZ * (cat * color.COLORSPACES["ACES RGB"].to_XYZ)
