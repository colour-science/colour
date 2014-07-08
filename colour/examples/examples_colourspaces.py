# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package *colourspaces* related examples.
"""

import pprint
import colour

# Displaying :attr:`colour.COLOURSPACES` data.
pprint.pprint(sorted(colour.COLOURSPACES.keys()))

colourspace = colour.COLOURSPACES["ACES RGB"]
print("Name: '{0}'".format(colourspace.name))
print("Primaries: '{0}'".format(colourspace.primaries))
print("Normalised primary matrix to 'CIE XYZ': '{0}'".format(colourspace.to_XYZ))
print("Normalised primary matrix from 'CIE XYZ': '{0}'".format(colourspace.from_XYZ))
print("Transfer function: '{0}'".format(colourspace.transfer_function))
print("Inverse transfer function: '{0}'".format(colourspace.inverse_transfer_function))

# Calculating *ACES RGB* to *sRGB* transformation matrix.
print("'ACES RGB' colourspace to 'sRGB' colourspace matrix:")
cat = colour.get_chromatic_adaptation_matrix(colour.xy_to_XYZ(colour.COLOURSPACES["ACES RGB"].whitepoint),
                                            colour.xy_to_XYZ(colour.COLOURSPACES["sRGB"].whitepoint))
print colour.COLOURSPACES["sRGB"].from_XYZ * (cat * colour.COLOURSPACES["ACES RGB"].to_XYZ)
