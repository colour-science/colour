#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package *chromatic adaptation* related examples.
"""

from numpy import matrix
import color

sourceXYZMatrix = matrix([[1.09923822], [1.000], [0.35445412]])
targetXYZMatrix = matrix([[0.96907232], [1.000], [1.121792157]])

# Retrieving the *chromatic adaptation* matrix from two source *CIE XYZ* matrices, default
# adaptation method is *CAT02*.
print(color.getChromaticAdaptationMatrix(sourceXYZMatrix, targetXYZMatrix))

# Specifying *Bradford* adaptation method.
print(color.getChromaticAdaptationMatrix(sourceXYZMatrix, targetXYZMatrix, method="Bradford"))

# Using :mod:`color.illuminants` data and :mod:`color.transformations` transformations.
illuminantA = color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["A"]
illuminantD60 = color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D60"]

print(color.getChromaticAdaptationMatrix(color.xy_to_XYZ(illuminantA),
										 color.xy_to_XYZ(illuminantD60),
										 method="Von Kries"))
