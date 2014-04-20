#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package *chromatic adaptation* related examples.
"""

from numpy import matrix
import color

source_XYZ_matrix = matrix([[1.09923822], [1.000], [0.35445412]])
target_XYZ_matrix = matrix([[0.96907232], [1.000], [1.121792157]])

# Retrieving the *chromatic adaptation* matrix from two source *CIE XYZ* matrices, default
# adaptation method is *CAT02*.
print(color.get_chromatic_adaptation_matrix(source_XYZ_matrix, target_XYZ_matrix))

# Specifying *Bradford* adaptation method.
print(color.get_chromatic_adaptation_matrix(source_XYZ_matrix, target_XYZ_matrix, method="Bradford"))

# Using :mod:`color.illuminants` data and :mod:`color.transformations` transformations.
A_illuminant = color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["A"]
D60_illuminant = color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D60"]

print(color.get_chromatic_adaptation_matrix(color.xy_to_XYZ(A_illuminant),
										 color.xy_to_XYZ(D60_illuminant),
										 method="Von Kries"))
