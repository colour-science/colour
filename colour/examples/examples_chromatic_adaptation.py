# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package *chromatic adaptation* related examples.
"""

from numpy import matrix
import colour

source_XYZ_matrix = matrix([[1.09923822], [1.000], [0.35445412]])
target_XYZ_matrix = matrix([[0.96907232], [1.000], [1.121792157]])

# Retrieving the *chromatic adaptation* matrix from two source *CIE XYZ* matrices, default
# adaptation method is *CAT02*.
print(colour.get_chromatic_adaptation_matrix(source_XYZ_matrix, target_XYZ_matrix))

# Specifying *Bradford* adaptation method.
print(colour.get_chromatic_adaptation_matrix(source_XYZ_matrix, target_XYZ_matrix, method="Bradford"))

# Using :mod:`colour.illuminants` data and :mod:`colour.computation.transformations` transformations.
A_illuminant = colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["A"]
D60_illuminant = colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D60"]

print(colour.get_chromatic_adaptation_matrix(colour.xy_to_XYZ(A_illuminant),
                                            colour.xy_to_XYZ(D60_illuminant),
                                            method="Von Kries"))
