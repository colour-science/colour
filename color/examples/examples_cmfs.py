# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *color matching functions* related examples.
"""

import pprint
import color

# Displaying available *color matching functions*.
pprint.pprint(color.CMFS)

# Converting from *Wright & Guild 1931 2 Degree RGB CMFs* color matching functions to *CIE 1931 2 Degree Standard Observer*:
print(color.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))

# Converting from *Stiles & Burch 1959 10 Degree RGB CMFs* color matching functions to *CIE 1964 10 Degree Standard Observer*:
print(color.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))

