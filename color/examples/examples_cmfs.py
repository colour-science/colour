# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *color matching functions* related examples.
"""

import pprint
import color

# Displaying available *color matching functions*.
pprint.pprint(color.CMFS)

# Converting from *Wright & Guild 1931 2 Degree RGB CMFs* color matching functions to *CIE 1931 2 Degree Standard Observer*.
print(color.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))
print(color.STANDARD_OBSERVERS_CMFS["CIE 1931 2 Degree Standard Observer"][700])

# Converting from *Stiles & Burch 1959 10 Degree RGB CMFs* color matching functions to *CIE 1964 10 Degree Standard Observer*.
print(color.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))
print(color.STANDARD_OBSERVERS_CMFS["CIE 1964 10 Degree Standard Observer"][700])

# Converting from *Stiles & Burch 1959 10 Degree RGB CMFs* color matching functions to *Stockman & Sharpe 10 Degree Cone Fundamentals*.
print(color.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700))
print(color.LMS_CMFS["Stockman & Sharpe 10 Degree Cone Fundamentals"][700])

# Converting from *Stockman & Sharpe 2 Degree Cone Fundamentals* color matching functions to *CIE 2012 2 Degree Standard Observer* spectral sensitivity functions.
print(color.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))
print(color.STANDARD_OBSERVERS_CMFS["CIE 2012 2 Degree Standard Observer"][700])

# Converting from *Stockman & Sharpe 10 Degree Cone Fundamentals* color matching functions to *CIE 2012 10 Degree Standard Observer* spectral sensitivity functions.
print(color.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))
print(color.STANDARD_OBSERVERS_CMFS["CIE 2012 10 Degree Standard Observer"][700])