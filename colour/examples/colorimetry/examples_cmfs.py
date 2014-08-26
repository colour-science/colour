#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *colour matching functions* computations.
"""

from __future__ import division, unicode_literals

from pprint import pprint

import colour

# Displaying available *colour matching functions*.
pprint(colour.CMFS)

# Converting from *Wright & Guild 1931 2 Degree RGB CMFs* colour matching
# functions to *CIE 1931 2 Degree Standard Observer*.
print(colour.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))
print(colour.STANDARD_OBSERVERS_CMFS[
    'CIE 1931 2 Degree Standard Observer'][700])

# Converting from *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
# functions to *CIE 1964 10 Degree Standard Observer*.
print(colour.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))
print(colour.STANDARD_OBSERVERS_CMFS[
    'CIE 1964 10 Degree Standard Observer'][700])

# Converting from *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
# functions to *Stockman & Sharpe 10 Degree Cone Fundamentals*.
print(colour.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700))
print(colour.LMS_CMFS['Stockman & Sharpe 10 Degree Cone Fundamentals'][700])

# Converting from *Stockman & Sharpe 2 Degree Cone Fundamentals* colour
# matching functions to *CIE 2012 2 Degree Standard Observer* spectral
# sensitivity functions.
print(colour.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))
print(colour.STANDARD_OBSERVERS_CMFS[
    'CIE 2012 2 Degree Standard Observer'][700])

# Converting from *Stockman & Sharpe 10 Degree Cone Fundamentals* colour
# matching functions to *CIE 2012 10 Degree Standard Observer* spectral
# sensitivity functions.
print(colour.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))
print(colour.STANDARD_OBSERVERS_CMFS[
    'CIE 2012 10 Degree Standard Observer'][700])
