# -*- coding: utf-8 -*-
"""
Showcases colour matching functions computations.
"""

from pprint import pprint

import colour
from colour.utilities import message_box

message_box('Colour Matching Functions Computations')

message_box('Colour matching functions dataset.')
pprint(sorted(colour.CMFS.keys()))

print('\n')

message_box(('Converting from "Wright & Guild 1931 2 Degree RGB CMFs" colour '
             'matching functions to "CIE 1931 2 Degree Standard Observer" at '
             'wavelength 700 nm.'))
print(
    colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'][700])
print(colour.colorimetry.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))

print('\n')

message_box(('Converting from "Stiles & Burch 1959 10 Degree RGB CMFs" colour '
             'matching functions to "CIE 1964 10 Degree Standard Observer" at '
             'wavelength 700 nm.'))
print(colour.STANDARD_OBSERVERS_CMFS['CIE 1964 10 Degree Standard Observer'][
    700])
print(colour.colorimetry.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))

print('\n')

message_box(('Converting from "Stiles & Burch 1959 10 Degree RGB CMFs" colour '
             'matching functions to '
             '"Stockman & Sharpe 10 Degree Cone Fundamentals" colour matching'
             ' functions at wavelength 700 nm.'))
print(colour.LMS_CMFS['Stockman & Sharpe 10 Degree Cone Fundamentals'][700])
print(colour.colorimetry.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700))

print('\n')

message_box(('Converting from "Stockman & Sharpe 2 Degree Cone Fundamentals" '
             'colour matching functions functions to '
             '"CIE 2012 2 Degree Standard Observer" spectral sensitivity '
             'functions at wavelength 700 nm.'))
print(
    colour.STANDARD_OBSERVERS_CMFS['CIE 2012 2 Degree Standard Observer'][700])
print(colour.colorimetry.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))

print('\n')

message_box(('Converting from "Stockman & Sharpe 10 Degree Cone Fundamentals" '
             'colour matching functions functions to '
             '"CIE 2012 10 Degree Standard Observer" spectral sensitivity '
             'functions at wavelength 700 nm.'))
print(colour.STANDARD_OBSERVERS_CMFS['CIE 2012 10 Degree Standard Observer'][
    700])
print(colour.colorimetry.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))
