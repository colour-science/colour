#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Y'CbCr* *colour encoding* computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Y\'CbCr" Colour Encoding Computations')

RGB = (0.35521588, 0.41000000, 0.24177934)
message_box(('Converting to "Y\'CbCr" colour encoding from given "Rec. 709" '
             'colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_YCbCr(RGB))

print('\n')

message_box(('Converting to "Y\'CbCr" colour encoding from given "Rec. 601" '
             'colourspace values using legal range and integer output:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_YCbCr(
    RGB, colour.YCBCR_WEIGHTS['Rec. 601'], out_legal=True, out_int=True))

print('\n')

YCbCr = (101, 111, 124)
message_box(('Converting to "Rec. 601" colourspace from given "Y\'CbCr" '
             'values using legal range and integer input:\n'
             '\n\t{0}'.format(RGB)))
print(colour.YCbCr_to_RGB(YCbCr, in_legal=True, in_int=True))

print('\n')

RGB = (0.18, 0.18, 0.18)
message_box(('Converting to "Yc\'Cbc\'Crc\'" colour encoding from given '
             '"Rec. 2020" values using legal range, integer output on '
             'a 10-bit system:\n\n\t{0}'.format(RGB)))
print(colour.RGB_to_YcCbcCrc(
    RGB, out_bits=10, out_legal=True, out_int=True, is_12_bits_system=False))

print('\n')

YcCbcCrc = (422, 512, 512)
message_box(('Converting to "Rec. 2020" colourspace from given "RGB" '
             'values using legal range, integer input on a 10-bit system:\n'
             '\n\t{0}'.format(RGB)))
print(colour.YcCbcCrc_to_RGB(
    YcCbcCrc, in_bits=10, in_legal=True, in_int=True, is_12_bits_system=False))
