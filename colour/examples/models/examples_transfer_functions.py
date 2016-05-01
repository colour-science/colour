#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour component transfer functions (CCTF) relates computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('Colour Component Transfer Functions (CCTF) Computations')

linear = 18 / 100

message_box(('Encoding to "BT.709" using given linear light value:\n'
             '\n\t{0}'.format(linear)))
print(colour.oetf_BT709(linear))

print('\n')

message_box(('Encoding to "Cineon" using given linear light value:\n'
             '\n\t{0}'.format(linear)))
print(colour.log_encoding_Cineon(linear))
print(colour.log_encoding_curve(linear, method='Cineon'))

print('\n')

log = 0.457319613085
message_box(('Decoding to linear light using given "Cineon" code value:\n'
             '\n\t{0}'.format(log)))
print(colour.log_decoding_Cineon(log))
print(colour.log_decoding_curve(log, method='Cineon'))

print('\n')

message_box(('Encoding to "PLog" using given linear light value:\n'
             '\n\t{0}'.format(linear)))
print(colour.log_encoding_PivotedLog(linear))
print(colour.log_encoding_curve(linear, method='PLog'))

print('\n')

log = 0.434995112414
message_box(('Decoding to linear light value using given "PLog" code value:\n'
             '\n\t{0}'.format(log)))
print(colour.log_decoding_PivotedLog(log))
print(colour.log_decoding_curve(log, method='PLog'))

print('\n')
