#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour component transfer functions (CCTF) relates computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('Colour Component Transfer Functions (CCTF) Computations')

C = 18 / 100

message_box(('Encoding to video component signal value using "BT.709" OETF '
             'and given linear-light value:\n'
             '\n\t{0}'.format(C)))
print(colour.oetf_BT709(C))
print(colour.oetf(C, function='BT.709'))

print('\n')

N = 0.40900773
message_box(('Decoding to linear-light value using "BT.1886" EOTF and given '
             ' video component signal value:\n'
             '\n\t{0}'.format(N)))
print(colour.eotf_BT1886(N))
print(colour.eotf(N, function='BT.1886'))

print('\n')

message_box(('Encoding to "Cineon" using given linear-light value:\n'
             '\n\t{0}'.format(C)))
print(colour.log_encoding_Cineon(C))
print(colour.log_encoding_curve(C, curve='Cineon'))

print('\n')

N = 0.45731961
message_box(('Decoding to linear-light using given "Cineon" code value:\n'
             '\n\t{0}'.format(N)))
print(colour.log_decoding_Cineon(N))
print(colour.log_decoding_curve(N, curve='Cineon'))

print('\n')

message_box(('Encoding to "PLog" using given linear-light value:\n'
             '\n\t{0}'.format(C)))
print(colour.log_encoding_PivotedLog(C))
print(colour.log_encoding_curve(C, curve='PLog'))

print('\n')

N = 0.43499511
message_box(('Decoding to linear-light value using given "PLog" code value:\n'
             '\n\t{0}'.format(N)))
print(colour.log_decoding_PivotedLog(N))
print(colour.log_decoding_curve(N, curve='PLog'))

print('\n')

message_box(('Encoding to video component signal value using a pure gamma '
             'function and given linear-light value:\n'
             '\n\t{0}'.format(C)))
print(colour.gamma_function(C, 1 / 2.2))

print('\n')

N = 0.45865645
message_box(('Decoding to linear-light value using a pure gamma function and '
             'given video component signal value:\n'
             '\n\t{0}'.format(N)))
print(colour.gamma_function(N, 2.2))
