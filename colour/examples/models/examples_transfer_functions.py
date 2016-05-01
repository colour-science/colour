#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *linear* to *log* and *log* to *linear* conversion computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Log" Conversion Computations')

linear = 18 / 100
message_box(('Converting "linear" to "log" using '
             '"Cineon" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_cineon(linear))
print(colour.linear_to_log(linear, method='Cineon'))

print('\n')

log = 0.457319613085
message_box(('Converting "log" to "linear" using '
             '"Cineon" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.cineon_to_linear(log))
print(colour.log_to_linear(log, method='Cineon'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"Panalog" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_panalog(linear))
print(colour.linear_to_log(linear, method='Panalog'))

print('\n')

log = 0.374576791382
message_box(('Converting "log" to "linear" using '
             '"Panalog" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.panalog_to_linear(log))
print(colour.log_to_linear(log, method='Panalog'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"REDLogFilm" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_red_log_film(linear))
print(colour.linear_to_log(linear, method='REDLogFilm'))

print('\n')

log = 0.637621845988
message_box(('Converting "log" to "linear" using '
             '"REDLogFilm" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.red_log_film_to_linear(log))
print(colour.log_to_linear(log, method='REDLogFilm'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"ViperLog" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_viper_log(linear))
print(colour.linear_to_log(linear, method='ViperLog'))

print('\n')

log = 0.63600806701
message_box(('Converting "log" to "linear" using '
             '"ViperLog" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.viper_log_to_linear(log))
print(colour.log_to_linear(log, method='ViperLog'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"PLog" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_pivoted_log(linear))
print(colour.linear_to_log(linear, method='PLog'))

print('\n')

log = 0.434995112414
message_box(('Converting "log" to "linear" using '
             '"PLog" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.pivoted_log_to_linear(log))
print(colour.log_to_linear(log, method='PLog'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"C-Log" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_c_log(linear))
print(colour.linear_to_log(linear, method='C-Log'))

print('\n')

log = 0.31201285555
message_box(('Converting "log" to "linear" using '
             '"C-Log" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.c_log_to_linear(log))
print(colour.log_to_linear(log, method='C-Log'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"ACEScc" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_aces_cc(linear))
print(colour.linear_to_log(linear, method='ACEScc'))

print('\n')

log = 27701.3889263
message_box(('Converting "log" to "linear" using '
             '"ACEScc" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.aces_cc_to_linear(log))
print(colour.log_to_linear(log, method='ACEScc'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"Alexa Log C" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_alexa_log_c(linear))
print(colour.linear_to_log(linear, method='Alexa Log C'))

print('\n')

log = 0.391006832034
message_box(('Converting "log" to "linear" using '
             '"Alexa Log C" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.alexa_log_c_to_linear(log))
print(colour.log_to_linear(log, method='Alexa Log C'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"S-Log" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_s_log(linear))
print(colour.linear_to_log(linear, method='S-Log'))

print('\n')

log = 0.359987846422
message_box(('Converting "log" to "linear" using '
             '"S-Log" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.s_log_to_linear(log))
print(colour.log_to_linear(log, method='S-Log'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"S-Log2" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_s_log2(linear))
print(colour.linear_to_log(linear, method='S-Log2'))

print('\n')

log = 0.384970815929
message_box(('Converting "log" to "linear" using '
             '"S-Log2" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.s_log2_to_linear(log))
print(colour.log_to_linear(log, method='S-Log2'))

print('\n')

message_box(('Converting "linear" to "log" using '
             '"S-Log3" method for '
             'given "linear" value:\n'
             '\n\t{0}'.format(linear)))
print(colour.linear_to_s_log3(linear))
print(colour.linear_to_log(linear, method='S-Log3'))

print('\n')

log = 0.410557184751
message_box(('Converting "log" to "linear" using '
             '"S-Log3" method for '
             'given "log" value:\n'
             '\n\t{0}'.format(log)))
print(colour.s_log3_to_linear(log))
print(colour.log_to_linear(log, method='S-Log3'))
