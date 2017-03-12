#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases luminous efficiency functions computations.
"""

from pprint import pprint

import colour
from colour.utilities.verbose import message_box

message_box('Luminous Efficiency Functions Computations')

message_box('Luminous efficiency functions dataset.')
pprint(sorted(colour.LEFS))

print('\n')

message_box(('Computing the mesopic luminous efficiency function for factor:\n'
             '\n\t0.2'))
print(colour.mesopic_luminous_efficiency_function(0.2).values)
