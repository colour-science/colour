#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Photometry* computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Photometry" Computations')

spd = colour.LIGHT_SOURCES_RELATIVE_SPDS['Neodimium Incandescent']
message_box(('Computing "Luminous Flux" for given spectral power '
             'distribution:\n'
             '\n\t{0}'.format(spd.name)))
print(colour.luminous_flux(spd))

print('\n')

message_box(('Computing "Luminous Efficiency" for given spectral power '
             'distribution:\n'
             '\n\t{0}'.format(spd.name)))
print(colour.luminous_efficiency(spd))

print('\n')

message_box(('Computing "Luminous Efficacy" for given spectral power '
             'distribution:\n'
             '\n\t{0}'.format(spd.name)))
print(colour.luminous_efficacy(spd))
