# -*- coding: utf-8 -*-
"""
Showcases *Photometry* computations.
"""

import colour
from colour.utilities import message_box

message_box('"Photometry" Computations')

sd = colour.LIGHT_SOURCES_SDS['Neodimium Incandescent']
message_box(('Computing "Luminous Flux" for given spectral '
             'distribution:\n'
             '\n\t{0}'.format(sd.name)))
print(colour.luminous_flux(sd))

print('\n')

message_box(('Computing "Luminous Efficiency" for given spectral '
             'distribution:\n'
             '\n\t{0}'.format(sd.name)))
print(colour.luminous_efficiency(sd))

print('\n')

message_box(('Computing "Luminous Efficacy" for given spectral '
             'distribution:\n'
             '\n\t{0}'.format(sd.name)))
print(colour.luminous_efficacy(sd))
