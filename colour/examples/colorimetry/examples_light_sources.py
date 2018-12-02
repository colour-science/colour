# -*- coding: utf-8 -*-
"""
Showcases light sources dataset.
"""

from pprint import pprint

import colour
from colour.utilities import message_box

message_box('Light Sources Dataset')

message_box('Light sources spectral distributions dataset.')
pprint(sorted(colour.LIGHT_SOURCES_SDS.keys()))

print('\n')

message_box('Light sources chromaticity coordinates dataset.')
# Filtering aliases.
observers = dict(((observer, dataset)
                  for observer, dataset in sorted(colour.LIGHT_SOURCES.items())
                  if ' ' in observer))
for observer, light_source in observers.items():
    print('"{0}".'.format(observer))
    for illuminant, xy in sorted(light_source.items()):
        print('\t"{0}": {1}'.format(illuminant, xy))
    print('\n')
