# -*- coding: utf-8 -*-
"""
Showcases illuminants dataset.
"""

from pprint import pprint

import colour
from colour.utilities import message_box

message_box('Illuminants Dataset')

message_box('Illuminants spectral distributions dataset.')
pprint(sorted(colour.ILLUMINANTS_SDS.keys()))

print('\n')

message_box('Illuminants chromaticity coordinates dataset.')
# Filtering aliases.
observers = dict(((observer, dataset)
                  for observer, dataset in sorted(colour.ILLUMINANTS.items())
                  if ' ' in observer))
for observer, illuminants in observers.items():
    print('"{0}".'.format(observer))
    for illuminant, xy in sorted(illuminants.items()):
        print('\t"{0}": {1}'.format(illuminant, xy))
    print('\n')
