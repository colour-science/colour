# -*- coding: utf-8 -*-
"""
Showcases input / output *CSV* tabular data related examples.
"""

import os
from pprint import pprint

import colour
from colour.io.common import format_spectral_data
from colour.utilities import message_box

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')

message_box('"CSV" Tabular Data IO')

message_box('Reading tabular data from "CSV" file.')
data = colour.read_spectral_data_from_csv_file(
    os.path.join(RESOURCES_DIRECTORY, 'babelcolor_average.csv'))
pprint(sorted(data.keys()))

print('\n')

message_box('Format spectral data for pretty printing.')
print(format_spectral_data(data))

print('\n')

message_box(('Reading spectral data from a "CSV" file directly as spectral '
             'distributions.'))
sds = colour.read_sds_from_csv_file(
    os.path.join(RESOURCES_DIRECTORY, 'babelcolor_average.csv'))
pprint(sds)
