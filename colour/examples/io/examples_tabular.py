"""Showcases input / output *CSV* tabular data related examples."""

import os
from pprint import pprint

import colour
from colour.utilities import message_box

ROOT_RESOURCES = os.path.join(os.path.dirname(__file__), "resources")

message_box('"CSV" Tabular Data IO')

message_box('Reading tabular data from "CSV" file.')
data_babelcolor_average = colour.read_spectral_data_from_csv_file(
    os.path.join(ROOT_RESOURCES, "babelcolor_average.csv")
)
pprint(sorted(data_babelcolor_average.keys()))

print("\n")

message_box(
    'Reading spectral data from a "CSV" file directly as spectral '
    "distributions."
)
sds_babelcolor_average = colour.read_sds_from_csv_file(
    os.path.join(ROOT_RESOURCES, "babelcolor_average.csv")
)
pprint(sds_babelcolor_average)
