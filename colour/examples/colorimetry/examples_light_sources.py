"""Showcases light sources datasets."""

from pprint import pprint

import colour
from colour.utilities import message_box

message_box("Light Sources Dataset")

message_box("Light sources spectral distributions datasets.")
pprint(sorted(colour.SDS_LIGHT_SOURCES.keys()))

print("\n")

message_box("Light sources chromaticity coordinates datasets.")
# Filtering aliases.
observers = {
    observer: dataset
    for observer, dataset in sorted(colour.CCS_LIGHT_SOURCES.items())
    if " " in observer
}
for observer, light_source in observers.items():
    print(f'"{observer}".')
    for illuminant, xy in sorted(light_source.items()):
        print(f'\t"{illuminant}": {xy}')
    print("\n")
