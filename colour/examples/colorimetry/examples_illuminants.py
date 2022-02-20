"""Showcases illuminants datasets."""

from pprint import pprint

import colour
from colour.utilities import message_box

message_box("Illuminants Dataset")

message_box("Illuminants spectral distributions dataset.")
pprint(sorted(colour.SDS_ILLUMINANTS.keys()))

print("\n")

message_box("Illuminants chromaticity coordinates dataset.")
# Filtering aliases.
observers = {
    observer: dataset
    for observer, dataset in sorted(colour.CCS_ILLUMINANTS.items())
    if " " in observer
}
for observer, illuminants in observers.items():
    print(f'"{observer}".')
    for illuminant, xy in sorted(illuminants.items()):
        print(f'\t"{illuminant}": {xy}')
    print("\n")
