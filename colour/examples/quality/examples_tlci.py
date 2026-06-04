"""
Demonstrate *Television Lighting Consistency Index* (TLCI-2012) computations.
"""

import colour
from colour.utilities import message_box

message_box("Television Lighting Consistency Index (TLCI-2012) Computations")

message_box('Computing TLCI-2012 for "FL2".')
print(colour.television_lighting_consistency_index(colour.SDS_ILLUMINANTS["FL2"]))

message_box("Fetching TLCI-2012 additional data.")
spec = colour.television_lighting_consistency_index(
    colour.SDS_ILLUMINANTS["FL2"],
    additional_data=True,
)
print(spec)
