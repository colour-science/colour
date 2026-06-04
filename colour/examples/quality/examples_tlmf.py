"""
Demonstrate *Television Luminaire Matching Factor* (TLMF-2013) computations.
"""

import colour
from colour.utilities import message_box

message_box("Television Luminaire Matching Factor (TLMF-2013) Computations")

message_box('Computing TLMF-2013 for "FL2" vs "D65".')
print(
    colour.television_luminaire_matching_factor(
        colour.SDS_ILLUMINANTS["FL2"], colour.SDS_ILLUMINANTS["D65"]
    )
)

message_box("Fetching TLMF-2013 additional data.")
spec = colour.television_luminaire_matching_factor(
    colour.SDS_ILLUMINANTS["FL2"],
    colour.SDS_ILLUMINANTS["D65"],
    additional_data=True,
)
print(spec)
