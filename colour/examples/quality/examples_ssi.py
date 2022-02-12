"""Showcases *Academy Spectral Similarity Index* (SSI) computations."""

import colour
from colour.utilities import message_box

message_box("Academy Spectral Similarity Index Computations")

message_box(
    'Computing the "CIE Illuminant B" "Academy Spectral Similarity Index (SSI)" '
    'with "CIE Standard Illuminant D65".'
)
print(
    colour.spectral_similarity_index(
        colour.SDS_ILLUMINANTS["B"], colour.SDS_ILLUMINANTS["D65"]
    )
)
