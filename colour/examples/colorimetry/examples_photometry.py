"""Showcases *Photometry* computations."""

import colour
from colour.utilities import message_box

message_box('"Photometry" Computations')

sd_light_source = colour.SDS_LIGHT_SOURCES["Neodimium Incandescent"]
message_box(
    f'Computing "Luminous Flux" for given spectral distribution:\n\n'
    f"\t{sd_light_source.name}"
)
print(colour.luminous_flux(sd_light_source))

print("\n")

message_box(
    f'Computing "Luminous Efficiency" for given spectral distribution:\n\n'
    f"\t{sd_light_source.name}"
)
print(colour.luminous_efficiency(sd_light_source))

print("\n")

message_box(
    f'Computing "Luminous Efficacy" for given spectral distribution:\n\n'
    f"\t{sd_light_source.name}"
)
print(colour.luminous_efficacy(sd_light_source))
