"""Showcases *ANSI/IES TM-30-18 Colour Rendition Report* plotting examples."""

import colour
from colour.plotting import (
    colour_style,
    plot_single_sd_colour_rendition_report,
)
from colour.utilities import message_box

message_box("ANSI/IES TM-30-18 Colour Rendition Report")

colour_style()

sd = colour.SDS_ILLUMINANTS["FL2"]

message_box('Plotting a full "ANSI/IES TM-30-18 Colour Rendition Report".')
plot_single_sd_colour_rendition_report(sd)

print("\n")

message_box(
    'Plotting an intermediate "ANSI/IES TM-30-18 Colour Rendition Report".'
)
plot_single_sd_colour_rendition_report(sd, "Intermediate")

print("\n")

message_box('Plotting a simple "ANSI/IES TM-30-18 Colour Rendition Report".')
plot_single_sd_colour_rendition_report(sd, "Simple")
