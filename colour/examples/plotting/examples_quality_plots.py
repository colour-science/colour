"""Showcases colour quality plotting examples."""

import colour
from colour.plotting import (
    colour_style,
    plot_multi_sds_colour_quality_scales_bars,
    plot_multi_sds_colour_rendering_indexes_bars,
    plot_single_sd_colour_quality_scale_bars,
    plot_single_sd_colour_rendering_index_bars,
)
from colour.utilities import message_box

message_box("Colour Quality Plots")

colour_style()

message_box('Plotting "F2" illuminant "Colour Rendering Index (CRI)".')
plot_single_sd_colour_rendering_index_bars(colour.SDS_ILLUMINANTS["FL2"])

print("\n")

message_box(
    "Plotting various illuminants and light sources "
    '"Colour Rendering Index (CRI)".'
)
plot_multi_sds_colour_rendering_indexes_bars(
    (
        colour.SDS_ILLUMINANTS["FL2"],
        colour.SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"],
        colour.SDS_LIGHT_SOURCES["Kinoton 75P"],
    )
)

print("\n")

message_box('Plotting "F2" illuminant "Colour Quality Scale (CQS)".')
plot_single_sd_colour_quality_scale_bars(colour.SDS_ILLUMINANTS["FL2"])

print("\n")

message_box(
    "Plotting various illuminants and light sources "
    '"Colour Quality Scale (CQS)".'
)
plot_multi_sds_colour_quality_scales_bars(
    (
        colour.SDS_ILLUMINANTS["FL2"],
        colour.SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"],
        colour.SDS_LIGHT_SOURCES["Kinoton 75P"],
    )
)
