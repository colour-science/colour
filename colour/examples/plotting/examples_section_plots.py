"""Showcases gamut section plotting examples."""

import numpy as np
from matplotlib.lines import Line2D

import colour.plotting
from colour.plotting import (
    colour_style,
    plot_visible_spectrum_section,
    plot_RGB_colourspace_section,
)
from colour.utilities import message_box

message_box("Gamut Section Plots")

colour_style()

message_box(
    'Plotting a "Visible Spectrum" section at 50% "Lightness" in the '
    '"CIE Luv" colourspace.'
)

plot_visible_spectrum_section(
    model="CIE Luv",
    origin=0.5,
)

print("\n")

message_box(
    'Plotting a "Visible Spectrum" section at 50% "Lightness" in the '
    '"CIE Luv" colourspace and customising the section styling.'
)

plot_visible_spectrum_section(
    model="CIE Luv", origin=0.5, section_colours="RGB", section_opacity=0.15
)

print("\n")

message_box(
    'Plotting a "Visible Spectrum" section at 50% "Lightness" in the '
    '"CIE Luv" colourspace.'
)

plot_visible_spectrum_section(model="CIE Luv", origin=0.5)

print("\n")

message_box(
    'Plotting a "Visible Spectrum" section at 25% along the "u" axis in the '
    '"CIE Luv" colourspace.'
)

plot_visible_spectrum_section(
    model="CIE Luv",
    axis="+x",
    origin=0.25,
    section_colours="RGB",
    section_opacity=0.15,
)

print("\n")

message_box(
    'Plotting a "sRGB" colourspace section at 50% "Lightness" in the '
    '"ICtCp" colourspace using section normalisation.'
)

plot_RGB_colourspace_section(
    colourspace="sRGB", model="ICtCp", origin=0.5, normalise=True
)

print("\n")

message_box(
    'Combining multiple hull sections together at 25% "Lightness" in the '
    '"Oklab" colourspace.'
)

figure, axes = plot_visible_spectrum_section(
    model="Oklab", origin=0.25, section_opacity=0.15, standalone=False
)
plot_RGB_colourspace_section(
    colourspace="sRGB",
    model="Oklab",
    origin=0.25,
    section_colours="RGB",
    section_opacity=0.15,
    contour_colours="RGB",
    axes=axes,
)

print("\n")

message_box(
    'Combining multiple hull sections together at varying "Lightness" in the '
    '"DIN99" colourspace.'
)

figure, axes = plot_visible_spectrum_section(
    model="DIN99", origin=0.5, section_opacity=0.15, standalone=False
)

bounding_box = [
    axes.get_xlim()[0],
    axes.get_xlim()[1],
    axes.get_ylim()[0],
    axes.get_ylim()[1],
]

section_colours = colour.notation.HEX_to_RGB(
    colour.plotting.CONSTANTS_COLOUR_STYLE.colour.cycle[:4]
)

origins = []
legend_lines = []
for i, RGB in zip(np.arange(0.5, 0.9, 0.1), section_colours):
    origins.append(i * 100)
    plot_RGB_colourspace_section(
        colourspace="sRGB",
        model="DIN99",
        origin=i,
        section_colours=RGB,
        section_opacity=0.15,
        contour_colours=RGB,
        axes=axes,
        standalone=False,
    )
    legend_lines.append(Line2D([0], [0], color=RGB, label=f"{i * 100}%"))

axes.legend(handles=legend_lines)

colour.plotting.render(
    title=f"Visible Spectrum - 50% - sRGB Sections - {origins}% -  DIN99",
    axes=axes,
    bounding_box=bounding_box,
)
