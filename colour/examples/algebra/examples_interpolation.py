"""Showcases interpolation computations."""

import matplotlib.pyplot as plt
import numpy as np
import os

import colour
from colour.hints import cast
from colour.plotting import render
from colour.utilities import message_box

message_box("Interpolation Computations")

message_box(
    'Comparing the "Sprague (1880)" and "Cubic Spline" recommended '
    'interpolation methods to the "Pchip" method.'
)

data_uniform = {
    340: 0.0000,
    360: 0.0000,
    380: 0.0000,
    400: 0.0641,
    420: 0.0645,
    440: 0.0562,
    460: 0.0537,
    480: 0.0559,
    500: 0.0651,
    520: 0.0705,
    540: 0.0772,
    560: 0.0870,
    580: 0.1128,
    600: 0.1360,
    620: 0.1511,
    640: 0.1688,
    660: 0.1996,
    680: 0.2397,
    700: 0.2852,
    720: 0.0000,
    740: 0.0000,
    760: 0.0000,
    780: 0.0000,
    800: 0.0000,
    820: 0.0000,
}

data_non_uniform = {
    340.1: 0.0000,
    360: 0.0000,
    380: 0.0000,
    400: 0.0641,
    420: 0.0645,
    440: 0.0562,
    460: 0.0537,
    480: 0.0559,
    500: 0.0651,
    520: 0.0705,
    540: 0.0772,
    560: 0.0870,
    580: 0.1128,
    600: 0.1360,
    620: 0.1511,
    640: 0.1688,
    660: 0.1996,
    680: 0.2397,
    700: 0.2852,
    720: 0.0000,
    740: 0.0000,
    760: 0.0000,
    780: 0.0000,
    800: 0.0000,
    820.9: 0.0000,
}

sd_base = colour.SpectralDistribution(data_uniform, name="Reference")
uniform_interpolated_sd = colour.SpectralDistribution(
    data_uniform, name="Uniform - Sprague Interpolation"
)
uniform_pchip_interpolated_sd = colour.SpectralDistribution(
    data_uniform, name="Uniform - Pchip Interpolation"
)
non_uniform_interpolated_sd = colour.SpectralDistribution(
    data_non_uniform, name="Non Uniform - Cubic Spline Interpolation"
)

uniform_interpolated_sd.interpolate(
    colour.SpectralShape(
        uniform_interpolated_sd.shape.start,
        uniform_interpolated_sd.shape.end,
        1,
    )
)
uniform_pchip_interpolated_sd.interpolate(
    colour.SpectralShape(
        uniform_pchip_interpolated_sd.shape.start,
        uniform_pchip_interpolated_sd.shape.end,
        1,
    ),
    interpolator=colour.PchipInterpolator,
)
non_uniform_interpolated_sd.interpolate(
    colour.SpectralShape(
        non_uniform_interpolated_sd.shape.start,
        non_uniform_interpolated_sd.shape.end,
        1,
    )
)

shape = sd_base.shape
x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []

plt.plot(
    sd_base.wavelengths, sd_base.values, "ro-", label=sd_base.name, linewidth=1
)
plt.plot(
    uniform_interpolated_sd.wavelengths,
    uniform_interpolated_sd.values,
    label=uniform_interpolated_sd.name,
    linewidth=1,
)
plt.plot(
    uniform_pchip_interpolated_sd.wavelengths,
    uniform_pchip_interpolated_sd.values,
    label=uniform_pchip_interpolated_sd.name,
    linewidth=1,
)
plt.plot(
    non_uniform_interpolated_sd.wavelengths,
    non_uniform_interpolated_sd.values,
    label=non_uniform_interpolated_sd.name,
    linewidth=1,
)

x_limit_min.append(shape.start)
x_limit_max.append(shape.end)
y_limit_min.append(min(sd_base.values))
y_limit_max.append(max(sd_base.values))

settings = {
    "x_label": "Wavelength $\\lambda$ (nm)",
    "y_label": "Spectral Distribution",
    "legend": True,
    "legend_location": "upper left",
    "x_ticker": True,
    "y_ticker": True,
    "bounding_box": (
        min(x_limit_min),
        max(x_limit_max),
        min(y_limit_min),
        max(y_limit_max),
    ),
}

render(**settings)

print("\n")

V_xyz = np.random.random((6, 3))
message_box(
    f'Performing "trilinear" interpolation of given "xyz" values:\n\n'
    f"{V_xyz}\n\n"
    f"using given interpolation table."
)
path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "io",
    "luts",
    "tests",
    "resources",
    "iridas_cube",
    "Colour_Correct.cube",
)
table = cast(colour.LUT3D, colour.read_LUT(path)).table
print(colour.table_interpolation(V_xyz, table, method="Trilinear"))
print(colour.algebra.table_interpolation_trilinear(V_xyz, table))

print("\n")

message_box(
    f'Performing "tetrahedral" interpolation of given "xyz" values:\n\n'
    f"{V_xyz}\n\n"
    f"using given interpolation table."
)
print(colour.table_interpolation(V_xyz, table, method="Tetrahedral"))
print(colour.algebra.table_interpolation_tetrahedral(V_xyz, table))
