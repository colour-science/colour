"""Showcases Helmholtz—Kohlrausch effect estimation computations."""

import colour
from colour.plotting import colour_style, plot_multi_colour_swatches
from colour.utilities import message_box

wp = colour.xy_to_Luv_uv([0.31271, 0.32902])

average_luminance = 0.14

swatches = [
    [0.45079660, 0.52288689],
    [0.19124902, 0.55444488],
    [0.13128455, 0.51210591],
    [0.14889223, 0.37091478],
    [0.28992574, 0.30964533],
]
swatches_XYZ = []
for patch in swatches:
    in_XYZ = colour.Luv_to_XYZ(colour.uv_to_Luv(patch))
    swatches_XYZ.append(in_XYZ * (average_luminance / in_XYZ[1]))

# Adapting Luminance, 250 cd/m^2 represents a typical modern computer
# display peak luminance.
L_a = 250 * average_luminance

bg_grey = colour.xy_to_XYZ(colour.Luv_uv_to_xy(wp)) * average_luminance

swatches_normal = []
swatches_VCC = []
swatches_VAC = []

for i in range(len(swatches)):
    VCC = colour.HelmholtzKohlrausch_effect_luminous_Nayatani1997(
        swatches[i], wp, L_a, method="VCC"
    )
    VAC = colour.HelmholtzKohlrausch_effect_luminous_Nayatani1997(
        swatches[i], wp, L_a, method="VAC"
    )

    swatches_normal.append(colour.XYZ_to_sRGB(bg_grey))
    swatches_normal.append(colour.XYZ_to_sRGB(swatches_XYZ[i]))

    swatches_VCC.append(colour.XYZ_to_sRGB(bg_grey))
    swatches_VCC.append(colour.XYZ_to_sRGB(swatches_XYZ[i] / VCC))

    swatches_VAC.append(colour.XYZ_to_sRGB(bg_grey * VAC))
    swatches_VAC.append(colour.XYZ_to_sRGB(swatches_XYZ[i]))

colour_style()

message_box(
    "Plotting swatches with the same luminance (Y).\n"
    "The Helmholtz—Kohlrausch effect will be very noticeable."
)
plot_multi_colour_swatches(swatches_normal, compare_swatches="stacked")

message_box(
    "Plotting HKE-compensated swatches with VCC method.\n"
    "The Helmholtz—Kohlrausch effect has been compensated using VCC"
    "(variable chromatic colour) method."
)
plot_multi_colour_swatches(swatches_VCC, compare_swatches="stacked")

message_box(
    "Plotting HKE-compensated swatches with VAC method.\n"
    "The Helmholtz—Kohlrausch effect has been compensated for using VAC"
    "(variable achromatic colour) method."
)
plot_multi_colour_swatches(swatches_VAC, compare_swatches="stacked")
