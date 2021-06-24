# -*- coding: utf-8 -*-
"""
Showcases Helmholtz—Kohlrausch effect estimation computations.
"""

import colour
from colour.plotting import (ColourSwatch, colour_style,
                             plot_multi_colour_swatches)
from colour.utilities import message_box


wp = colour.xy_to_Luv_uv([0.31271,0.32902])

main_luminance = 0.14

swatches = [
    [ 0.45079660, 0.52288689],
    [ 0.19124902, 0.55444488],
    [ 0.13128455, 0.51210591],
    [ 0.14889223, 0.37091478],
    [ 0.28992574, 0.30964533]
]
swatches_XYZ = []
for patch in swatches:
    in_XYZ = colour.Luv_to_XYZ((colour.uv_to_Luv(patch)))
    swatches_XYZ.append(in_XYZ * (main_luminance/in_XYZ[1]))

# Luminance adapting. 250 cd/m^2 is to represent an average computer display
L_a = 250 * main_luminance

bg_grey = colour.xy_to_XYZ(colour.Luv_uv_to_xy(wp)) * main_luminance

swatches_normal = []
swatches_VCC = []
swatches_VAC = []

for i in range (len(swatches)):
    VCC = colour.HelmholtzKohlrausch_effect_luminous_Nayatani1997(swatches[i],
        wp, L_a, method='VCC')
    VAC = colour.HelmholtzKohlrausch_effect_luminous_Nayatani1997(swatches[i],
        wp, L_a, method='VAC')

    swatches_normal.append(ColourSwatch(RGB=colour.XYZ_to_sRGB(bg_grey, apply_cctf_encoding=True)))
    swatches_normal.append(ColourSwatch(RGB=colour.XYZ_to_sRGB(swatches_XYZ[i], apply_cctf_encoding=True)))

    swatches_VCC.append(ColourSwatch(RGB=colour.XYZ_to_sRGB(bg_grey, apply_cctf_encoding=True)))
    swatches_VCC.append(ColourSwatch(RGB=colour.XYZ_to_sRGB(swatches_XYZ[i]/VCC, apply_cctf_encoding=True)))

    swatches_VAC.append(ColourSwatch(RGB=colour.XYZ_to_sRGB(bg_grey*VAC, apply_cctf_encoding=True)))
    swatches_VAC.append(ColourSwatch(RGB=colour.XYZ_to_sRGB(swatches_XYZ[i], apply_cctf_encoding=True)))

colour_style()

message_box('Plotting swatches with the same luminance (Y).\n'
    'The Helmholtz—Kohlrausch effect will be very noticable.')
plot_multi_colour_swatches(swatches_normal, compare_swatches='stacked')

message_box('Plotting HKE-compensated swatches with VCC method.\n'
    'The Helmholtz—Kohlrausch effect has been compensated using VCC'
    '(variable chromatic colour) method.')
plot_multi_colour_swatches(swatches_VCC, compare_swatches='stacked')

message_box('Plotting HKE-compensated swatches with VAC method.\n'
    'The Helmholtz—Kohlrausch effect has been compensated for using VAC'
    '(variable achromatic colour) method.')
plot_multi_colour_swatches(swatches_VAC, compare_swatches='stacked')