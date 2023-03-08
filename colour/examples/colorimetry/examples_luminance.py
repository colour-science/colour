"""Showcases *Luminance* computations."""

import colour

from colour.utilities import message_box

message_box('"Luminance" Computations')

V = 4.08244375
message_box(
    f'Computing "luminance" using "Newhall, Nickerson, and Judd (1943)" '
    f'method for given "Munsell" value:\n\n\t{V}'
)
print(colour.luminance(V * 10, method="Newhall 1943"))
print(colour.colorimetry.luminance_Newhall1943(V))

print("\n")

L = 41.527875844653451
message_box(
    f'Computing "luminance" using "CIE 1976" method for given '
    f'"Lightness":\n\n\t{L}'
)
print(colour.luminance(L))
print(colour.colorimetry.luminance_CIE1976(L))

print("\n")

L = 31.996390226262736
message_box(
    f'Computing "luminance" using "Fairchild and Wyble (2010)" method for '
    f'given "Lightness":\n\n\t{L}'
)
print(colour.luminance(L, method="Fairchild 2010") * 100)
print(colour.colorimetry.luminance_Fairchild2010(L) * 100)

print("\n")

L = 51.852958445912506
message_box(
    f'Computing "luminance" using "Fairchild and Chen (2011)" method for '
    f'given "Lightness":\n\n\t{L}'
)
print(colour.luminance(L, method="Fairchild 2011"))
print(colour.colorimetry.luminance_Fairchild2011(L) * 100)

print("\n")

message_box(
    f'Computing "luminance" using "ASTM D1535-08e1" method for given '
    f'"Munsell" value:\n\n\t{V}'
)
print(colour.luminance(V * 10, method="ASTM D1535"))
print(colour.colorimetry.luminance_ASTMD1535(V))
