"""Showcases *Delta E* colour difference computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Delta E" Computations')

Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
message_box(
    f'Computing "Delta E" with "CIE 1976" method from given "CIE L*a*b*" '
    f"colourspace matrices:\n\n"
    f"\t{Lab_1}\n"
    f"\t{Lab_2}"
)
print(colour.delta_E(Lab_1, Lab_2, method="CIE 1976"))
print(colour.difference.delta_E_CIE1976(Lab_1, Lab_2))

print("\n")

message_box(
    f'Computing "Delta E" with "CIE 1994" method from given "CIE L*a*b*" '
    f"colourspace matrices:\n\n"
    f"\t{Lab_1}\n"
    f"\t{Lab_2}"
)
print(colour.delta_E(Lab_1, Lab_2, method="CIE 1994"))
print(colour.difference.delta_E_CIE1994(Lab_1, Lab_2))

print("\n")

message_box(
    f'Computing "Delta E" with "CIE 1994" method from given "CIE L*a*b*" '
    f'colourspace matrices for "graphics arts" applications:\n\n'
    f"\t{Lab_1}\n"
    f"\t{Lab_2}"
)
print(colour.delta_E(Lab_1, Lab_2, method="CIE 1994", textiles=False))
print(colour.difference.delta_E_CIE1994(Lab_1, Lab_2, textiles=False))

print("\n")

message_box(
    f'Computing "Delta E" with "CIE 2000" method from given "CIE L*a*b*" '
    f"colourspace matrices:\n\n"
    f"\t{Lab_1}\n"
    f"\t{Lab_2}"
)
print(colour.delta_E(Lab_1, Lab_2, method="CIE 2000"))
print(colour.difference.delta_E_CIE2000(Lab_1, Lab_2))

print("\n")

message_box(
    f'Computing "Delta E" with "CMC" method from given "CIE L*a*b*" '
    f"colourspace matrices:\n\n"
    f"\t{Lab_1}\n"
    f"\t{Lab_2}"
)
print(colour.delta_E(Lab_1, Lab_2, method="CMC"))
print(colour.difference.delta_E_CMC(Lab_1, Lab_2))

print("\n")

message_box(
    f'Computing "Delta E" with "CMC" method from given "CIE L*a*b*" '
    f"colourspace matrices with imperceptibility threshold:\n\n"
    f"\t{Lab_1}\n"
    f"\t{Lab_2}"
)
print(colour.delta_E(Lab_1, Lab_2, method="CMC", l=1))  # noqa
print(colour.difference.delta_E_CMC(Lab_1, Lab_2, l=1))  # noqa

print("\n")

message_box(
    f'Computing "Delta E" with "DIN99" method from given "CIE L*a*b*" '
    f"colourspace matrices:\n\n"
    f"\t{Lab_1}\n"
    f"\t{Lab_2}"
)
print(colour.delta_E(Lab_1, Lab_2, method="DIN99"))
print(colour.difference.delta_E_DIN99(Lab_1, Lab_2))

print("\n")

Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
message_box(
    f'Computing "Delta E" with "Luo et al. (2006)" "CAM02-LCD" method from '
    f"given \"J'a'b'\" arrays:\n\n"
    f"\t{Jpapbp_1}\n"
    f"\t{Jpapbp_2}"
)
print(colour.delta_E(Jpapbp_1, Jpapbp_2, method="CAM02-LCD"))
print(colour.difference.delta_E_CAM02LCD(Jpapbp_1, Jpapbp_2))

print("\n")

message_box(
    f'Computing "Delta E" with "Luo et al. (2006)" "CAM02-SCD" method from '
    f"given \"J'a'b'\" arrays:\n\n"
    f"\t{Jpapbp_1}\n"
    f"\t{Jpapbp_2}"
)
print(colour.delta_E(Jpapbp_1, Jpapbp_2, method="CAM02-SCD"))
print(colour.difference.delta_E_CAM02SCD(Jpapbp_1, Jpapbp_2))

print("\n")

Jpapbp_1 = np.array([54.89102616, -9.42910274, -5.52845976])
Jpapbp_2 = np.array([54.81983401, -13.21630207, -4.15161146])
message_box(
    f'Computing "Delta E" with "Li et al. (2016)" "CAM02-LCD" method from '
    f"given \"J'a'b'\" arrays:\n\n"
    f"\t{Jpapbp_1}\n"
    f"\t{Jpapbp_2}"
)
print(colour.delta_E(Jpapbp_1, Jpapbp_2, method="CAM16-LCD"))
print(colour.difference.delta_E_CAM16LCD(Jpapbp_1, Jpapbp_2))

print("\n")

message_box(
    f'Computing "Delta E" with "Li et al. (2016)" "CAM16-SCD" method from '
    f"given \"J'a'b'\" arrays:\n\n"
    f"\t{Jpapbp_1}\n"
    f"\t{Jpapbp_2}"
)
print(colour.delta_E(Jpapbp_1, Jpapbp_2, method="CAM16-SCD"))
print(colour.difference.delta_E_CAM16SCD(Jpapbp_1, Jpapbp_2))
