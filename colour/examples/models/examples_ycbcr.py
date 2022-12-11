"""Showcases *Y'CbCr* *colour encoding* computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Y\'CbCr" Colour Encoding Computations')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    f'Converting to the "Y\'CbCr" colour encoding from given "ITU-R BT.709" '
    f"colourspace values:\n\n\t{RGB}"
)
print(colour.RGB_to_YCbCr(RGB))

print("\n")

message_box(
    f'Converting to the "Y\'CbCr" colour encoding from given"ITU-R BT.601" '
    f"colourspace values using legal range and int output:\n\n\t{RGB}"
)
print(
    colour.RGB_to_YCbCr(
        RGB, colour.WEIGHTS_YCBCR["ITU-R BT.601"], out_legal=True, out_int=True
    )
)

print("\n")

YCbCr = np.array([101, 111, 124])
message_box(
    f'Converting to the "ITU-R BT.601" colourspace from given "Y\'CbCr" '
    f"values using legal range and int input:\n\n\t{RGB}"
)
print(colour.YCbCr_to_RGB(YCbCr, in_legal=True, in_int=True))

print("\n")

RGB = np.array([0.18, 0.18, 0.18])
message_box(
    f"Converting to the \"Yc'Cbc'Crc'\" colour encoding from given "
    f'"ITU-R BT.2020" values using legal range, int output on a 10-bit '
    f"system:\n\n\t{RGB}"
)
print(
    colour.RGB_to_YcCbcCrc(
        RGB, out_bits=10, out_legal=True, out_int=True, is_12_bits_system=False
    )
)

print("\n")

YcCbcCrc = np.array([422, 512, 512])
message_box(
    f'Converting to the "ITU-R BT.2020" colourspace from given "RGB" values '
    f"using legal range, int input on a 10-bit system:\n\n\t{RGB}"
)
print(
    colour.YcCbcCrc_to_RGB(
        YcCbcCrc,
        in_bits=10,
        in_legal=True,
        in_int=True,
        is_12_bits_system=False,
    )
)
