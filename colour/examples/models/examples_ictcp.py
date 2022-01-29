"""
Showcases *ICtCp* *colour encoding* computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"ICtCp" Colour Encoding Computations')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    'Converting from "ITU-R BT.2020" colourspace to "ICtCp" colour '
    'encoding given "RGB" values:\n'
    "\n\t{}".format(RGB)
)
print(colour.RGB_to_ICtCp(RGB))

print("\n")

ICtCp = np.array([0.07351364, 0.00475253, 0.09351596])
message_box(
    'Converting from "ICtCp" colour encoding to "ITU-R BT.2020" '
    'colourspace given "ICtCp" values:\n'
    "\n\t{}".format(ICtCp)
)
print(colour.ICtCp_to_RGB(ICtCp))
