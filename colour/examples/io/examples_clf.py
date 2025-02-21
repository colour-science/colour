"""
Showcases using a *Common LUT Format (CLF)* file to build a colour transform.
"""

import os
from pprint import pprint

import numpy as np

import colour
from colour.utilities import is_clf_io_installed, message_box

if is_clf_io_installed():
    import colour_clf_io as clf_io

    ROOT_RESOURCES = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "io",
        "luts",
        "tests",
        "clf_apply",
        "resources",
    )

    message_box("Using a *Common LUT Format (CLF)* colour transformation.")

    message_box("Reading a CLF.")
    path = os.path.join(ROOT_RESOURCES, "LMT Kodak 2383 Print Emulation.xml")
    process_list = clf_io.read_clf(path)
    if process_list is None:
        err_msg = "Could not read the CLF."
        raise RuntimeError(err_msg)
    pprint(process_list)

    print("\n")

    message_box("Applying the CLF.")

    RGB = np.array([0.35521588, 0.41000000, 0.24177934])

    result = colour.io.luts.clf.apply(process_list, RGB)

    pprint(result)
