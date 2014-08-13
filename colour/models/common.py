#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Colour Models Utilities
==============================

Defines various colour models common utilities.

"""

from __future__ import unicode_literals

import numpy as np

from colour.algebra import to_ndarray
from colour.models import RGB_COLOURSPACES, XYZ_to_RGB

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "New BSD License - http://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["XYZ_to_sRGB"]


def XYZ_to_sRGB(XYZ,
                illuminant=RGB_COLOURSPACES.get("sRGB").whitepoint,
                chromatic_adaptation_method="CAT02",
                transfer_function=True):
    """
    Converts from *CIE XYZ* colourspace to *sRGB* colourspace.

    Parameters
    ----------
    XYZ : array_like, (3, 1)
        *CIE XYZ* colourspace matrix.
    illuminant : array_like, optional
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_method : unicode, optional
        ("XYZ Scaling", "Bradford", "Von Kries", "Fairchild", "CAT02")
        *Chromatic adaptation* method.
    transfer_function : bool, optional
        Apply *sRGB* *transfer function*.

    Returns
    -------
    ndarray, (3, 1)
        *sRGB* colour matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 1].

    Examples
    --------
    >>> colour.XYZ_to_sRGB(np.array([0.1180583421, 0.1034, 0.0515089229]))
    array([[ 0.48224885],
           [ 0.31651974],
           [ 0.22070513]])
    """

    sRGB = RGB_COLOURSPACES.get("sRGB")
    return XYZ_to_RGB(XYZ,
                      illuminant,
                      sRGB.whitepoint,
                      sRGB.to_RGB,
                      chromatic_adaptation_method,
                      sRGB.transfer_function if transfer_function else None)