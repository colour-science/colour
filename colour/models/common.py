#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Colour Models Utilities
==============================

Defines various colour models common utilities.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models import RGB_COLOURSPACES, XYZ_to_RGB

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_sRGB']


def XYZ_to_sRGB(XYZ,
                illuminant=RGB_COLOURSPACES.get('sRGB').whitepoint,
                chromatic_adaptation_method='CAT02',
                transfer_function=True):
    """
    Converts from *CIE XYZ* colourspace to *sRGB* colourspace.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.
    illuminant : array_like, optional
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_method : unicode, optional
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
        *Chromatic adaptation* method.
    transfer_function : bool, optional
        Apply *sRGB* *transfer function*.

    Returns
    -------
    ndarray, (3,)
        *sRGB* colour matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 1].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> XYZ_to_sRGB(XYZ)  # doctest: +ELLIPSIS
    array([ 0.1750135...,  0.3881879...,  0.3216195...])
    """

    sRGB = RGB_COLOURSPACES.get('sRGB')
    return XYZ_to_RGB(XYZ,
                      illuminant,
                      sRGB.whitepoint,
                      sRGB.to_RGB,
                      chromatic_adaptation_method,
                      sRGB.transfer_function if transfer_function else None)
