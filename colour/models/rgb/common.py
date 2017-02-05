#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common RGB Colour Models Utilities
==================================

Defines various RGB colour models common utilities.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.models.rgb import RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_RGB

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_sRGB', 'sRGB_to_XYZ']


def XYZ_to_sRGB(XYZ,
                illuminant=RGB_COLOURSPACES['sRGB'].whitepoint,
                chromatic_adaptation_transform='CAT02',
                apply_encoding_cctf=True):
    """
    Converts from *CIE XYZ* tristimulus values to *sRGB* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like, optional
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    apply_encoding_cctf : bool, optional
        Apply *sRGB* encoding colour component transfer function /
        opto-electronic transfer function.

    Returns
    -------
    ndarray
        *sRGB* colour array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_sRGB(XYZ)  # doctest: +ELLIPSIS
    array([ 0.1749881...,  0.3881947...,  0.3216031...])
    """

    sRGB = RGB_COLOURSPACES['sRGB']
    return XYZ_to_RGB(XYZ,
                      illuminant,
                      sRGB.whitepoint,
                      sRGB.XYZ_to_RGB_matrix,
                      chromatic_adaptation_transform,
                      sRGB.encoding_cctf if apply_encoding_cctf else None)


def sRGB_to_XYZ(RGB,
                illuminant=RGB_COLOURSPACES['sRGB'].whitepoint,
                chromatic_adaptation_method='CAT02',
                apply_decoding_cctf=True):
    """
    Converts from *sRGB* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB : array_like
        *sRGB* colourspace array.
    illuminant : array_like, optional
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_method : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* method.
    apply_decoding_cctf : bool, optional
        Apply *sRGB* decoding colour component transfer function  /
        electro-optical transfer function.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *RGB* colourspace array is in domain [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array([0.17498172, 0.38818743, 0.32159978])
    >>> sRGB_to_XYZ(RGB)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008...,  0.0955831...])
    """

    sRGB = RGB_COLOURSPACES['sRGB']
    return RGB_to_XYZ(RGB,
                      sRGB.whitepoint,
                      illuminant,
                      sRGB.RGB_to_XYZ_matrix,
                      chromatic_adaptation_method,
                      sRGB.decoding_cctf if apply_decoding_cctf else None)
