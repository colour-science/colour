#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visible Spectrum Volume Computations
====================================

Defines objects related to visible spectrum volume computations.

See Also
--------
`Spectrum Volume Computations Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/volume/spectrum.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.volume import is_within_mesh_volume

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['is_within_visible_spectrum']


def is_within_visible_spectrum(
        XYZ,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        tolerance=None):
    """
    Returns if given *CIE XYZ* tristimulus values are within visible spectrum
    volume / given colour matching functions volume.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    tolerance : numeric, optional
        Tolerance allowed in the inside-triangle check.

    Returns
    -------
    bool
        Is within visible spectrum.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> is_within_visible_spectrum(np.array([0.3205, 0.4131, 0.51]))
    array(True, dtype=bool)
    >>> a = np.array([[0.3205, 0.4131, 0.51],
    ...               [-0.0005, 0.0031, 0.001]])
    >>> is_within_visible_spectrum(a)
    array([ True, False], dtype=bool)
    """

    return is_within_mesh_volume(XYZ, cmfs.values, tolerance)
