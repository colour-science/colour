# -*- coding: utf-8 -*-
"""
Pointer's Gamut Volume Computations
===================================

Defines objects related to *Pointer's Gamut* volume computations.

See Also
--------
`Pointer's Gamut Volume Computations Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/volume/pointer_gamut.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.models import (Lab_to_XYZ, LCHab_to_Lab, POINTER_GAMUT_DATA,
                           POINTER_GAMUT_ILLUMINANT)
from colour.volume import is_within_mesh_volume

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['is_within_pointer_gamut']


def is_within_pointer_gamut(XYZ, tolerance=None):
    """
    Returns if given *CIE XYZ* tristimulus values are within Pointer's Gamut
    volume.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    tolerance : numeric, optional
        Tolerance allowed in the inside-triangle check.

    Returns
    -------
    bool
        Is within Pointer's Gamut.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> is_within_pointer_gamut(np.array([0.3205, 0.4131, 0.5100]))
    array(True, dtype=bool)
    >>> a = np.array([[0.3205, 0.4131, 0.5100], [0.0005, 0.0031, 0.0010]])
    >>> is_within_pointer_gamut(a)
    array([ True, False], dtype=bool)
    """

    XYZ_p = Lab_to_XYZ(
        LCHab_to_Lab(POINTER_GAMUT_DATA), POINTER_GAMUT_ILLUMINANT)

    return is_within_mesh_volume(XYZ, XYZ_p, tolerance)
