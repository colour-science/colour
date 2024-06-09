"""
Pointer's Gamut Volume Computations
===================================

Define the objects related to *Pointer's Gamut* volume computations.
"""

from __future__ import annotations

from colour.constants import EPSILON
from colour.hints import ArrayLike, NDArrayFloat
from colour.models import (
    CCS_ILLUMINANT_POINTER_GAMUT,
    DATA_POINTER_GAMUT_VOLUME,
    Lab_to_XYZ,
    LCHab_to_Lab,
)
from colour.volume import is_within_mesh_volume

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_within_pointer_gamut",
]


def is_within_pointer_gamut(
    XYZ: ArrayLike, tolerance: float = 100 * EPSILON
) -> NDArrayFloat:
    """
    Return whether given *CIE XYZ* tristimulus values are within Pointer's
    Gamut volume.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    tolerance
        Tolerance allowed in the inside-triangle check.

    Returns
    -------
    :class:`numpy.ndarray`
        Whether given *CIE XYZ* tristimulus values are within Pointer's Gamut
        volume.

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
        LCHab_to_Lab(DATA_POINTER_GAMUT_VOLUME), CCS_ILLUMINANT_POINTER_GAMUT
    )

    return is_within_mesh_volume(XYZ, XYZ_p, tolerance)
