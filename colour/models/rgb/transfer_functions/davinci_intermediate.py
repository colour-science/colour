# -*- coding: utf-8 -*-
"""
DaVinci Intermediate
====================

Defines the *DaVinci Intermediate* opto-electrical transfer function
(OETF) and its inverse:

-   :func:`colour.models.oetf_DaVinciIntermediate`
-   :func:`colour.models.oetf_inverse_DaVinciIntermediate`

References
----------
-   :cite:`BlackmagicDesign2020a` : Blackmagic Design. (2020). Wide Gamut
    Intermediate DaVinci Resolve. Retrieved December 12, 2020, from
    https://documents.blackmagicdesign.com/InformationNotes/\
DaVinci_Resolve_17_Wide_Gamut_Intermediate.pdf?_v=1607414410000
"""

from __future__ import annotations

import numpy as np

from colour.hints import FloatingOrArrayLike, FloatingOrNDArray
from colour.utilities import Structure, as_float, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CONSTANTS_DAVINCI_INTERMEDIATE',
    'oetf_DaVinciIntermediate',
    'oetf_inverse_DaVinciIntermediate',
]

CONSTANTS_DAVINCI_INTERMEDIATE: Structure = Structure(
    DI_A=0.0075,
    DI_B=7.0,
    DI_C=0.07329248,
    DI_M=10.44426855,
    DI_LIN_CUT=0.00262409,
    DI_LOG_CUT=0.02740668)
"""
*DaVinci Intermediate* colour component transfer functions constants.
"""


def oetf_DaVinciIntermediate(
        L: FloatingOrArrayLike,
        constants: Structure = CONSTANTS_DAVINCI_INTERMEDIATE
) -> FloatingOrNDArray:
    """
    Defines the *DaVinci Intermediate* opto-electronic transfer function.

    Parameters
    ----------
    L
        Linear light value :math`L`.
    constants
        *DaVinci Intermediate* colour component transfer function constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Encoded value :math:`V`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`BlackmagicDesign2020a`

    Examples
    --------
    >>> oetf_DaVinciIntermediate(0.18)  # doctest: +ELLIPSIS
    0.3360432...
    """

    L = to_domain_1(L)

    DI_LIN_CUT = constants.DI_LIN_CUT
    DI_A = constants.DI_A
    DI_B = constants.DI_B
    DI_C = constants.DI_C
    DI_M = constants.DI_M

    V_out = np.where(
        L <= DI_LIN_CUT,
        L * DI_M,
        DI_C * (np.log2(L + DI_A) + DI_B),
    )

    return as_float(from_range_1(V_out))


def oetf_inverse_DaVinciIntermediate(
        V: FloatingOrArrayLike,
        constants: Structure = CONSTANTS_DAVINCI_INTERMEDIATE
) -> FloatingOrNDArray:
    """
    Defines the *DaVinci Intermediate* inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    V
        Encoded value :math:`V`.
    constants
        *DaVinci Intermediate* colour component transfer function constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear light value :math`L`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`BlackmagicDesign2020a`

    Examples
    --------
    >>> oetf_inverse_DaVinciIntermediate(0.336043272384855)
    ... # doctest: +ELLIPSIS
    0.1799999...
    """

    V = to_domain_1(V)

    DI_LOG_CUT = constants.DI_LOG_CUT
    DI_A = constants.DI_A
    DI_B = constants.DI_B
    DI_C = constants.DI_C
    DI_M = constants.DI_M

    L_out = np.where(
        V <= DI_LOG_CUT,
        V / DI_M,
        2 ** ((V / DI_C) - DI_B) - DI_A,
    )
    return as_float(from_range_1(L_out))
