# -*- coding: utf-8 -*-
"""
Blackmagic Design Transfer Functions
====================================

Defines the *Blackmagic Design* colour component transfer functions:

-   :func:`colour.models.oetf_BlackmagicFilmGeneration5`
-   :func:`colour.models.oetf_inverse_BlackmagicFilmGeneration5`

References
----------
-   :cite:`BlackmagicDesign2021` : Blackmagic Design. (2021). Blackmagic
    Generation 5 Color Science. https://drive.google.com/file/d/\
1FF5WO2nvI9GEWb4_EntrBoV9ZIuFToZd/view
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
    'CONSTANTS_BLACKMAGIC_FILM_GENERATION_5',
    'oetf_BlackmagicFilmGeneration5',
    'oetf_inverse_BlackmagicFilmGeneration5',
]

CONSTANTS_BLACKMAGIC_FILM_GENERATION_5: Structure = Structure(
    A=0.08692876065491224,
    B=0.005494072432257808,
    C=0.5300133392291939,
    D=8.283605932402494,
    E=0.09246575342465753,
    LIN_CUT=0.005)
"""
*Blackmagic Film Generation 5* colour component transfer functions constants.
"""


def oetf_BlackmagicFilmGeneration5(
        x: FloatingOrArrayLike,
        constants: Structure = CONSTANTS_BLACKMAGIC_FILM_GENERATION_5
) -> FloatingOrNDArray:
    """
    Defines the *Blackmagic Film Generation 5* opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear light value :math`x`.
    constants
        *Blackmagic Film Generation 5* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Encoded value :math:`y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`BlackmagicDesign2021`

    Examples
    --------
    >>> oetf_BlackmagicFilmGeneration5(0.18)  # doctest: +ELLIPSIS
    0.3835616...
    """

    x = to_domain_1(x)

    A = constants.A
    B = constants.B
    C = constants.C
    D = constants.D
    E = constants.E
    LIN_CUT = constants.LIN_CUT

    V_out = np.where(
        x < LIN_CUT,
        D * x + E,
        A * np.log(x + B) + C,
    )

    return as_float(from_range_1(V_out))


def oetf_inverse_BlackmagicFilmGeneration5(
        y: FloatingOrArrayLike,
        constants: Structure = CONSTANTS_BLACKMAGIC_FILM_GENERATION_5
) -> FloatingOrNDArray:
    """
    Defines the *Blackmagic Film Generation 5* inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    y
        Encoded value :math:`y`.
    constants
        *Blackmagic Film Generation 5* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear light value :math`x`.

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
    :cite:`BlackmagicDesign2021`

    Examples
    --------
    >>> oetf_inverse_BlackmagicFilmGeneration5(0.38356164383561653)
    ... # doctest: +ELLIPSIS
    0.1799999...
    """

    y = to_domain_1(y)

    A = constants.A
    B = constants.B
    C = constants.C
    D = constants.D
    E = constants.E
    LIN_CUT = constants.LIN_CUT

    LOG_CUT = D * LIN_CUT + E

    x = np.where(
        y < LOG_CUT,
        (y - E) / D,
        np.exp((y - C) / A) - B,
    )
    return as_float(from_range_1(x))
