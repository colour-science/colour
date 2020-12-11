# -*- coding: utf-8 -*-
"""
DaVinci Intermediate Log Encoding
============================

Defines the *DaVinci Intermediate* log encoding:

-   :func:`colour.models.log_encoding_DaVinciIntermediate`
-   :func:`colour.models.log_decoding_DaVinciIntermediate`

References
----------
-   :cite:'DaVinci Wide Gamut Intermediate' (pp.4-6).
    https://documents.blackmagicdesign.com/InformationNotes/DaVinci_Resolve_17_Wide_Gamut_Intermediate.pdf?_v=1607414410000
"""

import numpy as np

from colour.utilities import Structure, as_float, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CONSTANTS_DAVINTERMEDIATE', 'log_encoding_DaVinciIntermediate',
    'log_decoding_DaVinciIntermediate'
]

CONSTANTS_DAVINTERMEDIATE = Structure(
    di_a=0.0075,
    di_b=7.0,
    di_c=0.07329248,
    di_m=10.44426855,
    di_lin_cut=0.00262409,
    di_log_cut=0.02740668)
"""
*DaVinci Intermediate* colourspace constants.

CONSTANTS_DAVINTERMEDIATE : Structure
"""


def log_encoding_DaVinciIntermediate(L_in,
                                     constants=CONSTANTS_DAVINTERMEDIATE,
                                     **kwargs):
    """
    Defines the *DaVinci Intermediate* log encoding curve
    / opto-electronic transfer function.

    Parameters
    ----------
    L_in : numeric or array_like
        Linear reflection data :math`L_{in}`.

    constants : Structure, optional
        *DaVinci Intermediate* constants.

     Other Parameters
     ----------------
     \\**kwargs : dict, optional
         Keywords arguments for deprecation management.

     Returns
     -------
    numeric or ndarray
        Non-linear data :math:`V_{out}`.

     Notes
     -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_in``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V_out``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

     References
     ----------
-   :cite:'DaVinci Wide Gamut Intermediate' (pp.5).
    https://documents.blackmagicdesign.com/InformationNotes/DaVinci_Resolve_17_Wide_Gamut_Intermediate.pdf?_v=1607414410000

     Examples
     --------
     >>> log_encoding_DaVinciIntermediate(0.18)  # doctest: +ELLIPSIS
     0.33604327238485526...

    >>> log_encoding_DaVinciIntermediate(100.0)
    1.0...

    """

    L_in = to_domain_1(L_in)
    cutlin = constants.di_lin_cut
    a = constants.di_a
    b = constants.di_b
    c = constants.di_c
    m = constants.di_m

    V_out = np.where(
        L_in <= cutlin,
        L_in * m,
        c * (np.log2(L_in + a) + b),
    )

    return as_float(from_range_1(V_out))


def log_decoding_DaVinciIntermediate(V_in,
                                     constants=CONSTANTS_DAVINTERMEDIATE,
                                     **kwargs):
    """
    Defines the the *DaVinci Intermediate* log decoding curve
    / electro-optical transfer function.

    Parameters
    ----------
      numeric or ndarray
        Non-linear data :math:`V_{in}`.

   constants : Structure, optional
        *DaVinci Intermediate* constants.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    L_out : numeric or array_like
        Linear reflection data :math`L_{out}`.

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
-   :cite:'DaVinci Wide Gamut Intermediate' (pp.5).
    https://documents.blackmagicdesign.com/InformationNotes/DaVinci_Resolve_17_Wide_Gamut_Intermediate.pdf?_v=1607414410000

    Examples
    --------
    >>> log_decoding_DaVinciIntermediate(0.903125) # doctest: +ELLIPSIS
    40.0...
    """

    V_in = to_domain_1(V_in)
    logcut = constants.di_log_cut
    a = constants.di_a
    b = constants.di_b
    c = constants.di_c
    m = constants.di_m
    L_out = np.where(
        V_in <= logcut,
        V_in / m,
        (2 ** ((V_in / c) - b)) - a,
    )
    return as_float(from_range_1(L_out))
