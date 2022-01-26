# -*- coding: utf-8 -*-
"""
ITU-R BT.601-7
==============

Defines the *ITU-R BT.601-7* opto-electrical transfer function (OETF)
and its inverse:

-   :func:`colour.models.oetf_BT601`
-   :func:`colour.models.oetf_inverse_BT601`

References
----------
-   :cite:`InternationalTelecommunicationUnion2011f` : International
    Telecommunication Union. (2011). Recommendation ITU-R BT.601-7 - Studio
    encoding parameters of digital television for standard 4:3 and wide-screen
    16:9 aspect ratios.
    http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.601-7-201103-I!!PDF-E.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import FloatingOrArrayLike, FloatingOrNDArray
from colour.utilities import (
    as_float,
    domain_range_scale,
    from_range_1,
    to_domain_1,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'oetf_BT601',
    'oetf_inverse_BT601',
]


def oetf_BT601(L: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines *Recommendation ITU-R BT.601-7* opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    L
        *Luminance* :math:`L` of the image.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding electrical signal :math:`E`.

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
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2011f`

    Examples
    --------
    >>> oetf_BT601(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    """

    L = to_domain_1(L)

    E = np.where(L < 0.018, L * 4.5, 1.099 * spow(L, 0.45) - 0.099)

    return as_float(from_range_1(E))


def oetf_inverse_BT601(E: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines *Recommendation ITU-R BT.601-7* inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    E
        Electrical signal :math:`E`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding *luminance* :math:`L` of the image.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2011f`

    Examples
    --------
    >>> oetf_inverse_BT601(0.409007728864150)  # doctest: +ELLIPSIS
    0.1...
    """

    E = to_domain_1(E)

    with domain_range_scale('ignore'):
        L = np.where(
            E < oetf_BT601(0.018),
            E / 4.5,
            spow((E + 0.099) / 1.099, 1 / 0.45),
        )

    return as_float(from_range_1(L))
