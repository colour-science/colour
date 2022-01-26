# -*- coding: utf-8 -*-
"""
Digital Cinema Distribution Master (DCDM)
=========================================

Defines the *DCDM* electro-optical transfer function (EOTF) and its
inverse:

-   :func:`colour.models.eotf_inverse_DCDM`
-   :func:`colour.models.eotf_DCDM`

References
----------
-   :cite:`DigitalCinemaInitiatives2007b` : Digital Cinema Initiatives. (2007).
    Digital Cinema System Specification - Version 1.1.
    http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import (
    Boolean,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    IntegerOrArrayLike,
    IntegerOrNDArray,
    Union,
)
from colour.utilities import as_float, as_int, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'eotf_inverse_DCDM',
    'eotf_DCDM',
]


def eotf_inverse_DCDM(XYZ: FloatingOrArrayLike, out_int: Boolean = False
                      ) -> Union[FloatingOrNDArray, IntegerOrNDArray]:
    """
    Defines the *DCDM* inverse electro-optical transfer function (EOTF).

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    out_int
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.integer` or :class:`numpy.ndarray`
        Non-linear *CIE XYZ'* tristimulus values.

    Warnings
    --------
    *DCDM* is an absolute transfer function.

    Notes
    -----

    -   *DCDM* is an absolute transfer function, thus the domain and range
        values for the *Reference* and *1* scales are only indicative that the
        data is not affected by scale transformations.

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``XYZ``        | ``UN``                | ``UN``        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``XYZ_p``      | ``UN``                | ``UN``        |
    +----------------+-----------------------+---------------+

    \\* This definition has an output integer switch, thus the domain-range
    scale information is only given for the floating point mode.

    References
    ----------
    :cite:`DigitalCinemaInitiatives2007b`

    Examples
    --------
    >>> eotf_inverse_DCDM(0.18)  # doctest: +ELLIPSIS
    0.1128186...
    >>> eotf_inverse_DCDM(0.18, out_int=True)
    462
    """

    XYZ = to_domain_1(XYZ)

    XYZ_p = spow(XYZ / 52.37, 1 / 2.6)

    if out_int:
        return as_int(np.round(4095 * XYZ_p))
    else:
        return as_float(from_range_1(XYZ_p))


def eotf_DCDM(XYZ_p: Union[FloatingOrArrayLike, IntegerOrArrayLike],
              in_int: Boolean = False) -> FloatingOrNDArray:
    """
    Defines the *DCDM* electro-optical transfer function (EOTF).

    Parameters
    ----------
    XYZ_p
        Non-linear *CIE XYZ'* tristimulus values.
    in_int
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Warnings
    --------
    *DCDM* is an absolute transfer function.

    Notes
    -----

    -   *DCDM* is an absolute transfer function, thus the domain and range
        values for the *Reference* and *1* scales are only indicative that the
        data is not affected by scale transformations.

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``XYZ_p``      | ``UN``                | ``UN``        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``XYZ``        | ``UN``                | ``UN``        |
    +----------------+-----------------------+---------------+

    \\* This definition has an input integer switch, thus the domain-range
    scale information is only given for the floating point mode.

    References
    ----------
    :cite:`DigitalCinemaInitiatives2007b`

    Examples
    --------
    >>> eotf_DCDM(0.11281860951766724)  # doctest: +ELLIPSIS
    0.18...
    >>> eotf_DCDM(462, in_int=True)  # doctest: +ELLIPSIS
    0.18...
    """

    XYZ_p = to_domain_1(XYZ_p)

    if in_int:
        XYZ_p = XYZ_p / 4095

    XYZ = 52.37 * spow(XYZ_p, 2.6)

    return as_float(from_range_1(XYZ))
