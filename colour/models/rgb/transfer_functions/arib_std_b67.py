# -*- coding: utf-8 -*-
"""
ARIB STD-B67 (Hybrid Log-Gamma)
===============================

Defines the *ARIB STD-B67 (Hybrid Log-Gamma)* opto-electrical transfer function
(OETF) and its inverse:

-   :func:`colour.models.oetf_ARIBSTDB67`
-   :func:`colour.models.oetf_inverse_ARIBSTDB67`

References
----------
-   :cite:`AssociationofRadioIndustriesandBusinesses2015a` : Association of
    Radio Industries and Businesses. (2015). Essential Parameter Values for the
    Extended Image Dynamic Range Television (EIDRTV) System for Programme
    Production.
    https://www.arib.or.jp/english/std_tr/broadcasting/desc/std-b67.html
"""

from __future__ import annotations

import numpy as np

from colour.hints import FloatingOrArrayLike, FloatingOrNDArray
from colour.models.rgb.transfer_functions import gamma_function
from colour.utilities import (
    Structure,
    as_float,
    as_float_array,
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
    'CONSTANTS_ARIBSTDB67',
    'oetf_ARIBSTDB67',
    'oetf_inverse_ARIBSTDB67',
]

CONSTANTS_ARIBSTDB67: Structure = Structure(
    a=0.17883277, b=0.28466892, c=0.55991073)
"""
*ARIB STD-B67 (Hybrid Log-Gamma)* constants.
"""


def oetf_ARIBSTDB67(
        E: FloatingOrArrayLike,
        r: FloatingOrArrayLike = 0.5,
        constants: Structure = CONSTANTS_ARIBSTDB67) -> FloatingOrNDArray:
    """
    Defines *ARIB STD-B67 (Hybrid Log-Gamma)* opto-electrical transfer
    function (OETF).

    Parameters
    ----------
    E
        Voltage normalised by the reference white level and proportional to
        the implicit light intensity that would be detected with a reference
        camera color channel R, G, B.
    r
        Video level corresponding to reference white level.
    constants
        *ARIB STD-B67 (Hybrid Log-Gamma)* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Resulting non-linear signal :math:`E'`.

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
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   This definition uses the *mirror* negative number handling mode of
        :func:`colour.models.gamma_function` definition to the sign of negative
        numbers.

    References
    ----------
    :cite:`AssociationofRadioIndustriesandBusinesses2015a`

    Examples
    --------
    >>> oetf_ARIBSTDB67(0.18)  # doctest: +ELLIPSIS
    0.2121320...
    """

    E = to_domain_1(E)
    r = as_float_array(r)

    a = constants.a
    b = constants.b
    c = constants.c

    E_p = np.where(E <= 1, r * gamma_function(E, 0.5, 'mirror'),
                   a * np.log(E - b) + c)

    return as_float(from_range_1(E_p))


def oetf_inverse_ARIBSTDB67(
        E_p: FloatingOrArrayLike,
        r: FloatingOrArrayLike = 0.5,
        constants: Structure = CONSTANTS_ARIBSTDB67) -> FloatingOrNDArray:
    """
    Defines *ARIB STD-B67 (Hybrid Log-Gamma)* inverse opto-electrical transfer
    function (OETF).

    Parameters
    ----------
    E_p
        Non-linear signal :math:`E'`.
    r
        Video level corresponding to reference white level.
    constants
        *ARIB STD-B67 (Hybrid Log-Gamma)* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Voltage :math:`E` normalised by the reference white level and
        proportional to the implicit light intensity that would be detected
        with a reference camera color channel R, G, B.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   This definition uses the *mirror* negative number handling mode of
        :func:`colour.models.gamma_function` definition to the sign of negative
        numbers.

    References
    ----------
    :cite:`AssociationofRadioIndustriesandBusinesses2015a`

    Examples
    --------
    >>> oetf_inverse_ARIBSTDB67(0.212132034355964)  # doctest: +ELLIPSIS
    0.1799999...
    """

    E_p = to_domain_1(E_p)

    a = constants.a
    b = constants.b
    c = constants.c

    with domain_range_scale('ignore'):
        E = np.where(
            E_p <= oetf_ARIBSTDB67(1),
            gamma_function((E_p / r), 2, 'mirror'),
            np.exp((E_p - c) / a) + b,
        )

    return as_float(from_range_1(E))
