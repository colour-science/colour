# -*- coding: utf-8 -*-
"""
Von Kries Chromatic Adaptation Model
====================================

Defines the *Von Kries* chromatic adaptation model objects:

-   :func:`colour.adaptation.matrix_chromatic_adaptation_VonKries`
-   :func:`colour.adaptation.chromatic_adaptation_VonKries`

References
----------
-   :cite:`Fairchild2013t` : Fairchild, M. D. (2013). Chromatic Adaptation
    Models. In Color Appearance Models (3rd ed., pp. 4179-4252). Wiley.
    ISBN:B00DAYO8E2
"""

from __future__ import annotations

import numpy as np

from colour.adaptation import CHROMATIC_ADAPTATION_TRANSFORMS
from colour.algebra import matrix_dot, vector_dot
from colour.hints import ArrayLike, Literal, NDArray, Union
from colour.utilities import (
    from_range_1,
    row_as_diagonal,
    to_domain_1,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'matrix_chromatic_adaptation_VonKries',
    'chromatic_adaptation_VonKries',
]


def matrix_chromatic_adaptation_VonKries(
        XYZ_w: ArrayLike,
        XYZ_wr: ArrayLike,
        transform: Union[Literal[
            'Bianco 2010', 'Bianco PC 2010', 'Bradford', 'CAT02 Brill 2008',
            'CAT02', 'CAT16', 'CMCCAT2000', 'CMCCAT97', 'Fairchild', 'Sharp',
            'Von Kries', 'XYZ Scaling'], str] = 'CAT02') -> NDArray:
    """
    Computes the *chromatic adaptation* matrix from test viewing conditions
    to reference viewing conditions.

    Parameters
    ----------
    XYZ_w
        Test viewing conditions *CIE XYZ* tristimulus values of whitepoint.
    XYZ_wr
        Reference viewing conditions *CIE XYZ* tristimulus values of
        whitepoint.
    transform
        Chromatic adaptation transform.

    Returns
    -------
    :class:`numpy.ndarray`
        Chromatic adaptation matrix :math:`M_{cat}`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_w``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wr`` | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2013t`

    Examples
    --------
    >>> XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr)
    ... # doctest: +ELLIPSIS
    array([[ 1.0425738...,  0.0308910..., -0.0528125...],
           [ 0.0221934...,  1.0018566..., -0.0210737...],
           [-0.0011648..., -0.0034205...,  0.7617890...]])

    Using Bradford method:

    >>> XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> method = 'Bradford'
    >>> matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr, method)
    ... # doctest: +ELLIPSIS
    array([[ 1.0479297...,  0.0229468..., -0.0501922...],
           [ 0.0296278...,  0.9904344..., -0.0170738...],
           [-0.0092430...,  0.0150551...,  0.7518742...]])
    """

    XYZ_w = to_domain_1(XYZ_w)
    XYZ_wr = to_domain_1(XYZ_wr)

    transform = validate_method(
        transform, CHROMATIC_ADAPTATION_TRANSFORMS,
        '"{0}" chromatic adaptation transform is invalid, '
        'it must be one of {1}!')

    M = CHROMATIC_ADAPTATION_TRANSFORMS[transform]

    RGB_w = np.einsum('...i,...ij->...j', XYZ_w, np.transpose(M))
    RGB_wr = np.einsum('...i,...ij->...j', XYZ_wr, np.transpose(M))

    D = RGB_wr / RGB_w

    D = row_as_diagonal(D)

    M_CAT = matrix_dot(np.linalg.inv(M), D)
    M_CAT = matrix_dot(M_CAT, M)

    return M_CAT


def chromatic_adaptation_VonKries(
        XYZ: ArrayLike,
        XYZ_w: ArrayLike,
        XYZ_wr: ArrayLike,
        transform: Union[Literal[
            'Bianco 2010', 'Bianco PC 2010', 'Bradford', 'CAT02 Brill 2008',
            'CAT02', 'CAT16', 'CMCCAT2000', 'CMCCAT97', 'Fairchild', 'Sharp',
            'Von Kries', 'XYZ Scaling'], str] = 'CAT02') -> NDArray:
    """
    Adapts given stimulus from test viewing conditions to reference viewing
    conditions.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of stimulus to adapt.
    XYZ_w
        Test viewing conditions *CIE XYZ* tristimulus values of whitepoint.
    XYZ_wr
        Reference viewing conditions *CIE XYZ* tristimulus values of
        whitepoint.
    transform
        Chromatic adaptation transform.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ_c* tristimulus values of the stimulus corresponding colour.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_n``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_r``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_c``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2013t`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr)  # doctest: +ELLIPSIS
    array([ 0.2163881...,  0.1257    ,  0.0384749...])

    Using Bradford method:

    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> transform = 'Bradford'
    >>> chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr, transform)
    ... # doctest: +ELLIPSIS
    array([ 0.2166600...,  0.1260477...,  0.0385506...])
    """

    XYZ = to_domain_1(XYZ)

    M_CAT = matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr, transform)
    XYZ_a = vector_dot(M_CAT, XYZ)

    return from_range_1(XYZ_a)
