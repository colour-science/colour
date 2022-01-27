# -*- coding: utf-8 -*-
"""
CMCCAT2000 Chromatic Adaptation Model
=====================================

Defines the *CMCCAT2000* chromatic adaptation model objects:

-   :class:`colour.adaptation.InductionFactors_CMCCAT2000`
-   :class:`colour.VIEWING_CONDITIONS_CMCCAT2000`
-   :func:`colour.adaptation.chromatic_adaptation_forward_CMCCAT2000`
-   :func:`colour.adaptation.chromatic_adaptation_inverse_CMCCAT2000`
-   :func:`colour.adaptation.chromatic_adaptation_CMCCAT2000`

References
----------
-   :cite:`Li2002a` : Li, C., Luo, M. R., Rigg, B., & Hunt, R. W. G. (2002).
    CMC 2000 chromatic adaptation transform: CMCCAT2000. Color Research &
    Application, 27(1), 49-58. doi:10.1002/col.10005
-   :cite:`Westland2012k` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    CMCCAT2000. In Computational Colour Science Using MATLAB (2nd ed., pp.
    83-86). ISBN:978-0-470-66569-5
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from colour.adaptation import CAT_CMCCAT2000
from colour.algebra import vector_dot
from colour.hints import (
    ArrayLike,
    Floating,
    FloatingOrArrayLike,
    Literal,
    NDArray,
    Union,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float_array,
    from_range_100,
    to_domain_100,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CAT_INVERSE_CMCCAT2000',
    'InductionFactors_CMCCAT2000',
    'VIEWING_CONDITIONS_CMCCAT2000',
    'chromatic_adaptation_forward_CMCCAT2000',
    'chromatic_adaptation_inverse_CMCCAT2000',
    'chromatic_adaptation_CMCCAT2000',
]

CAT_INVERSE_CMCCAT2000: NDArray = np.linalg.inv(CAT_CMCCAT2000)
"""
Inverse *CMCCAT2000* chromatic adaptation transform.

CAT_INVERSE_CMCCAT2000
"""


class InductionFactors_CMCCAT2000(NamedTuple):
    """
    *CMCCAT2000* chromatic adaptation model induction factors.

    Parameters
    ----------
    F
        :math:`F` surround condition.

    References
    ----------
    :cite:`Li2002a`, :cite:`Westland2012k`
    """

    F: Floating


VIEWING_CONDITIONS_CMCCAT2000: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        'Average': InductionFactors_CMCCAT2000(1),
        'Dim': InductionFactors_CMCCAT2000(0.8),
        'Dark': InductionFactors_CMCCAT2000(0.8)
    })
VIEWING_CONDITIONS_CMCCAT2000.__doc__ = """
Reference *CMCCAT2000* chromatic adaptation model viewing conditions.

References
----------
:cite:`Li2002a`, :cite:`Westland2012k`
"""


def chromatic_adaptation_forward_CMCCAT2000(
        XYZ: ArrayLike,
        XYZ_w: ArrayLike,
        XYZ_wr: ArrayLike,
        L_A1: FloatingOrArrayLike,
        L_A2: FloatingOrArrayLike,
        surround: InductionFactors_CMCCAT2000 = VIEWING_CONDITIONS_CMCCAT2000[
            'Average']) -> NDArray:
    """
    Adapts given stimulus *CIE XYZ* tristimulus values from test viewing
    conditions to reference viewing conditions using *CMCCAT2000* forward
    chromatic adaptation model.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the stimulus to adapt.
    XYZ_w
        Test viewing condition *CIE XYZ* tristimulus values of the whitepoint.
    XYZ_wr
        Reference viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    L_A1
        Luminance of test adapting field :math:`L_{A1}` in :math:`cd/m^2`.
    L_A2
        Luminance of reference adapting field :math:`L_{A2}` in :math:`cd/m^2`.
    surround
        Surround viewing conditions induction factors.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ_c* tristimulus values of the stimulus corresponding colour.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_w``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wr`` | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_c``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Li2002a`, :cite:`Westland2012k`

    Examples
    --------
    >>> XYZ = np.array([22.48, 22.74, 8.54])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2)
    ... # doctest: +ELLIPSIS
    array([ 19.5269832...,  23.0683396...,  24.9717522...])
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    XYZ_wr = to_domain_100(XYZ_wr)
    L_A1 = as_float_array(L_A1)
    L_A2 = as_float_array(L_A2)

    RGB = vector_dot(CAT_CMCCAT2000, XYZ)
    RGB_w = vector_dot(CAT_CMCCAT2000, XYZ_w)
    RGB_wr = vector_dot(CAT_CMCCAT2000, XYZ_wr)

    D = (surround.F * (0.08 * np.log10(0.5 * (L_A1 + L_A2)) + 0.76 - 0.45 *
                       (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = np.clip(D, 0, 1)
    a = D * XYZ_w[..., 1] / XYZ_wr[..., 1]

    RGB_c = (
        RGB * (a[..., np.newaxis] * (RGB_wr / RGB_w) + 1 - D[..., np.newaxis]))
    XYZ_c = vector_dot(CAT_INVERSE_CMCCAT2000, RGB_c)

    return from_range_100(XYZ_c)


def chromatic_adaptation_inverse_CMCCAT2000(
        XYZ_c: ArrayLike,
        XYZ_w: ArrayLike,
        XYZ_wr: ArrayLike,
        L_A1: FloatingOrArrayLike,
        L_A2: FloatingOrArrayLike,
        surround: InductionFactors_CMCCAT2000 = VIEWING_CONDITIONS_CMCCAT2000[
            'Average']) -> NDArray:
    """
    Adapts given stimulus corresponding colour *CIE XYZ* tristimulus values
    from reference viewing conditions to test viewing conditions using
    *CMCCAT2000* inverse chromatic adaptation model.

    Parameters
    ----------
    XYZ_c
        *CIE XYZ* tristimulus values of the stimulus to adapt.
    XYZ_w
        Test viewing condition *CIE XYZ* tristimulus values of the whitepoint.
    XYZ_wr
        Reference viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    L_A1
        Luminance of test adapting field :math:`L_{A1}` in :math:`cd/m^2`.
    L_A2
        Luminance of reference adapting field :math:`L_{A2}` in :math:`cd/m^2`.
    surround
        Surround viewing conditions induction factors.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ_c* tristimulus values of the adapted stimulus.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_c``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_w``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wr`` | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Li2002a`, :cite:`Westland2012k`

    Examples
    --------
    >>> XYZ_c = np.array([19.53, 23.07, 24.97])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_inverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1,
    ...                                         L_A2)
    ... # doctest: +ELLIPSIS
    array([ 22.4839876...,  22.7419485...,   8.5393392...])
    """

    XYZ_c = to_domain_100(XYZ_c)
    XYZ_w = to_domain_100(XYZ_w)
    XYZ_wr = to_domain_100(XYZ_wr)
    L_A1 = as_float_array(L_A1)
    L_A2 = as_float_array(L_A2)

    RGB_c = vector_dot(CAT_CMCCAT2000, XYZ_c)
    RGB_w = vector_dot(CAT_CMCCAT2000, XYZ_w)
    RGB_wr = vector_dot(CAT_CMCCAT2000, XYZ_wr)

    D = (surround.F * (0.08 * np.log10(0.5 * (L_A1 + L_A2)) + 0.76 - 0.45 *
                       (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = np.clip(D, 0, 1)
    a = D * XYZ_w[..., 1] / XYZ_wr[..., 1]

    RGB = (RGB_c / (a[..., np.newaxis] *
                    (RGB_wr / RGB_w) + 1 - D[..., np.newaxis]))
    XYZ = vector_dot(CAT_INVERSE_CMCCAT2000, RGB)

    return from_range_100(XYZ)


def chromatic_adaptation_CMCCAT2000(
        XYZ: ArrayLike,
        XYZ_w: ArrayLike,
        XYZ_wr: ArrayLike,
        L_A1: FloatingOrArrayLike,
        L_A2: FloatingOrArrayLike,
        surround: InductionFactors_CMCCAT2000 = VIEWING_CONDITIONS_CMCCAT2000[
            'Average'],
        direction: Union[Literal['Forward', 'Inverse'], str] = 'Forward'
) -> NDArray:
    """
    Adapts given stimulus *CIE XYZ* tristimulus values using given viewing
    conditions.

    This definition is a convenient wrapper around
    :func:`colour.adaptation.chromatic_adaptation_forward_CMCCAT2000` and
    :func:`colour.adaptation.chromatic_adaptation_inverse_CMCCAT2000`.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the stimulus to adapt.
    XYZ_w
        Source viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    XYZ_wr
        Target viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    L_A1
        Luminance of test adapting field :math:`L_{A1}` in :math:`cd/m^2`.
    L_A2
        Luminance of reference adapting field :math:`L_{A2}` in :math:`cd/m^2`.
    surround
        Surround viewing conditions induction factors.
    direction
        Chromatic adaptation direction.

    Returns
    -------
    :class:`numpy.ndarray`
        Adapted stimulus *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_w``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wr`` | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Li2002a`, :cite:`Westland2012k`

    Examples
    --------
    >>> XYZ = np.array([22.48, 22.74, 8.54])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_CMCCAT2000(
    ...     XYZ, XYZ_w, XYZ_wr, L_A1, L_A2, direction='Forward')
    ... # doctest: +ELLIPSIS
    array([ 19.5269832...,  23.0683396...,  24.9717522...])

    Using the *CMCCAT2000* inverse model:

    >>> XYZ = np.array([19.52698326, 23.06833960, 24.97175229])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_CMCCAT2000(
    ...     XYZ, XYZ_w, XYZ_wr, L_A1, L_A2, direction='Inverse')
    ... # doctest: +ELLIPSIS
    array([ 22.48,  22.74,   8.54])
    """

    direction = validate_method(
        direction, ['Forward', 'Inverse'],
        '"{0}" direction is invalid, it must be one of {1}!')

    if direction == 'forward':
        return chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr,
                                                       L_A1, L_A2, surround)
    else:
        return chromatic_adaptation_inverse_CMCCAT2000(XYZ, XYZ_w, XYZ_wr,
                                                       L_A1, L_A2, surround)
