# -*- coding: utf-8 -*-
"""
CMCCAT2000 Chromatic Adaptation Model
=====================================

Defines *CMCCAT2000* chromatic adaptation model objects:

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

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.adaptation import CAT_CMCCAT2000
from colour.utilities import (CaseInsensitiveMapping, as_float_array,
                              vector_dot, from_range_100, to_domain_100)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CAT_INVERSE_CMCCAT2000', 'InductionFactors_CMCCAT2000',
    'VIEWING_CONDITIONS_CMCCAT2000', 'chromatic_adaptation_forward_CMCCAT2000',
    'chromatic_adaptation_inverse_CMCCAT2000',
    'chromatic_adaptation_CMCCAT2000'
]

CAT_INVERSE_CMCCAT2000 = np.linalg.inv(CAT_CMCCAT2000)
"""
Inverse *CMCCAT2000* chromatic adaptation transform.

CAT_INVERSE_CMCCAT2000 : array_like, (3, 3)
"""


class InductionFactors_CMCCAT2000(
        namedtuple('InductionFactors_CMCCAT2000', ('F', ))):
    """
    *CMCCAT2000* chromatic adaptation model induction factors.

    Parameters
    ----------
    F : numeric or array_like
        :math:`F` surround condition.

    References
    ----------
    :cite:`Li2002a`, :cite:`Westland2012k`
    """


VIEWING_CONDITIONS_CMCCAT2000 = CaseInsensitiveMapping({
    'Average': InductionFactors_CMCCAT2000(1),
    'Dim': InductionFactors_CMCCAT2000(0.8),
    'Dark': InductionFactors_CMCCAT2000(0.8)
})
VIEWING_CONDITIONS_CMCCAT2000.__doc__ = """
Reference *CMCCAT2000* chromatic adaptation model viewing conditions.

References
----------
:cite:`Li2002a`, :cite:`Westland2012k`

VIEWING_CONDITIONS_CMCCAT2000 : CaseInsensitiveMapping
    ('Average', 'Dim', 'Dark')
"""


def chromatic_adaptation_forward_CMCCAT2000(
        XYZ,
        XYZ_w,
        XYZ_wr,
        L_A1,
        L_A2,
        surround=VIEWING_CONDITIONS_CMCCAT2000['Average']):
    """
    Adapts given stimulus *CIE XYZ* tristimulus values from test viewing
    conditions to reference viewing conditions using *CMCCAT2000* forward
    chromatic adaptation model.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of the stimulus to adapt.
    XYZ_w : array_like
        Test viewing condition *CIE XYZ* tristimulus values of the whitepoint.
    XYZ_wr : array_like
        Reference viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    L_A1 : numeric or array_like
        Luminance of test adapting field :math:`L_{A1}` in :math:`cd/m^2`.
    L_A2 : numeric or array_like
        Luminance of reference adapting field :math:`L_{A2}` in :math:`cd/m^2`.
    surround : InductionFactors_CMCCAT2000, optional
        Surround viewing conditions induction factors.

    Returns
    -------
    ndarray
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
        XYZ_c,
        XYZ_w,
        XYZ_wr,
        L_A1,
        L_A2,
        surround=VIEWING_CONDITIONS_CMCCAT2000['Average']):
    """
    Adapts given stimulus corresponding colour *CIE XYZ* tristimulus values
    from reference viewing conditions to test viewing conditions using
    *CMCCAT2000* inverse chromatic adaptation model.

    Parameters
    ----------
    XYZ_c : array_like
        *CIE XYZ* tristimulus values of the stimulus to adapt.
    XYZ_w : array_like
        Test viewing condition *CIE XYZ* tristimulus values of the whitepoint.
    XYZ_wr : array_like
        Reference viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    L_A1 : numeric or array_like
        Luminance of test adapting field :math:`L_{A1}` in :math:`cd/m^2`.
    L_A2 : numeric or array_like
        Luminance of reference adapting field :math:`L_{A2}` in :math:`cd/m^2`.
    surround : InductionFactors_CMCCAT2000, optional
        Surround viewing conditions induction factors.

    Returns
    -------
    ndarray
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
        XYZ,
        XYZ_w,
        XYZ_wr,
        L_A1,
        L_A2,
        surround=VIEWING_CONDITIONS_CMCCAT2000['Average'],
        direction='Forward'):
    """
    Adapts given stimulus *CIE XYZ* tristimulus values using given viewing
    conditions.

    This definition is a convenient wrapper around
    :func:`colour.adaptation.chromatic_adaptation_forward_CMCCAT2000` and
    :func:`colour.adaptation.chromatic_adaptation_inverse_CMCCAT2000`.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of the stimulus to adapt.
    XYZ_w : array_like
        Source viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    XYZ_wr : array_like
        Target viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    L_A1 : numeric or array_like
        Luminance of test adapting field :math:`L_{A1}` in :math:`cd/m^2`.
    L_A2 : numeric or array_like
        Luminance of reference adapting field :math:`L_{A2}` in :math:`cd/m^2`.
    surround : InductionFactors_CMCCAT2000, optional
        Surround viewing conditions induction factors.
    direction : unicode, optional
        **{'Forward', 'Inverse'}**,
        Chromatic adaptation direction.

    Returns
    -------
    ndarray
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

    if direction.lower() == 'forward':
        return chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr,
                                                       L_A1, L_A2, surround)
    else:
        return chromatic_adaptation_inverse_CMCCAT2000(XYZ, XYZ_w, XYZ_wr,
                                                       L_A1, L_A2, surround)
