# -*- coding: utf-8 -*-
"""
CMCCAT2000 Chromatic Adaptation Model
=====================================

Defines *CMCCAT2000* chromatic adaptation model objects:

-   :class:`colour.adaptation.CMCCAT2000_InductionFactors`
-   :class:`colour.CMCCAT2000_VIEWING_CONDITIONS`
-   :func:`colour.adaptation.chromatic_adaptation_forward_CMCCAT2000`
-   :func:`colour.adaptation.chromatic_adaptation_reverse_CMCCAT2000`
-   :func:`colour.adaptation.chromatic_adaptation_CMCCAT2000`

See Also
--------
`CMCCAT2000 Chromatic Adaptation Model Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/adaptation/cmccat2000.ipynb>`_

References
----------
-   :cite:`Li2002a` : Li, C., Luo, M. R., Rigg, B., & Hunt, R. W. G. (2002).
    CMC 2000 chromatic adaptation transform: CMCCAT2000. Color Research &
    Application, 27(1), 49-58. doi:10.1002/col.10005
-   :cite:`Westland2012k` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    CMCCAT2000. In Computational Colour Science Using MATLAB
    (2nd ed., pp. 83-86). ISBN:978-0-470-66569-5
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.adaptation import CMCCAT2000_CAT
from colour.utilities import (CaseInsensitiveMapping, as_float_array,
                              dot_vector, from_range_100, to_domain_100)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'CMCCAT2000_INVERSE_CAT', 'CMCCAT2000_InductionFactors',
    'CMCCAT2000_VIEWING_CONDITIONS', 'chromatic_adaptation_forward_CMCCAT2000',
    'chromatic_adaptation_reverse_CMCCAT2000',
    'chromatic_adaptation_CMCCAT2000'
]

CMCCAT2000_INVERSE_CAT = np.linalg.inv(CMCCAT2000_CAT)
"""
Inverse *CMCCAT2000* chromatic adaptation transform.

CMCCAT2000_INVERSE_CAT : array_like, (3, 3)
"""


class CMCCAT2000_InductionFactors(
        namedtuple('CMCCAT2000_InductionFactors', ('F', ))):
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


CMCCAT2000_VIEWING_CONDITIONS = CaseInsensitiveMapping({
    'Average': CMCCAT2000_InductionFactors(1),
    'Dim': CMCCAT2000_InductionFactors(0.8),
    'Dark': CMCCAT2000_InductionFactors(0.8)
})
CMCCAT2000_VIEWING_CONDITIONS.__doc__ = """
Reference *CMCCAT2000* chromatic adaptation model viewing conditions.

References
----------
:cite:`Li2002a`, :cite:`Westland2012k`

CMCCAT2000_VIEWING_CONDITIONS : CaseInsensitiveMapping
    ('Average', 'Dim', 'Dark')
"""


def chromatic_adaptation_forward_CMCCAT2000(
        XYZ,
        XYZ_w,
        XYZ_wr,
        L_A1,
        L_A2,
        surround=CMCCAT2000_VIEWING_CONDITIONS['Average']):
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
    surround : CMCCAT2000_InductionFactors, optional
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

    RGB = dot_vector(CMCCAT2000_CAT, XYZ)
    RGB_w = dot_vector(CMCCAT2000_CAT, XYZ_w)
    RGB_wr = dot_vector(CMCCAT2000_CAT, XYZ_wr)

    D = (surround.F * (0.08 * np.log10(0.5 * (L_A1 + L_A2)) + 0.76 - 0.45 *
                       (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = np.clip(D, 0, 1)
    a = D * XYZ_w[..., 1] / XYZ_wr[..., 1]

    RGB_c = (
        RGB * (a[..., np.newaxis] * (RGB_wr / RGB_w) + 1 - D[..., np.newaxis]))
    XYZ_c = dot_vector(CMCCAT2000_INVERSE_CAT, RGB_c)

    return from_range_100(XYZ_c)


def chromatic_adaptation_reverse_CMCCAT2000(
        XYZ_c,
        XYZ_w,
        XYZ_wr,
        L_A1,
        L_A2,
        surround=CMCCAT2000_VIEWING_CONDITIONS['Average']):
    """
    Adapts given stimulus corresponding colour *CIE XYZ* tristimulus values
    from reference viewing conditions to test viewing conditions using
    *CMCCAT2000* reverse chromatic adaptation model.

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
    surround : CMCCAT2000_InductionFactors, optional
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
    >>> chromatic_adaptation_reverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1,
    ...                                         L_A2)
    ... # doctest: +ELLIPSIS
    array([ 22.4839876...,  22.7419485...,   8.5393392...])
    """

    XYZ_c = to_domain_100(XYZ_c)
    XYZ_w = to_domain_100(XYZ_w)
    XYZ_wr = to_domain_100(XYZ_wr)
    L_A1 = as_float_array(L_A1)
    L_A2 = as_float_array(L_A2)

    RGB_c = dot_vector(CMCCAT2000_CAT, XYZ_c)
    RGB_w = dot_vector(CMCCAT2000_CAT, XYZ_w)
    RGB_wr = dot_vector(CMCCAT2000_CAT, XYZ_wr)

    D = (surround.F * (0.08 * np.log10(0.5 * (L_A1 + L_A2)) + 0.76 - 0.45 *
                       (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = np.clip(D, 0, 1)
    a = D * XYZ_w[..., 1] / XYZ_wr[..., 1]

    RGB = (RGB_c / (a[..., np.newaxis] *
                    (RGB_wr / RGB_w) + 1 - D[..., np.newaxis]))
    XYZ = dot_vector(CMCCAT2000_INVERSE_CAT, RGB)

    return from_range_100(XYZ)


def chromatic_adaptation_CMCCAT2000(
        XYZ,
        XYZ_w,
        XYZ_wr,
        L_A1,
        L_A2,
        surround=CMCCAT2000_VIEWING_CONDITIONS['Average'],
        direction='Forward'):
    """
    Adapts given stimulus *CIE XYZ* tristimulus values using given viewing
    conditions.

    This definition is a convenient wrapper around
    :func:`colour.adaptation.chromatic_adaptation_forward_CMCCAT2000` and
    :func:`colour.adaptation.chromatic_adaptation_reverse_CMCCAT2000`.

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
    surround : CMCCAT2000_InductionFactors, optional
        Surround viewing conditions induction factors.
    direction : unicode, optional
        **{'Forward', 'Reverse'}**,
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

    Using the *CMCCAT2000* reverse model:

    >>> XYZ = np.array([19.52698326, 23.06833960, 24.97175229])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_CMCCAT2000(
    ...     XYZ, XYZ_w, XYZ_wr, L_A1, L_A2, direction='Reverse')
    ... # doctest: +ELLIPSIS
    array([ 22.48,  22.74,   8.54])
    """

    if direction.lower() == 'forward':
        return chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr,
                                                       L_A1, L_A2, surround)
    else:
        return chromatic_adaptation_reverse_CMCCAT2000(XYZ, XYZ_w, XYZ_wr,
                                                       L_A1, L_A2, surround)
