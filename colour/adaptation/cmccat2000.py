#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CMCCAT2000 Chromatic Adaptation Model
=====================================

Defines *CMCCAT2000* chromatic adaptation model objects:

-   :class:`CMCCAT2000_InductionFactors`
-   :class:`CMCCAT2000_VIEWING_CONDITIONS`
-   :func:`chromatic_adaptation_forward_CMCCAT2000`
-   :func:`chromatic_adaptation_reverse_CMCCAT2000`
-   :func:`chromatic_adaptation_CMCCAT2000`

See Also
--------
`CMCCAT2000 Chromatic Adaptation Model Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/adaptation/cmccat2000.ipynb>`_

References
----------
.. [1]  Li, C., Luo, M. R., Rigg, B., & Hunt, R. W. G. (2002). CMC 2000
        chromatic adaptation transform: CMCCAT2000. Color Research & …, 27(1),
        49–58. doi:10.1002/col.10005
.. [2]  Westland, S., Ripamonti, C., & Cheung, V. (2012). CMCCAT2000. In
        Computational Colour Science Using MATLAB (2nd ed., pp. 83–86).
        ISBN:978-0-470-66569-5
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.adaptation import CMCCAT2000_CAT
from colour.utilities import CaseInsensitiveMapping, dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CMCCAT2000_INVERSE_CAT',
           'CMCCAT2000_InductionFactors',
           'CMCCAT2000_VIEWING_CONDITIONS',
           'chromatic_adaptation_forward_CMCCAT2000',
           'chromatic_adaptation_reverse_CMCCAT2000',
           'chromatic_adaptation_CMCCAT2000']

CMCCAT2000_INVERSE_CAT = np.linalg.inv(CMCCAT2000_CAT)
"""
Inverse *CMCCAT2000* chromatic adaptation transform.

CMCCAT2000_INVERSE_CAT : array_like, (3, 3)
"""


class CMCCAT2000_InductionFactors(
    namedtuple('CMCCAT2000_InductionFactors',
               ('F',))):
    """
    *CMCCAT2000* chromatic adaptation model induction factors.

    Parameters
    ----------
    F : numeric or array_like
        :math:`F` surround condition.
    """


CMCCAT2000_VIEWING_CONDITIONS = CaseInsensitiveMapping(
    {'Average': CMCCAT2000_InductionFactors(1.),
     'Dim': CMCCAT2000_InductionFactors(0.8),
     'Dark': CMCCAT2000_InductionFactors(0.8)})
"""
Reference *CMCCAT2000* chromatic adaptation model viewing conditions.

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

    Warning
    -------
    The input domain and output range of that definition are non standard!

    Notes
    -----
    -   Input *CIE XYZ*, *CIE XYZ_w* and *CIE XYZ_wr* tristimulus values are in
        domain [0, 100].
    -   Output *CIE XYZ_c* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> XYZ = np.array([22.48, 22.74, 8.54])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_forward_CMCCAT2000(  # doctest: +ELLIPSIS
    ...     XYZ, XYZ_w, XYZ_wr, L_A1, L_A2)
    array([ 19.5269832...,  23.0683396...,  24.9717522...])
    """

    XYZ = np.asarray(XYZ)
    XYZ_w = np.asarray(XYZ_w)
    XYZ_wr = np.asarray(XYZ_wr)
    L_A1 = np.asarray(L_A1)
    L_A2 = np.asarray(L_A2)

    RGB = dot_vector(CMCCAT2000_CAT, XYZ)
    RGB_w = dot_vector(CMCCAT2000_CAT, XYZ_w)
    RGB_wr = dot_vector(CMCCAT2000_CAT, XYZ_wr)

    D = (surround.F *
         (0.08 * np.log10(0.5 * (L_A1 + L_A2)) +
          0.76 - 0.45 * (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = np.clip(D, 0, 1)
    a = D * XYZ_w[..., 1] / XYZ_wr[..., 1]

    RGB_c = (RGB *
             (a[..., np.newaxis] * (RGB_wr / RGB_w) + 1 - D[..., np.newaxis]))
    XYZ_c = dot_vector(CMCCAT2000_INVERSE_CAT, RGB_c)

    return XYZ_c


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

    Warning
    -------
    The input domain and output range of that definition are non standard!

    Notes
    -----
    -   Input *CIE XYZ_c*, *CIE XYZ_w* and *CIE XYZ_wr* tristimulus values
        are in domain [0, 100].
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> XYZ_c = np.array([19.53, 23.07, 24.97])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_reverse_CMCCAT2000(  # doctest: +ELLIPSIS
    ...     XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2)
    array([ 22.4839876...,  22.7419485...,   8.5393392...])
    """

    XYZ_c = np.asarray(XYZ_c)
    XYZ_w = np.asarray(XYZ_w)
    XYZ_wr = np.asarray(XYZ_wr)
    L_A1 = np.asarray(L_A1)
    L_A2 = np.asarray(L_A2)

    RGB_c = dot_vector(CMCCAT2000_CAT, XYZ_c)
    RGB_w = dot_vector(CMCCAT2000_CAT, XYZ_w)
    RGB_wr = dot_vector(CMCCAT2000_CAT, XYZ_wr)

    D = (surround.F *
         (0.08 * np.log10(0.5 * (L_A1 + L_A2)) +
          0.76 - 0.45 * (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = np.clip(D, 0, 1)
    a = D * XYZ_w[..., 1] / XYZ_wr[..., 1]

    RGB = (RGB_c /
           (a[..., np.newaxis] * (RGB_wr / RGB_w) + 1 - D[..., np.newaxis]))
    XYZ = dot_vector(CMCCAT2000_INVERSE_CAT, RGB)

    return XYZ


def chromatic_adaptation_CMCCAT2000(
        XYZ,
        XYZ_w,
        XYZ_wr,
        L_A1,
        L_A2,
        surround=CMCCAT2000_VIEWING_CONDITIONS['Average'],
        method='Forward'):
    """
    Adapts given stimulus *CIE XYZ* tristimulus values using given viewing
    conditions.

    This definition is a convenient wrapper around
    :func:`chromatic_adaptation_forward_CMCCAT2000` and
    :func:`chromatic_adaptation_reverse_CMCCAT2000`.

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
    method : unicode, optional
        **{'Forward', 'Reverse'}**,
        Chromatic adaptation method.

    Returns
    -------
    ndarray
        Adapted stimulus *CIE XYZ* tristimulus values.

    Warning
    -------
    The input domain and output range of that definition are non standard!

    Notes
    -----
    -   Input *CIE XYZ*, *CIE XYZ_w* and *CIE XYZ_wr* tristimulus values are in
        domain [0, 100].
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> XYZ = np.array([22.48, 22.74, 8.54])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_CMCCAT2000(  # doctest: +ELLIPSIS
    ...     XYZ, XYZ_w, XYZ_wr, L_A1, L_A2, method='Forward')
    array([ 19.5269832...,  23.0683396...,  24.9717522...])

    Using the *CMCCAT2000* reverse model:

    >>> XYZ = np.array([19.52698326, 23.06833960, 24.97175229])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> L_A1 = 200
    >>> L_A2 = 200
    >>> chromatic_adaptation_CMCCAT2000(  # doctest: +ELLIPSIS
    ...     XYZ, XYZ_w, XYZ_wr, L_A1, L_A2, method='Reverse')
    array([ 22.48,  22.74,   8.54])
    """

    if method.lower() == 'forward':
        return chromatic_adaptation_forward_CMCCAT2000(
            XYZ, XYZ_w, XYZ_wr, L_A1, L_A2, surround)
    else:
        return chromatic_adaptation_reverse_CMCCAT2000(
            XYZ, XYZ_w, XYZ_wr, L_A1, L_A2, surround)
