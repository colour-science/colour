# -*- coding: utf-8 -*-
"""
Von Kries Chromatic Adaptation Model
====================================

Defines *Von Kries* chromatic adaptation model objects:

-   :func:`colour.adaptation.chromatic_adaptation_matrix_VonKries`
-   :func:`colour.adaptation.chromatic_adaptation_VonKries`

See Also
--------
`Chromatic Adaptation Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/adaptation/vonkries.ipynb>`_

References
----------
-   :cite:`Fairchild2013t` : Fairchild, M. D. (2013). Chromatic Adaptation
    Models. In Color Appearance Models (3rd ed., pp. 4179-4252). Wiley.
    ISBN:B00DAYO8E2
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.adaptation import CHROMATIC_ADAPTATION_TRANSFORMS
from colour.utilities import (dot_matrix, dot_vector, from_range_1,
                              row_as_diagonal, to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'chromatic_adaptation_matrix_VonKries', 'chromatic_adaptation_VonKries'
]


def chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr, transform='CAT02'):
    """
    Computes the *chromatic adaptation* matrix from test viewing conditions
    to reference viewing conditions.

    Parameters
    ----------
    XYZ_w : array_like
        Test viewing condition *CIE XYZ* tristimulus values of whitepoint.
    XYZ_wr : array_like
        Reference viewing condition *CIE XYZ* tristimulus values of whitepoint.
    transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        Chromatic adaptation transform.

    Returns
    -------
    ndarray
        Chromatic adaptation matrix :math:`M_{cat}`.

    Raises
    ------
    KeyError
        If chromatic adaptation method is not defined.

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
    >>> chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr)
    ... # doctest: +ELLIPSIS
    array([[ 1.0425738...,  0.0308910..., -0.0528125...],
           [ 0.0221934...,  1.0018566..., -0.0210737...],
           [-0.0011648..., -0.0034205...,  0.7617890...]])

    Using Bradford method:

    >>> XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> method = 'Bradford'
    >>> chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr, method)
    ... # doctest: +ELLIPSIS
    array([[ 1.0479297...,  0.0229468..., -0.0501922...],
           [ 0.0296278...,  0.9904344..., -0.0170738...],
           [-0.0092430...,  0.0150551...,  0.7518742...]])
    """

    XYZ_w = to_domain_1(XYZ_w)
    XYZ_wr = to_domain_1(XYZ_wr)

    M = CHROMATIC_ADAPTATION_TRANSFORMS.get(transform)

    if M is None:
        raise KeyError(
            '"{0}" chromatic adaptation transform is not defined! Supported '
            'methods: "{1}".'.format(transform,
                                     CHROMATIC_ADAPTATION_TRANSFORMS.keys()))

    rgb_w = np.einsum('...i,...ij->...j', XYZ_w, np.transpose(M))
    rgb_wr = np.einsum('...i,...ij->...j', XYZ_wr, np.transpose(M))

    D = rgb_wr / rgb_w

    D = row_as_diagonal(D)

    M_CAT = dot_matrix(np.linalg.inv(M), D)
    M_CAT = dot_matrix(M_CAT, M)

    return M_CAT


def chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr, transform='CAT02'):
    """
    Adapts given stimulus from test viewing conditions to reference viewing
    conditions.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of stimulus to adapt.
    XYZ_w : array_like
        Test viewing condition *CIE XYZ* tristimulus values of whitepoint.
    XYZ_wr : array_like
        Reference viewing condition *CIE XYZ* tristimulus values of whitepoint.
    transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        Chromatic adaptation transform.

    Returns
    -------
    ndarray
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

    M_CAT = chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr, transform)
    XYZ_a = dot_vector(M_CAT, XYZ)

    return from_range_1(XYZ_a)
