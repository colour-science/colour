#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ITU-R BT.1886
=============

Defines *Recommendation ITU-R BT.1886* opto-electrical transfer function
(OETF / OECF) and electro-optical transfer function (EOTF / EOCF):

-   :func:`oetf_BT1886`
-   :func:`eotf_BT1886`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (2011). Recommendation ITU-R
        BT.1886 - Reference electro-optical transfer function for flat panel
        displays used in HDTV studio production BT Series Broadcasting service.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oetf_BT1886',
           'eotf_BT1886']


def oetf_BT1886(L, L_B=64, L_W=940):
    """
    Defines *Recommendation ITU-R BT.1886* opto-electrical transfer function
    (OETF / OECF).

    Parameters
    ----------
    L : numeric or array_like
        Screen luminance in :math:`cd/m^2`.
    L_B : numeric, optional
        Screen luminance for black.
    L_W : numeric, optional
        Screen luminance for white.

    Returns
    -------
    numeric or ndarray
        Input video signal level (normalized, black at :math:`V = 0`, to white
        at :math:`V = 1`.

    Warning
    -------
    *Recommendation ITU-R BT.1886* doesn't specify an opto-electrical
    transfer function. This definition is used for symmetry in unit tests and
    other computations but should not be used as an *OETF*.

    Examples
    --------
    >>> oetf_BT1886(277.98159179331145)  # doctest: +ELLIPSIS
    0.4090077...
    """

    warning(('*Recommendation ITU-R BT.1886* doesn\'t specify an '
             'opto-electrical transfer function. This definition is used '
             'for symmetry in unit tests and others computations but should '
             'not be used as an *OETF*!'))

    L = np.asarray(L)

    gamma = 2.40
    gamma_d = 1 / gamma

    n = L_W ** gamma_d - L_B ** gamma_d
    a = n ** gamma
    b = L_B ** gamma_d / n

    V = (L / a) ** gamma_d - b

    return V


def eotf_BT1886(V, L_B=64, L_W=940):
    """
    Defines *Recommendation ITU-R BT.1886* electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    V : numeric or array_like
        Input video signal level (normalized, black at :math:`V = 0`, to white
        at :math:`V = 1`. For content mastered per
        *Recommendation ITU-R BT.709*, 10-bit digital code values :math:`D` map
        into values of :math:`V` per the following equation:
        :math:`V = (D–64)/876`
    L_B : numeric, optional
        Screen luminance for black.
    L_W : numeric, optional
        Screen luminance for white.

    Returns
    -------
    numeric or ndarray
        Screen luminance in :math:`cd/m^2`.

    Examples
    --------
    >>> eotf_BT1886(0.409007728864150)  # doctest: +ELLIPSIS
    277.9815917...
    """

    V = np.asarray(V)

    gamma = 2.40
    gamma_d = 1 / gamma

    n = L_W ** gamma_d - L_B ** gamma_d
    a = n ** gamma
    b = L_B ** gamma_d / n
    L = a * np.maximum(V + b, 0) ** gamma

    return L
