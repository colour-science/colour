#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ITU-R BT.1886 EOTF / EOCF
=========================

Defines *Recommendation ITU-R BT.1886* EOTF / EOCF:

-   :func:`BT_1886_EOCF`

See Also
--------
`ITU-R BT.1886 EOTF / EOCF IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/bt_1886.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (2011). Recommendation ITU-R
        BT.1886 - Reference electro-optical transfer function for flat panel
        displays used in HDTV studio production BT Series Broadcasting service.
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['BT_1886_EOCF']


def BT_1886_EOCF(V, L_W=940, L_B=64):
    """
    Defines *Recommendation ITU-R BT.1886* Electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    V : numeric or array_like
        Input video signal level (normalized, black at :math:`V = 0`, to white
        at :math:`V = 1`. For content mastered per
        *Recommendation ITU-R BT.709*, 10-bit digital code values :math:`D` map
        into values of :math:`V` per the following equation:
        :math:`V = (D–64)/876`
    L_W : numeric, optional
        Screen luminance for white.
    L_B : numeric, optional
        Screen luminance for black.

    Returns
    -------
    numeric or ndarray
        Screen luminance in :math:`cd/m^2`.

    Examples
    --------
    >>> BT_1886_EOCF(0.5)  # doctest: +ELLIPSIS
    350.8224951...
    """

    V = np.asarray(V)

    gamma = 2.40
    gamma_d = 1 / gamma

    n = L_W ** gamma_d - L_B ** gamma_d
    a = n ** gamma
    b = L_B ** gamma_d / n
    L = a * np.maximum(V + b, 0) ** gamma

    return L
