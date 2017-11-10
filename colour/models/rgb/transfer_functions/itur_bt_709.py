#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ITU-R BT.709-6
==============

Defines *ITU-R BT.709-6* opto-electrical transfer function (OETF / OECF) and
its reverse:

-   :func:`oetf_BT709`
-   :func:`oetf_reverse_BT709`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (2015). Recommendation
        ITU-R BT.709-6 - Parameter values for the HDTV standards for production
        and international programme exchange BT Series Broadcasting service.
        Retrieved from https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oetf_BT709', 'oetf_reverse_BT709']


def oetf_BT709(L):
    """
    Defines *Recommendation ITU-R BT.709-6* opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    L : numeric or array_like
        *Luminance* :math:`L` of the image.

    Returns
    -------
    numeric or ndarray
        Corresponding electrical signal :math:`V`.

    Examples
    --------
    >>> oetf_BT709(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    """

    L = np.asarray(L)

    return as_numeric(
        np.where(L < 0.018, L * 4.5, 1.099 * (L ** 0.45) - 0.099))


def oetf_reverse_BT709(V):
    """
    Defines *Recommendation ITU-R BT.709-6* reverse opto-electronic transfer
    function (OETF / OECF).

    Parameters
    ----------
    V : numeric or array_like
        Electrical signal :math:`V`.

    Returns
    -------
    numeric or ndarray
        Corresponding *luminance* :math:`L` of the image.

    Examples
    --------
    >>> oetf_reverse_BT709(0.409007728864150)  # doctest: +ELLIPSIS
    0.1...
    """

    V = np.asarray(V)

    return as_numeric(
        np.where(V < oetf_BT709(0.018), V / 4.5, ((V + 0.099) / 1.099) ** (
            1 / 0.45)))
