#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canon Log Encoding
==================

Defines the *Canon Log* encoding:

-   :func:`log_encoding_CanonLog`
-   :func:`log_decoding_CanonLog`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Thorpe, L. (2012). CANON-LOG TRANSFER CHARACTERISTIC. Retrieved
        from http://downloads.canon.com/CDLC/\
Canon-Log_Transfer_Characteristic_6-20-2012.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_CanonLog',
           'log_decoding_CanonLog']


def log_encoding_CanonLog(x):
    """
    Defines the *Canon Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Examples
    --------
    >>> log_encoding_CanonLog(0.20) * 100  # doctest: +ELLIPSIS
    32.7953896...
    """

    x = np.asarray(x)

    return 0.529136 * np.log10(10.1596 * x + 1) + 0.0730597


def log_decoding_CanonLog(y):
    """
    Defines the *Canon Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Examples
    --------
    >>> log_decoding_CanonLog(32.795389693580908 / 100)  # doctest: +ELLIPSIS
    0.19999999...
    """

    y = np.asarray(y)

    return (10 ** ((y - 0.0730597) / 0.529136) - 1) / 10.1596
