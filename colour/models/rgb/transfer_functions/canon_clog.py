#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canon C-Log Encoding
====================

Defines the *Canon C-Log* encoding:

-   :func:`log_encoding_CLog`
-   :func:`log_decoding_CLog`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
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
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_CLog',
           'log_decoding_CLog']


def log_encoding_CLog(value):
    """
    Defines the *Canon C-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> log_encoding_CLog(0.20) * 100  # doctest: +ELLIPSIS
    32.7953896...
    """

    value = np.asarray(value)

    return 0.529136 * np.log10(10.1596 * value + 1) + 0.0730597


def log_decoding_CLog(value):
    """
    Defines the *Canon C-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_CLog(32.795389693580908 / 100)  # doctest: +ELLIPSIS
    0.19999999...
    """

    value = np.asarray(value)

    return (-0.071622555735168 *
            (1.3742747797867 - np.exp(1) ** (4.3515940948906 * value)))
