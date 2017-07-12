#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GoPro Encodings
===============

Defines the *GoPro* encodings:

-   :func:`log_encoding_Protune`
-   :func:`log_decoding_Protune`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  GoPro, Duiker, H.-P., & Mansencal, T. (2016). gopro.py. Retrieved
        April 12, 2017, from https://github.com/hpd/OpenColorIO-Configs/blob/\
master/aces_1.0.3/python/aces_ocio/colorspaces/gopro.py
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_Protune', 'log_decoding_Protune']


def log_encoding_Protune(x):
    """
    Defines the *Protune* log encoding curve / opto-electronic transfer
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
    >>> log_encoding_Protune(0.18)  # doctest: +ELLIPSIS
    0.6456234...
    """

    x = np.asarray(x)

    return np.log(x * 112 + 1) / np.log(113)


def log_decoding_Protune(y):
    """
    Defines the *Protune* log decoding curve / electro-optical transfer
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
    >>> log_decoding_Protune(0.645623486803636)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    return (113 ** y - 1) / 112
