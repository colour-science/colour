#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Viper Log Encodings
===================

Defines the *Viper Log* encoding:

-   :func:`log_encoding_ViperLog`
-   :func:`log_decoding_ViperLog`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Sony Imageworks. (2012). make.py. Retrieved November 27, 2014, from
        https://github.com/imageworks/OpenColorIO-Configs/\
blob/master/nuke-default/make.py
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_ViperLog',
           'log_decoding_ViperLog']


def log_encoding_ViperLog(x):
    """
    Defines the *Viper Log* log encoding curve / opto-electronic transfer
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
    >>> log_encoding_ViperLog(0.18)  # doctest: +ELLIPSIS
    0.6360080...
    """

    x = np.asarray(x)

    return (1023 + 500 * np.log10(x)) / 1023


def log_decoding_ViperLog(y):
    """
    Defines the *Viper Log* log decoding curve / electro-optical transfer
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
    >>> log_decoding_ViperLog(0.636008067010413)  # doctest: +ELLIPSIS
    0.1799999...
    """

    y = np.asarray(y)

    return 10 ** ((1023 * y - 1023) / 500)
