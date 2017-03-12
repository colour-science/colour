#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Panalog Encoding
================

Defines the *Panalog* encoding:

-   :func:`log_encoding_Panalog`
-   :func:`log_decoding_Panalog`

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

__all__ = ['log_encoding_Panalog',
           'log_decoding_Panalog']


def log_encoding_Panalog(x,
                         black_offset=10 ** ((64 - 681) / 444)):
    """
    Defines the *Panalog* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Warnings
    --------
    These are estimations known to be close enough, the actual log encoding
    curves are not published.

    Examples
    --------
    >>> log_encoding_Panalog(0.18)  # doctest: +ELLIPSIS
    0.3745767...
    """

    x = np.asarray(x)

    return ((681 + 444 *
             np.log10(x * (1 - black_offset) + black_offset)) / 1023)


def log_decoding_Panalog(y,
                         black_offset=10 ** ((64 - 681) / 444)):
    """
    Defines the *Panalog* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Warnings
    --------
    These are estimations known to be close enough, the actual log encoding
    curves are not published.

    Examples
    --------
    >>> log_decoding_Panalog(0.374576791382298)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    return ((10 ** ((1023 * y - 681) / 444) - black_offset) /
            (1 - black_offset))
