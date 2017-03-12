#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kodak Cineon Encoding
=====================

Defines the *Kodak Cineon* encoding:

-   :func:`log_encoding_Cineon`
-   :func:`log_decoding_Cineon`

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

__all__ = ['log_encoding_Cineon',
           'log_decoding_Cineon']


def log_encoding_Cineon(x,
                        black_offset=10 ** ((95 - 685) / 300)):
    """
    Defines the *Cineon* log encoding curve / opto-electronic transfer
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

    Examples
    --------
    >>> log_encoding_Cineon(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    x = np.asarray(x)

    return ((685 + 300 *
             np.log10(x * (1 - black_offset) + black_offset)) / 1023)


def log_decoding_Cineon(y,
                        black_offset=10 ** ((95 - 685) / 300)):
    """
    Defines the *Cineon* log decoding curve / electro-optical transfer
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

    Examples
    --------
    >>> log_decoding_Cineon(0.457319613085418)  # doctest: +ELLIPSIS
    0.1799999...
    """

    y = np.asarray(y)

    return ((10 ** ((1023 * y - 685) / 300) - black_offset) /
            (1 - black_offset))
