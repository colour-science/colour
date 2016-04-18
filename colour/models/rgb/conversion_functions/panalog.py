#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Panalog Encoding
================

Defines the *Panalog* encoding:

-   :def:`log_encoding_Panalog`
-   :def:`log_decoding_Panalog`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
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
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_Panalog',
           'log_decoding_Panalog']


def log_encoding_Panalog(value,
                         black_offset=10 ** ((64 - 681) / 444),
                         **kwargs):
    """
    Defines the *Panalog* log encoding curve / opto-electronic conversion
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    black_offset : numeric or array_like
        Black offset.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* encoding curves.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> log_encoding_Panalog(0.18)  # doctest: +ELLIPSIS
    0.3745767...
    """

    value = np.asarray(value)

    return ((681 + 444 *
             np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def log_decoding_Panalog(value,
                         black_offset=10 ** ((64 - 681) / 444),
                         **kwargs):
    """
    Defines the *Panalog* log decoding curve / electro-optical conversion
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    black_offset : numeric or array_like
        Black offset.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* decoding curves.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_Panalog(0.37457679138229816)  # doctest: +ELLIPSIS
    0.1...
    """

    value = np.asarray(value)

    return ((10 ** ((1023 * value - 681) / 444) - black_offset) /
            (1 - black_offset))
