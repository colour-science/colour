#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kodak Cineon Encoding
=====================

Defines the *Kodak Cineon* encoding:

-   :def:`log_encoding_Cineon`
-   :def:`log_decoding_Cineon`

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

__all__ = ['log_encoding_Cineon',
           'log_decoding_Cineon']


def log_encoding_Cineon(value,
                        black_offset=10 ** ((95 - 685) / 300)):
    """
    Defines the *Cineon* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> log_encoding_Cineon(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    value = np.asarray(value)

    return ((685 + 300 *
             np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def log_decoding_Cineon(value,
                        black_offset=10 ** ((95 - 685) / 300)):
    """
    Defines the *Cineon* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_Cineon(0.45731961308541841)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    return ((10 ** ((1023 * value - 685) / 300) - black_offset) /
            (1 - black_offset))
