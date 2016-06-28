#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RED Log Encodings
=================

Defines the *RED* log encodings:

-   :func:`log_encoding_REDLog`
-   :func:`log_decoding_REDLog`

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

from colour.models.rgb.transfer_functions import (
    log_encoding_Cineon,
    log_decoding_Cineon)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_REDLog',
           'log_decoding_REDLog',
           'log_encoding_REDLogFilm',
           'log_decoding_REDLogFilm']


def log_encoding_REDLog(value,
                        black_offset=10 ** ((0 - 1023) / 511)):
    """
    Defines the *REDLog* log encoding curve / opto-electronic transfer
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
    >>> log_encoding_REDLog(0.18)  # doctest: +ELLIPSIS
    0.6376218...
    """

    value = np.asarray(value)

    return ((1023 +
             511 * np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def log_decoding_REDLog(value,
                        black_offset=10 ** ((0 - 1023) / 511)):
    """
    Defines the *REDLog* log decoding curve / electro-optical transfer
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
    >>> log_decoding_REDLog(0.63762184598817484)  # doctest: +ELLIPSIS
    0.1...
    """

    value = np.asarray(value)

    return (((10 **
              ((1023 * value - 1023) / 511)) - black_offset) /
            (1 - black_offset))


def log_encoding_REDLogFilm(value,
                            black_offset=10 ** ((95 - 685) / 300)):
    """
    Defines the *REDLogFilm* log encoding curve / opto-electronic transfer
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
    >>> log_encoding_REDLogFilm(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    return log_encoding_Cineon(value, black_offset)


def log_decoding_REDLogFilm(value,
                            black_offset=10 ** ((95 - 685) / 300)):
    """
    Defines the *REDLogFilm* log decoding curve / electro-optical transfer
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
    >>> log_decoding_REDLogFilm(0.45731961308541841)  # doctest: +ELLIPSIS
    0.18...
    """

    return log_decoding_Cineon(value, black_offset)
