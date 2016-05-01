#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProPhoto RGB EOTF / EOCF and OETF / EOCF
========================================

Defines the *ProPhoto RGB* log encoding:

-   :def:`log_encoding_ProPhotoRGB`
-   :def:`log_decoding_ProPhotoRGB`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  ANSI. (2003). Specification of ROMM RGB. Retrieved from
        http://www.color.org/ROMMRGB.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_ProPhotoRGB',
           'log_decoding_ProPhotoRGB']


def log_encoding_ProPhotoRGB(value):
    """
    Defines the *ProPhoto RGB* colourspace opto-electronic transfer function.

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
    >>> log_encoding_ProPhotoRGB(0.18)  # doctest: +ELLIPSIS
    0.3857114...
    """

    value = np.asarray(value)

    return as_numeric(np.where(value < 0.001953,
                               value * 16,
                               value ** (1 / 1.8)))


def log_decoding_ProPhotoRGB(value):
    """
    Defines the *ProPhoto RGB* colourspace electro-optical transfer function.

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
    >>> log_decoding_ProPhotoRGB(0.3857114247511376)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    return as_numeric(np.where(
        value < log_encoding_ProPhotoRGB(0.001953),
        value / 16,
        value ** 1.8))
