#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
V-Log Log Encoding
==================

Defines the *V-Log* log encoding:

-   :func:`log_encoding_VLog`
-   :func:`log_decoding_VLog`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Panasonic. (2014). VARICAM V-Log/V-Gamut. Retrieved from
        http://pro-av.panasonic.net/en/varicam/common/pdf/\
VARICAM_V-Log_V-Gamut.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import Structure, as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['V_LOG_CONSTANTS',
           'log_encoding_VLog',
           'log_decoding_VLog']

V_LOG_CONSTANTS = Structure(cut1=0.01,
                            cut2=0.181,
                            b=0.00873,
                            c=0.241514,
                            d=0.598206)
"""
*V-Log* colourspace constants.

V_LOG_CONSTANTS : Structure
"""


def log_encoding_VLog(value):
    """
    Defines the *V-Log* log encoding curve / opto-electronic transfer
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
    >>> log_encoding_VLog(0.18)  # doctest: +ELLIPSIS
    0.4233114...
    """

    value = np.asarray(value)

    cut1 = V_LOG_CONSTANTS.cut1
    b = V_LOG_CONSTANTS.b
    c = V_LOG_CONSTANTS.c
    d = V_LOG_CONSTANTS.d

    value = np.where(value < cut1,
                     5.6 * value + 0.125,
                     c * np.log10(value + b) + d)

    return as_numeric(value)


def log_decoding_VLog(value):
    """
    Defines the *V-Log* log decoding curve / electro-optical transfer
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
    >>> log_decoding_VLog(0.42331144876013616)  # doctest: +ELLIPSIS
    0.1799999...
    """

    value = np.asarray(value)

    cut2 = V_LOG_CONSTANTS.cut2
    b = V_LOG_CONSTANTS.b
    c = V_LOG_CONSTANTS.c
    d = V_LOG_CONSTANTS.d

    value = np.where(value < cut2,
                     (value - 0.125) / 5.6,
                     np.power(10, ((value - d) / c)) - b)

    return as_numeric(value)
