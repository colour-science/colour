#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Panasonic V-Log Log Encoding
============================

Defines the *Panasonic V-Log* log encoding:

-   :func:`log_encoding_VLog`
-   :func:`log_decoding_VLog`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
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
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['VLOG_CONSTANTS',
           'log_encoding_VLog',
           'log_decoding_VLog']

VLOG_CONSTANTS = Structure(cut1=0.01,
                           cut2=0.181,
                           b=0.00873,
                           c=0.241514,
                           d=0.598206)
"""
*Panasonic V-Log* colourspace constants.

VLOG_CONSTANTS : Structure
"""


def log_encoding_VLog(L_in):
    """
    Defines the *Panasonic V-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    L_in : numeric or array_like
        Linear reflection data :math`L_{in}`.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`V_{out}`.

    Examples
    --------
    >>> log_encoding_VLog(0.18)  # doctest: +ELLIPSIS
    0.4233114...
    """

    L_in = np.asarray(L_in)

    cut1 = VLOG_CONSTANTS.cut1
    b = VLOG_CONSTANTS.b
    c = VLOG_CONSTANTS.c
    d = VLOG_CONSTANTS.d

    L_in = np.where(L_in < cut1,
                    5.6 * L_in + 0.125,
                    c * np.log10(L_in + b) + d)

    return as_numeric(L_in)


def log_decoding_VLog(V_out):
    """
    Defines the *Panasonic V-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    V_out : numeric or array_like
        Non-linear data :math:`V_{out}`.

    Returns
    -------
    numeric or ndarray
        Linear reflection data :math`L_{in}`.

    Examples
    --------
    >>> log_decoding_VLog(0.423311448760136)  # doctest: +ELLIPSIS
    0.1799999...
    """

    V_out = np.asarray(V_out)

    cut2 = VLOG_CONSTANTS.cut2
    b = VLOG_CONSTANTS.b
    c = VLOG_CONSTANTS.c
    d = VLOG_CONSTANTS.d

    V_out = np.where(V_out < cut2,
                     (V_out - 0.125) / 5.6,
                     np.power(10, ((V_out - d) / c)) - b)

    return as_numeric(V_out)
