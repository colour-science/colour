#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DCI-P3 EOTF / EOCF and OETF / EOCF
==================================

Defines the *DCI-P3* log encoding:

-   :def:`log_encoding_DCIP3`
-   :def:`log_decoding_DCIP3`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Digital Cinema Initiatives. (2007). Digital Cinema System
        Specification - Version 1.1. Retrieved from
        http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_DCIP3',
           'log_decoding_DCIP3']


def log_encoding_DCIP3(value, **kwargs):
    """
    Defines the *DCI-P3* colourspace opto-electronic conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* encoding curves.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> log_encoding_DCIP3(0.18)  # doctest: +ELLIPSIS
    461.9922059...
    """

    value = np.asarray(value)

    return 4095 * (value / 52.37) ** (1 / 2.6)


def log_decoding_DCIP3(value, **kwargs):
    """
    Defines the *DCI-P3* colourspace electro-optical conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* decoding curves.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_DCIP3(461.99220597484737)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    return 52.37 * (value / 4095) ** 2.6
