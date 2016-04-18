#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Viper Log Encodings
===================

Defines the *Viper Log* encoding:

-   :def:`log_encoding_ViperLog`
-   :def:`log_decoding_ViperLog`

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

__all__ = ['log_encoding_ViperLog',
           'log_decoding_ViperLog']


def log_encoding_ViperLog(value, **kwargs):
    """
    Defines the *Viper Log* log encoding curve / opto-electronic conversion
    function.

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
    >>> log_encoding_ViperLog(0.18)  # doctest: +ELLIPSIS
    0.6360080...
    """

    value = np.asarray(value)

    return (1023 + 500 * np.log10(value)) / 1023


def log_decoding_ViperLog(value, **kwargs):
    """
    Defines the *Viper Log* log decoding curve / electro-optical conversion
    function.

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
    >>> log_decoding_ViperLog(0.63600806701041346)  # doctest: +ELLIPSIS
    0.1799999...
    """

    value = np.asarray(value)

    return 10 ** ((1023 * value - 1023) / 500)
