#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pivoted Log Encoding
====================

Defines the *Pivoted Log* encoding:

-   :func:`log_encoding_PivotedLog`
-   :func:`log_decoding_PivotedLog`

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

__all__ = ['log_encoding_PivotedLog',
           'log_decoding_PivotedLog']


def log_encoding_PivotedLog(x,
                            log_reference=445,
                            linear_reference=0.18,
                            negative_gamma=0.6,
                            density_per_code_value=0.002):
    """
    Defines the *Josh Pines* style *Pivoted Log* log encoding curve /
    opto-electronic transfer function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    log_reference : numeric or array_like
        Log reference.
    linear_reference : numeric or array_like
        Linear reference.
    negative_gamma : numeric or array_like
        Negative gamma.
    density_per_code_value : numeric or array_like
        Density per code value.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Examples
    --------
    >>> log_encoding_PivotedLog(0.18)  # doctest: +ELLIPSIS
    0.4349951...
    """

    x = np.asarray(x)

    return ((log_reference + np.log10(x / linear_reference) /
             (density_per_code_value / negative_gamma)) / 1023)


def log_decoding_PivotedLog(y,
                            log_reference=445,
                            linear_reference=0.18,
                            negative_gamma=0.6,
                            density_per_code_value=0.002):
    """
    Defines the *Josh Pines* style *Pivoted Log* log decoding curve /
    electro-optical transfer function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.
    log_reference : numeric or array_like
        Log reference.
    linear_reference : numeric or array_like
        Linear reference.
    negative_gamma : numeric or array_like
        Negative gamma.
    density_per_code_value : numeric or array_like
        Density per code value.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Examples
    --------
    >>> log_decoding_PivotedLog(0.434995112414467)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    return (10 ** ((y * 1023 - log_reference) *
                   (density_per_code_value / negative_gamma)) *
            linear_reference)
