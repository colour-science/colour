#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gamma Colour Component Transfer Function
========================================

Defines gamma encoding / decoding colour component transfer function related
objects:

- :func:`gamma_function`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['gamma_function']


def gamma_function(a, exponent=1.0):
    """
    Defines a typical gamma encoding / decoding function.

    Parameters
    ----------
    a : numeric or array_like
        Array to encode / decode.
    exponent : numeric, optional
        Encoding / decoding exponent.

    Returns
    -------
    numeric or ndarray
        Encoded / decoded array.

    Examples
    --------
    >>> gamma_function(0.18, 2.2)  # doctest: +ELLIPSIS
    0.0229932...
    """

    a = np.asarray(a)

    return a ** exponent
