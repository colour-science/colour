#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear Colour Component Transfer Function
=========================================

Defines linear encoding / decoding colour component transfer function related
objects:

- :func:`function_linear`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['function_linear']


def function_linear(a):
    """
    Defines a typical linear encoding / decoding function, essentially a
    pass-through function.

    Parameters
    ----------
    a : numeric or array_like
        Array to encode / decode.

    Returns
    -------
    numeric or ndarray
        Encoded / decoded array.

    Examples
    --------
    >>> function_linear(0.18)
    0.18
    """

    return a
