# -*- coding: utf-8 -*-
"""
Gamma Colour Component Transfer Function
========================================

Defines gamma encoding / decoding colour component transfer function related
objects:

- :func:`colour.gamma_function`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import spow
from colour.utilities import as_float_array, as_float

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['gamma_function']


def gamma_function(a, exponent=1, negative_number_handling='Indeterminate'):
    """
    Defines a typical gamma encoding / decoding function.

    Parameters
    ----------
    a : numeric or array_like
        Array to encode / decode.
    exponent : numeric or array_like, optional
        Encoding / decoding exponent.
    negative_number_handling : unicode, optional
        **{'Indeterminate', 'Mirror', 'Preserve', 'Clamp'}**,
        Defines the behaviour for ``a`` negative numbers and / or the
        definition return value:

        -   *Indeterminate*: The behaviour will be indeterminate and
            definition return value might contain *nans*.
        -   *Mirror*: The definition return value will be mirrored around
            abscissa and ordinate axis, i.e. Blackmagic Design: Davinci Resolve
            behaviour.
        -   *Preserve*: The definition will preserve any negative number in
            ``a``, i.e. The Foundry Nuke behaviour.
        -   *Clamp*: The definition will clamp any negative number in ``a`` to
            0.

    Returns
    -------
    numeric or ndarray
        Encoded / decoded array.

    Raises
    ------
    ValueError
        If the negative number handling method is not defined.

    Examples
    --------
    >>> gamma_function(0.18, 2.2)  # doctest: +ELLIPSIS
    0.0229932...
    >>> gamma_function(-0.18, 2.0)  # doctest: +ELLIPSIS
    0.0323999...
    >>> gamma_function(-0.18, 2.2)
    nan
    >>> gamma_function(-0.18, 2.2, 'Mirror')  # doctest: +ELLIPSIS
    -0.0229932...
    >>> gamma_function(-0.18, 2.2, 'Preserve')  # doctest: +ELLIPSIS
    -0.1...
    >>> gamma_function(-0.18, 2.2, 'Clamp')  # doctest: +ELLIPSIS
    0.0
    """

    a = as_float_array(a)
    exponent = as_float_array(exponent)

    negative_number_handling = negative_number_handling.lower()
    if negative_number_handling == 'indeterminate':
        return as_float(a ** exponent)
    elif negative_number_handling == 'mirror':
        return spow(a, exponent)
    elif negative_number_handling == 'preserve':
        return as_float(np.where(a <= 0, a, a ** exponent))
    elif negative_number_handling == 'clamp':
        return as_float(np.where(a <= 0, 0, a ** exponent))
    else:
        raise ValueError(
            'Undefined negative number handling method: "{0}".'.format(
                negative_number_handling))
