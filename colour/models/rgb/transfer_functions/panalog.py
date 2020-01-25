# -*- coding: utf-8 -*-
"""
Panalog Encoding
================

Defines the *Panalog* encoding:

-   :func:`colour.models.log_encoding_Panalog`
-   :func:`colour.models.log_decoding_Panalog`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`SonyImageworks2012a` : Sony Imageworks. (2012). make.py. Retrieved
    November 27, 2014, from https://github.com/imageworks/OpenColorIO-Configs/\
blob/master/nuke-default/make.py
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['log_encoding_Panalog', 'log_decoding_Panalog']


def log_encoding_Panalog(x, black_offset=10 ** ((64 - 681) / 444)):
    """
    Defines the *Panalog* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Warnings
    --------
    These are estimations known to be close enough, the actual log encoding
    curves are not published.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_encoding_Panalog(0.18)  # doctest: +ELLIPSIS
    0.3745767...
    """

    x = to_domain_1(x)

    y = (681 + 444 * np.log10(x * (1 - black_offset) + black_offset)) / 1023

    return from_range_1(y)


def log_decoding_Panalog(y, black_offset=10 ** ((64 - 681) / 444)):
    """
    Defines the *Panalog* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Warnings
    --------
    These are estimations known to be close enough, the actual log encoding
    curves are not published.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_decoding_Panalog(0.374576791382298)  # doctest: +ELLIPSIS
    0.1...
    """

    y = to_domain_1(y)

    x = (10 ** ((1023 * y - 681) / 444) - black_offset) / (1 - black_offset)

    return from_range_1(x)
