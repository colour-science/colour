# -*- coding: utf-8 -*-
"""
Pivoted Log Encoding
====================

Defines the *Pivoted Log* encoding:

-   :func:`colour.models.log_encoding_PivotedLog`
-   :func:`colour.models.log_decoding_PivotedLog`

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

__all__ = ['log_encoding_PivotedLog', 'log_decoding_PivotedLog']


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
    >>> log_encoding_PivotedLog(0.18)  # doctest: +ELLIPSIS
    0.4349951...
    """

    x = to_domain_1(x)

    y = ((log_reference + np.log10(x / linear_reference) /
          (density_per_code_value / negative_gamma)) / 1023)

    return from_range_1(y)


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
    >>> log_decoding_PivotedLog(0.434995112414467)  # doctest: +ELLIPSIS
    0.1...
    """

    y = to_domain_1(y)

    x = (10 ** ((y * 1023 - log_reference) *
                (density_per_code_value / negative_gamma)) * linear_reference)

    return from_range_1(x)
