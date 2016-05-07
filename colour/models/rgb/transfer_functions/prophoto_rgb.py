#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProPhoto RGB OETF (OECF) and EOTF (EOCF)
========================================

Defines the *ProPhoto RGB* OETF (OECF) and EOTF (EOCF):

-   :func:`oetf_ProPhotoRGB`
-   :func:`eotf_ProPhotoRGB`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Spaulding, K. E., Woolfe, G. J., & Giorgianni, E. J. (2000). Reference
        Input/Output Medium Metric RGB Color Encodings (RIMM/ROMM RGB), 1–8.
        Retrieved from http://www.photo-lovers.org/pdf/color/romm.pdf
.. [3]  ANSI. (2003). Specification of ROMM RGB. Retrieved from
        http://www.color.org/ROMMRGB.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oetf_ProPhotoRGB',
           'eotf_ProPhotoRGB']


def oetf_ProPhotoRGB(value):
    """
    Defines the *ProPhoto RGB* colourspace opto-electronic transfer function.

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
    >>> oetf_ProPhotoRGB(0.18)  # doctest: +ELLIPSIS
    0.3857114...
    """

    value = np.asarray(value)

    return as_numeric(np.where(value < 0.001953,
                               value * 16,
                               value ** (1 / 1.8)))


def eotf_ProPhotoRGB(value):
    """
    Defines the *ProPhoto RGB* colourspace electro-optical transfer function.

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
    >>> eotf_ProPhotoRGB(0.3857114247511376)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    return as_numeric(np.where(
        value < oetf_ProPhotoRGB(0.001953),
        value / 16,
        value ** 1.8))
