#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ITU-R BT.709-6 EOTF / EOCF and OETF / EOCF
==========================================

Defines *ITU-R BT.709-6* EOTF / EOCF and OETF / EOCF:

-   :func:`oecf_BT709`
-   :func:`eocf_BT709`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (2002). Parameter values for
        the HDTV standards for production and international programme exchange
        BT Series Broadcasting service. In Recommendation ITU-R BT.709-6
        (Vol. 5, pp. 1–32). ISBN:9519982000
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric, warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oecf_BT709',
           'eocf_BT709']


def oecf_BT709(value):
    """
    Defines *Recommendation ITU-R BT.709-6* opto-electronic conversion
    function.

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
    >>> oecf_BT709(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    """

    value = np.asarray(value)

    return as_numeric(np.where(value < 0.018,
                               value * 4.5,
                               1.099 * (value ** 0.45) - 0.099))


def eocf_BT709(value):
    """
    Defines *Recommendation ITU-R BT.709-6* electro-optical conversion
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Warning
    -------
    *Recommendation ITU-R BT.709-6* doesn't specify an electro-optical
    conversion function. This definition is used for symmetry in unit tests and
    other computations but should not be used as an *EOCF* for *Rec. 709*
    colourspace!

    Examples
    --------
    >>> eocf_BT709(0.4090077288641504)  # doctest: +ELLIPSIS
    0.1...
    """

    warning(('*Recommendation ITU-R BT.709-6* doesn\'t specify an '
             'electro-optical conversion function. This definition is used '
             'for symmetry in unit tests and others computations but should '
             'not be used as an *EOCF*!'))

    value = np.asarray(value)

    return as_numeric(np.where(value < oecf_BT709(0.018),
                               value / 4.5,
                               ((value + 0.099) / 1.099) ** (1 / 0.45)))
