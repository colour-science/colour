#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ITU-R BT.2020 EOTF / EOCF and OETF / EOCF
=========================================

Defines *ITU-R BT.2020* EOTF / EOCF and OETF / EOCF:

-   :func:`oetf_BT2020`
-   :func:`eotf_BT2020`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (2014). Parameter values for
        ultra-high definition television systems for production and
        international programme exchange. In Recommendation ITU-R BT.2020
        (Vol. 1, pp. 1–8). Retrieved from
        http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2020-1-201406-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import Structure, as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['BT_2020_CONSTANTS',
           'oetf_BT2020',
           'eotf_BT2020']

BT_2020_CONSTANTS = Structure(alpha=lambda x: 1.099 if x else 1.0993,
                              beta=lambda x: 0.018 if x else 0.0181)
"""
*BT.2020* colourspace constants.

BT_2020_CONSTANTS : Structure
"""


def oetf_BT2020(value, is_10_bits_system=True):
    """
    Defines *Recommendation ITU-R BT.2020* opto-electrical transfer function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    is_10_bits_system : bool
        *BT.709* *alpha* and *beta* constants are used if system is 10 bit.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> oetf_BT2020(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    """

    value = np.asarray(value)

    a = BT_2020_CONSTANTS.alpha(is_10_bits_system)
    b = BT_2020_CONSTANTS.beta(is_10_bits_system)

    return as_numeric(np.where(value < b,
                               value * 4.5,
                               a * (value ** 0.45) - (a - 1)))


def eotf_BT2020(value, is_10_bits_system=True):
    """
    Defines *Recommendation ITU-R BT.2020* electro-optical transfer function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    is_10_bits_system : bool
        *BT.709* *alpha* and *beta* constants are used if system is 10 bit.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> eotf_BT2020(0.7055150899221212)  # doctest: +ELLIPSIS
    0.5...
    """

    value = np.asarray(value)

    a = BT_2020_CONSTANTS.alpha(is_10_bits_system)
    b = BT_2020_CONSTANTS.beta(is_10_bits_system)

    return as_numeric(np.where(value < oetf_BT2020(b),
                               value / 4.5,
                               ((value + (a - 1)) / a) ** (1 / 0.45)))
