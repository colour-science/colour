#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ITU-R BT.2020
=============

Defines *ITU-R BT.2020* opto-electrical transfer function (OETF / OECF) and
electro-optical transfer function (EOTF / EOCF):

-   :func:`oetf_BT2020`
-   :func:`eotf_BT2020`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (2015). Recommendation
        ITU-R BT.2020 - Parameter values for ultra-high definition television
        systems for production and international programme exchange (Vol. 1).
        Retrieved from https://www.itu.int/dms_pubrec/\
itu-r/rec/bt/R-REC-BT.2020-2-201510-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import Structure, as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['BT2020_CONSTANTS',
           'oetf_BT2020',
           'eotf_BT2020']

BT2020_CONSTANTS = Structure(alpha=lambda x: 1.0993 if x else 1.099,
                             beta=lambda x: 0.0181 if x else 0.018)
"""
*BT.2020* colourspace constants.

BT2020_CONSTANTS : Structure
"""


def oetf_BT2020(E, is_12_bits_system=False):
    """
    Defines *Recommendation ITU-R BT.2020* opto-electrical transfer function
    (OETF / OECF).

    Parameters
    ----------
    E : numeric or array_like
        Voltage :math:`E` normalized by the reference white level and
        proportional to the implicit light intensity that would be detected
        with a reference camera colour channel R, G, B.
    is_12_bits_system : bool
        *BT.709* *alpha* and *beta* constants are used if system is not 12-bit.

    Returns
    -------
    numeric or ndarray
        Resulting non-linear signal :math:`E'`.

    Examples
    --------
    >>> oetf_BT2020(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    """

    E = np.asarray(E)

    a = BT2020_CONSTANTS.alpha(is_12_bits_system)
    b = BT2020_CONSTANTS.beta(is_12_bits_system)

    return as_numeric(np.where(E < b,
                               E * 4.5,
                               a * (E ** 0.45) - (a - 1)))


def eotf_BT2020(E_p, is_12_bits_system=False):
    """
    Defines *Recommendation ITU-R BT.2020* electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    E_p : numeric or array_like
        Non-linear signal :math:`E'`.
    is_12_bits_system : bool
        *BT.709* *alpha* and *beta* constants are used if system is not 12-bit.

    Returns
    -------
    numeric or ndarray
        Resulting voltage :math:`E`.

    Examples
    --------
    >>> eotf_BT2020(0.705515089922121)  # doctest: +ELLIPSIS
    0.4999999...
    """

    E_p = np.asarray(E_p)

    a = BT2020_CONSTANTS.alpha(is_12_bits_system)
    b = BT2020_CONSTANTS.beta(is_12_bits_system)

    return as_numeric(np.where(E_p < oetf_BT2020(b),
                               E_p / 4.5,
                               ((E_p + (a - 1)) / a) ** (1 / 0.45)))
