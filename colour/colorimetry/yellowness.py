#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yellowness Index :math:`Y`
==========================

Defines *yellowness* index :math:`Y` computation objects:

-   :func:`yellowness_ASTMD1925`
-   :func:`yellowness_ASTME313`

See Also
--------
`Yellowness Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/yellowness.ipynb>`_

References
----------
.. [1]  X-Rite, & Pantone. (2012). Color iQC and Color iMatch Color
        Calculations Guide. Retrieved from
        http://www.xrite.com/documents/literature/en/\
09_Color_Calculations_en.pdf
"""

from __future__ import division, unicode_literals

from colour.utilities import CaseInsensitiveMapping, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'yellowness_ASTMD1925', 'yellowness_ASTME313', 'YELLOWNESS_METHODS',
    'yellowness'
]


def yellowness_ASTMD1925(XYZ):
    """
    Returns the *yellowness* index :math:`YI` of given sample *CIE XYZ*
    tristimulus values using *ASTM D1925* method. [1]_

    ASTM D1925 has been specifically developed for the definition of the
    Yellowness of homogeneous, non-fluorescent, almost neutral-transparent,
    white-scattering or opaque plastics as they will be reviewed under daylight
    condition. It can be other materials as well, as long as they fit into this
    description.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`YI`.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 100].

    Warning
    -------
    The input domain of that definition is non standard!

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness_ASTMD1925(XYZ)  # doctest: +ELLIPSIS
    10.2999999...
    """

    X, Y, Z = tsplit(XYZ)

    YI = (100 * (1.28 * X - 1.06 * Z)) / Y

    return YI


def yellowness_ASTME313(XYZ):
    """
    Returns the *yellowness* index :math:`YI` of given sample *CIE XYZ*
    tristimulus values using *ASTM E313* method. [1]_

    ASTM E313 has successfully been used for a variety of white or near white
    materials. This includes coatings, Plastics, Textiles.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`YI`.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 100].

    Warning
    -------
    The input domain of that definition is non standard!

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness_ASTME313(XYZ)  # doctest: +ELLIPSIS
    11.0650000...
    """

    _X, Y, Z = tsplit(XYZ)

    WI = 100 * (1 - (0.847 * Z) / Y)

    return WI


YELLOWNESS_METHODS = CaseInsensitiveMapping({
    'ASTM D1925': yellowness_ASTMD1925,
    'ASTM E313': yellowness_ASTME313
})
"""
Supported *yellowness* computations methods.

YELLOWNESS_METHODS : CaseInsensitiveMapping
    **{'ASTM E313', 'ASTM D1925'}**
"""


def yellowness(XYZ, method='ASTM E313'):
    """
    Returns the *yellowness* :math:`W` using given method.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.
    method : unicode, optional
        **{'ASTM E313', 'ASTM D1925'}**,
        Computation method.

    Returns
    -------
    numeric or ndarray
        *yellowness* :math:`Y`.

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness(XYZ)  # doctest: +ELLIPSIS
    11.0650000...
    >>> method = 'ASTM D1925'
    >>> yellowness(XYZ, method=method)  # doctest: +ELLIPSIS
    10.2999999...
    """

    return YELLOWNESS_METHODS.get(method)(XYZ)
