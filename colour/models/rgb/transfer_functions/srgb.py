#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sRGB Colourspace
================

Defines *sRGB* colourspace opto-electrical transfer function (OETF / OECF) and
electro-optical transfer function (EOTF / EOCF):

-   :func:`oetf_sRGB`
-   :func:`eotf_sRGB`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Electrotechnical Commission. (1999). IEC 61966-2-1:1999 -
        Multimedia systems and equipment - Colour measurement and management -
        Part 2-1: Colour management - Default RGB colour space - sRGB, 51.
        Retrieved from https://webstore.iec.ch/publication/6169
.. [2]  International Telecommunication Union. (2015). Recommendation
        ITU-R BT.709-6 - Parameter values for the HDTV standards for
        production and international programme exchange BT Series Broadcasting
        service (Vol. 5). Retrieved from https://www.itu.int/dms_pubrec/\
itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric, warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oetf_sRGB',
           'eotf_sRGB']


def oetf_sRGB(L):
    """
    Defines the *sRGB* colourspace opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    L : numeric or array_like
        *Luminance* :math:`L` of the image.

    Returns
    -------
    numeric or ndarray
        Corresponding electrical signal :math:`V`.

    Examples
    --------
    >>> oetf_sRGB(0.18)  # doctest: +ELLIPSIS
    0.4613561...
    """

    L = np.asarray(L)

    return as_numeric(np.where(L <= 0.0031308,
                               L * 12.92,
                               1.055 * (L ** (1 / 2.4)) - 0.055))


def eotf_sRGB(V):
    """
    Defines the *sRGB* colourspace electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    V : numeric or array_like
        Electrical signal :math:`V`.

    Returns
    -------
    numeric or ndarray
        Corresponding *luminance* :math:`L` of the image.

    Examples
    --------
    >>> eotf_sRGB(0.461356129500442)  # doctest: +ELLIPSIS
    0.1...
    """

    warning(('*sRGB* *OETF* is a piece-wise function: in order to reduce '
             'noise in dark region, a line segment limits the slope of the '
             'power function (slope of a power function is infinite at zero). '
             'This is not needed for *sRGB* *EOTF*, a pure gamma 2.2 function '
             'should be use instead. This definition is used for symmetry in '
             'unit tests and others computations but should not be used as an '
             '*EOTF*!'))

    V = np.asarray(V)

    return as_numeric(np.where(V <= oetf_sRGB(0.0031308),
                               V / 12.92,
                               ((V + 0.055) / 1.055) ** 2.4))
