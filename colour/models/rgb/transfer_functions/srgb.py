#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sRGB OETF (OECF) and EOTF (EOCF)
================================

Defines *sRGB* OETF (OECF) and EOTF (EOCF):

-   :func:`oetf_sRGB`
-   :func:`eotf_sRGB`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Electrotechnical Commission. (1999). IEC 61966-2-1:1999 -
        Multimedia systems and equipment - Colour measurement and management -
        Part 2-1: Colour management - Default RGB colour space - sRGB, 51.
        Retrieved from https://webstore.iec.ch/publication/6169
.. [2]  International Telecommunication Union. (2002). Parameter values for
        the HDTV standards for production and international programme exchange
        BT Series Broadcasting service. In Recommendation ITU-R BT.709-6
        (Vol. 5, pp. 1–32). ISBN:9519982000
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

__all__ = ['oetf_sRGB',
           'eotf_sRGB']


def oetf_sRGB(value):
    """
    Defines the *sRGB* colourspace OETF (OECF) opto-electronic transfer
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
    >>> oetf_sRGB(0.18)  # doctest: +ELLIPSIS
    0.4613561...
    """

    value = np.asarray(value)

    return as_numeric(np.where(value <= 0.0031308,
                               value * 12.92,
                               1.055 * (value ** (1 / 2.4)) - 0.055))


def eotf_sRGB(value):
    """
    Defines the *sRGB* colourspace EOTF (EOCF) electro-optical transfer
    function.

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
    >>> eotf_sRGB(0.46135612950044164)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    return as_numeric(np.where(value <= oetf_sRGB(0.0031308),
                               value / 12.92,
                               ((value + 0.055) / 1.055) ** 2.4))
