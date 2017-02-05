#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DCI-P3 Colourspace
==================

Defines the *DCI-P3* colourspace opto-electrical transfer function
(OETF / OECF) and electro-optical transfer function (EOTF / EOCF):

-   :func:`oetf_DCIP3`
-   :func:`eotf_DCIP3`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Digital Cinema Initiatives. (2007). Digital Cinema System
        Specification - Version 1.1. Retrieved from
        http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oetf_DCIP3',
           'eotf_DCIP3']


def oetf_DCIP3(XYZ):
    """
    Defines the *DCI-P3* colourspace opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    XYZ : numeric or array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    numeric or ndarray
        Non-linear *CIE XYZ'* tristimulus values.

    Examples
    --------
    >>> oetf_DCIP3(0.18)  # doctest: +ELLIPSIS
    461.9922059...
    """

    XYZ = np.asarray(XYZ)

    return 4095 * (XYZ / 52.37) ** (1 / 2.6)


def eotf_DCIP3(XYZ_p):
    """
    Defines the *DCI-P3* colourspace electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    XYZ_p : numeric or array_like
        Non-linear *CIE XYZ'* tristimulus values.

    Returns
    -------
    numeric or ndarray
        *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> eotf_DCIP3(461.99220597484737)  # doctest: +ELLIPSIS
    0.18...
    """

    XYZ_p = np.asarray(XYZ_p)

    return 52.37 * (XYZ_p / 4095) ** 2.6
