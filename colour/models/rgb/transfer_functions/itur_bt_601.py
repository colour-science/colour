# -*- coding: utf-8 -*-
"""
ITU-R BT.601-7
==============

Defines *ITU-R BT.601-7* opto-electrical transfer function (OETF / OECF) and
its reverse:

-   :func:`colour.models.oetf_BT601`
-   :func:`colour.models.oetf_reverse_BT601`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`InternationalTelecommunicationUnion2011f` : International
    Telecommunication Union. (2011). Recommendation ITU-R BT.601-7 - Studio
    encoding parameters of digital television for standard 4:3 and wide-screen
    16:9 aspect ratios. Retrieved from
    http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.601-7-201103-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oetf_BT601', 'oetf_reverse_BT601']


def oetf_BT601(L):
    """
    Defines *Recommendation ITU-R BT.601-7* opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    L : numeric or array_like
        *Luminance* :math:`L` of the image.

    Returns
    -------
    numeric or ndarray
        Corresponding electrical signal :math:`Es`.

    References
    ----------
    -   :cite:`InternationalTelecommunicationUnion2011f`

    Examples
    --------
    >>> oetf_BT601(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    """

    L = np.asarray(L)

    return as_numeric(
        np.where(L < 0.018, L * 4.5, 1.099 * (L ** 0.45) - 0.099))


def oetf_reverse_BT601(E):
    """
    Defines *Recommendation ITU-R BT.601-7* reverse opto-electronic transfer
    function (OETF / OECF).

    Parameters
    ----------
    E : numeric or array_like
        Electrical signal :math:`E`.

    Returns
    -------
    numeric or ndarray
        Corresponding *luminance* :math:`L` of the image.

    References
    ----------
    -   :cite:`InternationalTelecommunicationUnion2011f`

    Examples
    --------
    >>> oetf_reverse_BT601(0.409007728864150)  # doctest: +ELLIPSIS
    0.1...
    """

    E = np.asarray(E)

    return as_numeric(
        np.where(E < oetf_BT601(0.018), E / 4.5, ((E + 0.099) / 1.099) ** (
            1 / 0.45)))
