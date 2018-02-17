# -*- coding: utf-8 -*-
"""
SMPTE 240M
==========

Defines *SMPTE 240M* opto-electrical transfer function (OETF / OECF) and
electro-optical transfer function (EOTF / EOCF):

-   :func:`colour.models.oetf_SMPTE240M`
-   :func:`colour.models.eotf_SMPTE240M`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`SocietyofMotionPictureandTelevisionEngineers1999b` : Society of
    Motion Picture and Television Engineers. (1999). ANSI/SMPTE 240M-1995 -
    Signal Parameters - 1125-Line High-Definition Production Systems. Retrieved
    from http://car.france3.mars.free.fr/HD/INA- 26 jan 06/\
SMPTE normes et confs/s240m.pdf
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

__all__ = ['oetf_SMPTE240M', 'eotf_SMPTE240M']


def oetf_SMPTE240M(L_c):
    """
    Defines *SMPTE 240M* opto-electrical transfer function (OETF / OECF).

    Parameters
    ----------
    L_c : numeric or array_like
        Light input :math:`L_c` to the reference camera normalized to the
        system reference white.

    Returns
    -------
    numeric or ndarray
        Video signal output :math:`V_c` of the reference camera normalized to
        the system reference white.

    References
    ----------
    -   :cite:`SocietyofMotionPictureandTelevisionEngineers1999b`

    Examples
    --------
    >>> oetf_SMPTE240M(0.18)  # doctest: +ELLIPSIS
    0.4022857...
    """

    L_c = np.asarray(L_c)

    return as_numeric(
        np.where(L_c < 0.0228, 4 * L_c, 1.1115 * L_c ** 0.45 - 0.1115))


def eotf_SMPTE240M(V_r):
    """
    Defines *SMPTE 240M* electro-optical transfer function (EOTF / EOCF).

    Parameters
    ----------
    V_r : numeric or array_like
        Video signal level :math:`V_r` driving the reference reproducer
        normalized to the system reference white.

    Returns
    -------
    numeric or ndarray
         Light output :math:`L_r` from the reference reproducer normalized to
         the system reference white.

    References
    ----------
    -   :cite:`SocietyofMotionPictureandTelevisionEngineers1999b`

    Examples
    --------
    >>> eotf_SMPTE240M(0.402285796753870)  # doctest: +ELLIPSIS
    0.1...
    """

    V_r = np.asarray(V_r)

    return as_numeric(
        np.where(V_r < oetf_SMPTE240M(0.0228), V_r / 4, ((
            V_r + 0.1115) / 1.1115) ** (1 / 0.45)))
