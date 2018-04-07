# -*- coding: utf-8 -*-
"""
ARIB STD-B67 (Hybrid Log-Gamma)
===============================

Defines *ARIB STD-B67 (Hybrid Log-Gamma)* opto-electrical transfer function
(OETF / OECF) and its reverse:

-   :func:`colour.models.oetf_ARIBSTDB67`
-   :func:`colour.models.oetf_reverse_ARIBSTDB67`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`AssociationofRadioIndustriesandBusinesses2015a` : Association of
    Radio Industries and Businesses. (2015). Essential Parameter Values for the
    Extended Image Dynamic Range Television (EIDRTV) System for Programme
    Production. Retrieved from
    https://www.arib.or.jp/english/std_tr/broadcasting/desc/std-b67.html
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import (Structure, as_numeric, domain_range_scale,
                              from_range_1, to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ARIBSTDB67_CONSTANTS', 'oetf_ARIBSTDB67', 'oetf_reverse_ARIBSTDB67'
]

ARIBSTDB67_CONSTANTS = Structure(a=0.17883277, b=0.28466892, c=0.55991073)
"""
*ARIB STD-B67 (Hybrid Log-Gamma)* constants.

ARIBSTDB67_CONSTANTS : Structure
"""


def oetf_ARIBSTDB67(E, r=0.5, constants=ARIBSTDB67_CONSTANTS):
    """
    Defines *ARIB STD-B67 (Hybrid Log-Gamma)* opto-electrical transfer
    function (OETF / OECF).

    Parameters
    ----------
    E : numeric or array_like
        Voltage normalised by the reference white level and proportional to
        the implicit light intensity that would be detected with a reference
        camera color channel R, G, B.
    r : numeric, optional
        Video level corresponding to reference white level.
    constants : Structure, optional
        *ARIB STD-B67 (Hybrid Log-Gamma)* constants.

    Returns
    -------
    numeric or ndarray
        Resulting non-linear signal :math:`E'`.

    References
    ----------
    -   :cite:`AssociationofRadioIndustriesandBusinesses2015a`

    Examples
    --------
    >>> oetf_ARIBSTDB67(0.18)  # doctest: +ELLIPSIS
    0.2121320...
    """

    E = to_domain_1(E)

    a = constants.a
    b = constants.b
    c = constants.c

    E_p = np.where(E <= 1, r * np.sqrt(E), a * np.log(E - b) + c)

    return as_numeric(from_range_1(E_p))


def oetf_reverse_ARIBSTDB67(E_p, r=0.5, constants=ARIBSTDB67_CONSTANTS):
    """
    Defines *ARIB STD-B67 (Hybrid Log-Gamma)* reverse opto-electrical transfer
    function (OETF / OECF).

    Parameters
    ----------
    E_p : numeric or array_like
        Non-linear signal :math:`E'`.
    r : numeric, optional
        Video level corresponding to reference white level.
    constants : Structure, optional
        *ARIB STD-B67 (Hybrid Log-Gamma)* constants.

    Returns
    -------
    numeric or ndarray
        Voltage :math:`E` normalised by the reference white level and
        proportional to the implicit light intensity that would be detected
        with a reference camera color channel R, G, B.

    References
    ----------
    -   :cite:`AssociationofRadioIndustriesandBusinesses2015a`

    Examples
    --------
    >>> oetf_reverse_ARIBSTDB67(0.212132034355964)  # doctest: +ELLIPSIS
    0.1799999...
    """

    E_p = to_domain_1(E_p)

    a = constants.a
    b = constants.b
    c = constants.c

    with domain_range_scale('ignore'):
        E = np.where(E_p <= oetf_ARIBSTDB67(1), (E_p / r) ** 2,
                     np.exp((E_p - c) / a) + b)

    return as_numeric(from_range_1(E))
