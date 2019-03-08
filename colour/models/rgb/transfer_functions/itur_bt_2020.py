# -*- coding: utf-8 -*-
"""
ITU-R BT.2020
=============

Defines *ITU-R BT.2020* opto-electrical transfer function (OETF / OECF) and
electro-optical transfer function (EOTF / EOCF):

-   :func:`colour.models.oetf_BT2020`
-   :func:`colour.models.eotf_BT2020`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`InternationalTelecommunicationUnion2015h` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.2020 - Parameter
    values for ultra-high definition television systems for production and
    international programme exchange. Retrieved from
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2020-2-201510-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import spow
from colour.utilities import (Structure, as_float, domain_range_scale,
                              from_range_1, to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'BT2020_CONSTANTS', 'BT2020_CONSTANTS_PRECISE', 'oetf_BT2020',
    'eotf_BT2020'
]

BT2020_CONSTANTS = Structure(
    alpha=lambda x: 1.0993 if x else 1.099,
    beta=lambda x: 0.0181 if x else 0.018)
"""
*BT.2020* colourspace constants.

BT2020_CONSTANTS : Structure
"""

BT2020_CONSTANTS_PRECISE = Structure(
    alpha=lambda x: 1.09929682680944, beta=lambda x: 0.018053968510807)
"""
*BT.2020* colourspace constants at double precision to connect the two curve
segments smoothly.

References
----------
:cite:`InternationalTelecommunicationUnion2015h`

BT2020_CONSTANTS_PRECISE : Structure
"""


def oetf_BT2020(E, is_12_bits_system=False, constants=BT2020_CONSTANTS):
    """
    Defines *Recommendation ITU-R BT.2020* opto-electrical transfer function
    (OETF / OECF).

    Parameters
    ----------
    E : numeric or array_like
        Voltage :math:`E` normalised by the reference white level and
        proportional to the implicit light intensity that would be detected
        with a reference camera colour channel R, G, B.
    is_12_bits_system : bool
        *BT.709* *alpha* and *beta* constants are used if system is not 12-bit.
    constants : Structure, optional
        *Recommendation ITU-R BT.2020* constants.

    Returns
    -------
    numeric or ndarray
        Resulting non-linear signal :math:`E'`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2015h`

    Examples
    --------
    >>> oetf_BT2020(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    """

    E = to_domain_1(E)

    a = constants.alpha(is_12_bits_system)
    b = constants.beta(is_12_bits_system)

    E_p = np.where(E < b, E * 4.5, a * spow(E, 0.45) - (a - 1))

    return as_float(from_range_1(E_p))


def eotf_BT2020(E_p, is_12_bits_system=False, constants=BT2020_CONSTANTS):
    """
    Defines *Recommendation ITU-R BT.2020* electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    E_p : numeric or array_like
        Non-linear signal :math:`E'`.
    is_12_bits_system : bool
        *BT.709* *alpha* and *beta* constants are used if system is not 12-bit.
    constants : Structure, optional
        *Recommendation ITU-R BT.2020* constants.

    Returns
    -------
    numeric or ndarray
        Resulting voltage :math:`E`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2015h`

    Examples
    --------
    >>> eotf_BT2020(0.705515089922121)  # doctest: +ELLIPSIS
    0.4999999...
    """

    E_p = to_domain_1(E_p)

    a = constants.alpha(is_12_bits_system)
    b = constants.beta(is_12_bits_system)

    with domain_range_scale('ignore'):
        E = np.where(
            E_p < oetf_BT2020(b),
            E_p / 4.5,
            spow((E_p + (a - 1)) / a, 1 / 0.45),
        )

    return as_float(from_range_1(E))
