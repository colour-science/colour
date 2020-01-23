# -*- coding: utf-8 -*-
"""
SMPTE ST 2084:2014
==================

Defines *SMPTE ST 2084:2014* opto-electrical transfer function (OETF / OECF)
and electro-optical transfer function (EOTF / EOCF):

-   :func:`colour.models.eotf_ST2084`
-   :func:`colour.models.eotf_inverse_ST2084`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Miller2014a` : Miller, S. (2014). A Perceptual EOTF for Extended
    Dynamic Range Imagery. Retrieved from https://www.smpte.org/sites/default/\
files/2014-05-06-EOTF-Miller-1-2-handout.pdf
-   :cite:`SocietyofMotionPictureandTelevisionEngineers2014a` : Society of
    Motion Picture and Television Engineers. (2014). SMPTE ST 2084:2014 -
    Dynamic Range Electro-Optical Transfer Function of Mastering Reference
    Displays. doi:10.5594/SMPTE.ST2084.2014
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import spow
from colour.utilities import Structure, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['ST2084_CONSTANTS', 'eotf_inverse_ST2084', 'eotf_ST2084']

ST2084_CONSTANTS = Structure(
    m_1=2610 / 4096 * (1 / 4),
    m_2=2523 / 4096 * 128,
    c_1=3424 / 4096,
    c_2=2413 / 4096 * 32,
    c_3=2392 / 4096 * 32)
"""
Constants for *SMPTE ST 2084:2014* inverse electro-optical transfer function
(EOTF / EOCF) and electro-optical transfer function (EOTF / EOCF).

ST2084_CONSTANTS : Structure
"""


def eotf_inverse_ST2084(C, L_p=10000, constants=ST2084_CONSTANTS):
    """
    Defines *SMPTE ST 2084:2014* optimised perceptual inverse electro-optical
    transfer function (EOTF / EOCF).

    Parameters
    ----------
    C : numeric or array_like
        Target optical output :math:`C` in :math:`cd/m^2` of the ideal
        reference display.
    L_p : numeric, optional
        System peak luminance :math:`cd/m^2`, this parameter should stay at its
        default :math:`10000 cd/m^2` value for practical applications. It is
        exposed so that the definition can be used as a fitting function.
    constants : Structure, optional
        *SMPTE ST 2084:2014* constants.

    Returns
    -------
    numeric or ndarray
        Color value abbreviated as :math:`N`, that is directly proportional to
        the encoded signal representation, and which is not directly
        proportional to the optical output of a display device.

    Warnings
    --------
    *SMPTE ST 2084:2014* is an absolute transfer function.

    Notes
    -----

    -   *SMPTE ST 2084:2014* is an absolute transfer function, thus the
        domain and range values for the *Reference* and *1* scales are only
        indicative that the data is not affected by scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``C``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``N``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Miller2014a`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers2014a`

    Examples
    --------
    >>> eotf_inverse_ST2084(100)  # doctest: +ELLIPSIS
    0.5080784...
    """

    C = to_domain_1(C)

    Y_p = spow(C / L_p, constants.m_1)

    N = spow((constants.c_1 + constants.c_2 * Y_p) / (constants.c_3 * Y_p + 1),
             constants.m_2)

    return from_range_1(N)


def eotf_ST2084(N, L_p=10000, constants=ST2084_CONSTANTS):
    """
    Defines *SMPTE ST 2084:2014* optimised perceptual electro-optical transfer
    function (EOTF / EOCF).

    This perceptual quantizer (PQ) has been modeled by Dolby Laboratories
    using *Barten (1999)* contrast sensitivity function.

    Parameters
    ----------
    N : numeric or array_like
        Color value abbreviated as :math:`N`, that is directly proportional to
        the encoded signal representation, and which is not directly
        proportional to the optical output of a display device.
    L_p : numeric, optional
        System peak luminance :math:`cd/m^2`, this parameter should stay at its
        default :math:`10000 cd/m^2` value for practical applications. It is
        exposed so that the definition can be used as a fitting function.
    constants : Structure, optional
        *SMPTE ST 2084:2014* constants.

    Returns
    -------
    numeric or ndarray
          Target optical output :math:`C` in :math:`cd/m^2` of the ideal
          reference display.

    Warnings
    --------
    *SMPTE ST 2084:2014* is an absolute transfer function.

    Notes
    -----

    -   *SMPTE ST 2084:2014* is an absolute transfer function, thus the
        domain and range values for the *Reference* and *1* scales are only
        indicative that the data is not affected by scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``N``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``C``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Miller2014a`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers2014a`

    Examples
    --------
    >>> eotf_ST2084(0.508078421517399)  # doctest: +ELLIPSIS
    100.0000000...
    """

    N = to_domain_1(N)

    m_1_d = 1 / constants.m_1
    m_2_d = 1 / constants.m_2

    V_p = spow(N, m_2_d)

    n = V_p - constants.c_1
    # Limiting negative values.
    n = np.where(n < 0, 0, n)

    L = spow((n / (constants.c_2 - constants.c_3 * V_p)), m_1_d)
    C = L_p * L

    return from_range_1(C)
