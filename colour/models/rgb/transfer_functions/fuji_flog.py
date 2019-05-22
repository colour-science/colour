# -*- coding: utf-8 -*-
"""
Fuji F-Log Log Encoding
=======================

Defines the *Fuji F-Log* log encoding:

-   :func:`colour.models.log_encoding_FLog`
-   :func:`colour.models.log_decoding_FLog`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Fujifilm2016` : Fujifilm. (2016). F-Log Data Sheet Ver.1.0. \
Retrieved from https://www.fujifilm.com/support/digital_cameras/\
software/lut/pdf/F-Log_DataSheet_E_Ver.1.0.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import Structure, as_float, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FLOG_CONSTANTS', 'log_encoding_FLog', 'log_decoding_FLog']

FLOG_CONSTANTS = Structure(
    cut1=0.00089, cut2=0.100537775223865, a=0.555556, b=0.009468, c=0.344676,
    d=0.790453, e=8.735631, f=0.092864)
"""
*Fuji F-Log* colourspace constants.

FLOG_CONSTANTS : Structure
"""


def log_encoding_FLog(L_in,
                      bit_depth=10,
                      out_legal=False,
                      constants=FLOG_CONSTANTS):
    """
    Defines the *Fuji F-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    L_in : numeric or array_like
        Linear reflection data :math`L_{in}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_legal : bool, optional
        Whether the non-linear *Fuji F-Log* data :math:`F_{out}` is
        encoded in legal range.
    constants : Structure, optional
        *Fuji F-Log* constants.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`F_{out}`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_in``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_out``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`FujiVer1`

    Examples
    --------
    >>> log_encoding_FLog(0.18)  # doctest: +ELLIPSIS
    0.4633365...
    """

    L_in = to_domain_1(L_in)

    cut1 = constants.cut1
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d
    e = constants.e
    f = constants.f

    F_out = np.where(
        L_in < cut1,
        e * L_in + f,
        c * np.log10(a * L_in + b) + d,
    )

    F_out = F_out if out_legal else legal_to_full(F_out, bit_depth)

    return as_float(from_range_1(F_out))


def log_decoding_FLog(F_out,
                      bit_depth=10,
                      in_legal=False,
                      constants=FLOG_CONSTANTS):
    """
    Defines the *Fuji F-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    F_out : numeric or array_like
        Non-linear data :math:`F_{out}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_legal : bool, optional
        Whether the non-linear *Panasonic V-Log* data :math:`V_{out}` is
        encoded in legal range.
    constants : Structure, optional
        *Fuji F-Log* constants.

    Returns
    -------
    numeric or ndarray
        Linear reflection data :math`L_{in}`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_out``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_in``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`FujiVer1`

    Examples
    --------
    >>> log_decoding_FLog(0.4633365)  # doctest: +ELLIPSIS
    0.1800000...
    """

    F_out = to_domain_1(F_out)

    F_out = F_out if in_legal else full_to_legal(F_out, bit_depth)

    cut2 = constants.cut2
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d
    e = constants.e
    f = constants.f

    L_in = np.where(
        F_out < cut2,
        (F_out - f) / e,
        (10 ** ((F_out - d) / c)) / a - b / a
    )

    return as_float(from_range_1(L_in))
