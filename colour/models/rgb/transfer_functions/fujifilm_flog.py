# -*- coding: utf-8 -*-
"""
Fujifilm F-Log Log Encoding
===========================

Defines the *Fujifilm F-Log* log encoding:

-   :func:`colour.models.log_encoding_FLog`
-   :func:`colour.models.log_decoding_FLog`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Fujifilm2016` : Fujifilm. (2016). F-Log Data Sheet Ver.1.0.
    Retrieved from https://www.fujifilm.com/support/digital_cameras/\
software/lut/pdf/F-Log_DataSheet_E_Ver.1.0.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import Structure, as_float, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['FLOG_CONSTANTS', 'log_encoding_FLog', 'log_decoding_FLog']

FLOG_CONSTANTS = Structure(
    cut1=0.00089,
    cut2=0.100537775223865,
    a=0.555556,
    b=0.009468,
    c=0.344676,
    d=0.790453,
    e=8.735631,
    f=0.092864)
"""
*Fujifilm F-Log* colourspace constants.

FLOG_CONSTANTS : Structure
"""


def log_encoding_FLog(in_r,
                      bit_depth=10,
                      out_normalised_code_value=True,
                      in_reflection=True,
                      constants=FLOG_CONSTANTS):
    """
    Defines the *Fujifilm F-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    in_r : numeric or array_like
        Linear reflection data :math`in`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_normalised_code_value : bool, optional
        Whether the non-linear *Fujifilm F-Log* data :math:`out` is encoded as
        normalised code values.
    in_reflection : bool, optional
        Whether the light level :math`in` to a camera is reflection.
    constants : Structure, optional
        *Fujifilm F-Log* constants.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`out`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``in_r``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``out_r``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fujifilm2016`

    Examples
    --------
    >>> log_encoding_FLog(0.18)  # doctest: +ELLIPSIS
    0.4593184...

    The values of *2-2. F-Log Code Value* table in :cite:`Fujifilm2016` are
    obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_FLog(x, 10, False) * 100, 1)
    array([  3.5,  46.3,  73.2])
    >>> np.around(log_encoding_FLog(x) * (2 ** 10 - 1)).astype(np.int)
    array([ 95, 470, 705])
    """

    in_r = to_domain_1(in_r)

    if not in_reflection:
        in_r = in_r * 0.9

    cut1 = constants.cut1
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d
    e = constants.e
    f = constants.f

    out_r = np.where(
        in_r < cut1,
        e * in_r + f,
        c * np.log10(a * in_r + b) + d,
    )

    out_r = (out_r
             if out_normalised_code_value else legal_to_full(out_r, bit_depth))

    return as_float(from_range_1(out_r))


def log_decoding_FLog(out_r,
                      bit_depth=10,
                      in_normalised_code_value=True,
                      out_reflection=True,
                      constants=FLOG_CONSTANTS):
    """
    Defines the *Fujifilm F-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    out_r : numeric or array_like
        Non-linear data :math:`out`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_normalised_code_value : bool, optional
        Whether the non-linear *Fujifilm F-Log* data :math:`out` is encoded as
        normalised code values.
    out_reflection : bool, optional
        Whether the light level :math`in` to a camera is reflection.
    constants : Structure, optional
        *Fujifilm F-Log* constants.

    Returns
    -------
    numeric or ndarray
        Linear reflection data :math`in`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``out_r``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``in_r``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fujifilm2016`

    Examples
    --------
    >>> log_decoding_FLog(0.45931845866162124)  # doctest: +ELLIPSIS
    0.1800000...
    """

    out_r = to_domain_1(out_r)

    out_r = (out_r
             if in_normalised_code_value else full_to_legal(out_r, bit_depth))

    cut2 = constants.cut2
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d
    e = constants.e
    f = constants.f

    in_r = np.where(
        out_r < cut2,
        (out_r - f) / e,
        (10 ** ((out_r - d) / c)) / a - b / a,
    )

    if not out_reflection:
        in_r = in_r / 0.9

    return as_float(from_range_1(in_r))
