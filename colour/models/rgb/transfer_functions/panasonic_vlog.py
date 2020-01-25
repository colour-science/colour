# -*- coding: utf-8 -*-
"""
Panasonic V-Log Log Encoding
============================

Defines the *Panasonic V-Log* log encoding:

-   :func:`colour.models.log_encoding_VLog`
-   :func:`colour.models.log_decoding_VLog`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Panasonic2014a` : Panasonic. (2014). VARICAM V-Log/V-Gamut.
    Retrieved from http://pro-av.panasonic.net/en/varicam/common/pdf/\
VARICAM_V-Log_V-Gamut.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import Structure, as_float, from_range_1, to_domain_1
from colour.utilities.deprecation import handle_arguments_deprecation

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['VLOG_CONSTANTS', 'log_encoding_VLog', 'log_decoding_VLog']

VLOG_CONSTANTS = Structure(
    cut1=0.01, cut2=0.181, b=0.00873, c=0.241514, d=0.598206)
"""
*Panasonic V-Log* colourspace constants.

VLOG_CONSTANTS : Structure
"""


def log_encoding_VLog(L_in,
                      bit_depth=10,
                      out_normalised_code_value=True,
                      in_reflection=True,
                      constants=VLOG_CONSTANTS,
                      **kwargs):
    """
    Defines the *Panasonic V-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    L_in : numeric or array_like
        Linear reflection data :math`L_{in}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_normalised_code_value : bool, optional
        Whether the non-linear *Panasonic V-Log* data :math:`V_{out}` is
        encoded as normalised code values.
    in_reflection : bool, optional
        Whether the light level :math`L_{in}` to a camera is reflection.
    constants : Structure, optional
        *Panasonic V-Log* constants.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`V_{out}`.

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
    | ``V_out``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Panasonic2014a`

    Examples
    --------
    >>> log_encoding_VLog(0.18)  # doctest: +ELLIPSIS
    0.4233114...

    The values of *Fig.2.2 V-Log Code Value* table in :cite:`Panasonic2014a`
    are obtained as follows:

    >>> L_in = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_VLog(L_in, 10, False) * 100).astype(np.int)
    array([ 7, 42, 61])
    >>> np.around(log_encoding_VLog(L_in) * (2 ** 10 - 1)).astype(np.int)
    array([128, 433, 602])
    >>> np.around(log_encoding_VLog(L_in) * (2 ** 12 - 1)).astype(np.int)
    array([ 512, 1733, 2409])

    Note that some values in the last column values of
    *Fig.2.2 V-Log Code Value* table in :cite:`Panasonic2014a` are different
    by a code: [512, 1732, 2408].
    """

    out_normalised_code_value = handle_arguments_deprecation({
        'ArgumentRenamed': [['out_legal', 'out_normalised_code_value']],
    }, **kwargs).get('out_normalised_code_value', out_normalised_code_value)

    L_in = to_domain_1(L_in)

    if not in_reflection:
        L_in = L_in * 0.9

    cut1 = constants.cut1
    b = constants.b
    c = constants.c
    d = constants.d

    V_out = np.where(
        L_in < cut1,
        5.6 * L_in + 0.125,
        c * np.log10(L_in + b) + d,
    )

    V_out = (V_out
             if out_normalised_code_value else legal_to_full(V_out, bit_depth))

    return as_float(from_range_1(V_out))


def log_decoding_VLog(V_out,
                      bit_depth=10,
                      in_normalised_code_value=True,
                      out_reflection=True,
                      constants=VLOG_CONSTANTS,
                      **kwargs):
    """
    Defines the *Panasonic V-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    V_out : numeric or array_like
        Non-linear data :math:`V_{out}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_normalised_code_value : bool, optional
        Whether the non-linear *Panasonic V-Log* data :math:`V_{out}` is
        encoded as normalised code values.
    out_reflection : bool, optional
        Whether the light level :math`L_{in}` to a camera is reflection.
    constants : Structure, optional
        *Panasonic V-Log* constants.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    numeric or ndarray
        Linear reflection data :math`L_{in}`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V_out``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_in``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Panasonic2014a`

    Examples
    --------
    >>> log_decoding_VLog(0.423311448760136)  # doctest: +ELLIPSIS
    0.1799999...
    """

    in_normalised_code_value = handle_arguments_deprecation({
        'ArgumentRenamed': [['in_legal', 'in_normalised_code_value']],
    }, **kwargs).get('in_normalised_code_value', in_normalised_code_value)

    V_out = to_domain_1(V_out)

    V_out = (V_out
             if in_normalised_code_value else full_to_legal(V_out, bit_depth))

    cut2 = constants.cut2
    b = constants.b
    c = constants.c
    d = constants.d

    L_in = np.where(
        V_out < cut2,
        (V_out - 0.125) / 5.6,
        10 ** ((V_out - d) / c) - b,
    )

    if not out_reflection:
        L_in = L_in / 0.9

    return as_float(from_range_1(L_in))
