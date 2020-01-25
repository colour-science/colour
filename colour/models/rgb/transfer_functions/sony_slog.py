# -*- coding: utf-8 -*-
"""
Sony S-Log Encodings
====================

Defines the *Sony S-Log* log encodings:

-   :func:`colour.models.log_encoding_SLog`
-   :func:`colour.models.log_decoding_SLog`
-   :func:`colour.models.log_encoding_SLog2`
-   :func:`colour.models.log_decoding_SLog2`
-   :func:`colour.models.log_encoding_SLog3`
-   :func:`colour.models.log_decoding_SLog3`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`SonyCorporation2012a` : Sony Corporation. (2012). S-Log2 Technical
    Paper. Retrieved from https://pro.sony.com/bbsccms/assets/files/micro/\
dmpc/training/S-Log2_Technical_PaperV1_0.pdf
-   :cite:`SonyCorporationd` : Sony Corporation. (n.d.). Technical Summary for
    S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3. Retrieved from
    http://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/2/\
TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import (as_float, domain_range_scale, from_range_1,
                              to_domain_1)
from colour.utilities.deprecation import handle_arguments_deprecation

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'log_encoding_SLog', 'log_decoding_SLog', 'log_encoding_SLog2',
    'log_decoding_SLog2', 'log_encoding_SLog3', 'log_decoding_SLog3'
]


def log_encoding_SLog(x,
                      bit_depth=10,
                      out_normalised_code_value=True,
                      in_reflection=True,
                      **kwargs):
    """
    Defines the *Sony S-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_normalised_code_value : bool, optional
        Whether the non-linear *Sony S-Log* data :math:`y` is encoded as
        normalised code values.
    in_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    numeric or ndarray
        Non-linear *Sony S-Log* data :math:`y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_encoding_SLog(0.18)  # doctest: +ELLIPSIS
    0.3849708...

    The values of *IRE and CV of S-Log2 @ISO800* table in
    :cite:`SonyCorporation2012a` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog(x, 10, False) * 100).astype(np.int)
    array([ 3, 38, 65])
    >>> np.around(log_encoding_SLog(x) * (2 ** 10 - 1)).astype(np.int)
    array([ 90, 394, 636])
    """

    out_normalised_code_value = handle_arguments_deprecation({
        'ArgumentRenamed': [['out_legal', 'out_normalised_code_value']],
    }, **kwargs).get('out_normalised_code_value', out_normalised_code_value)

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    y = np.where(
        x >= 0,
        ((0.432699 * np.log10(x + 0.037584) + 0.616596) + 0.03),
        x * 5 + 0.030001222851889303,
    )

    y = full_to_legal(y, bit_depth) if out_normalised_code_value else y

    return as_float(from_range_1(y))


def log_decoding_SLog(y,
                      bit_depth=10,
                      in_normalised_code_value=True,
                      out_reflection=True,
                      **kwargs):
    """
    Defines the *Sony S-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear *Sony S-Log* data :math:`y`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_normalised_code_value : bool, optional
        Whether the non-linear *Sony S-Log* data :math:`y` is encoded as
        normalised code values.
    out_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    numeric or ndarray
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_decoding_SLog(0.384970815928670)  # doctest: +ELLIPSIS
    0.1...
    """

    in_normalised_code_value = handle_arguments_deprecation({
        'ArgumentRenamed': [['in_legal', 'in_normalised_code_value']],
    }, **kwargs).get('in_normalised_code_value', in_normalised_code_value)

    y = to_domain_1(y)

    x = legal_to_full(y, bit_depth) if in_normalised_code_value else y

    with domain_range_scale('ignore'):
        x = np.where(
            y >= log_encoding_SLog(0.0, bit_depth, in_normalised_code_value),
            10 ** ((x - 0.616596 - 0.03) / 0.432699) - 0.037584,
            (x - 0.030001222851889303) / 5.0,
        )

    if out_reflection:
        x = x * 0.9

    return as_float(from_range_1(x))


def log_encoding_SLog2(x,
                       bit_depth=10,
                       out_normalised_code_value=True,
                       in_reflection=True,
                       **kwargs):
    """
    Defines the *Sony S-Log2* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_normalised_code_value : bool, optional
        Whether the non-linear *Sony S-Log2* data :math:`y` is encoded as
        normalised code values.
    in_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    numeric or ndarray
        Non-linear *Sony S-Log2* data :math:`y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_encoding_SLog2(0.18)  # doctest: +ELLIPSIS
    0.3395325...

    The values of *IRE and CV of S-Log2 @ISO800* table in
    :cite:`SonyCorporation2012a` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog2(x, 10, False) * 100).astype(np.int)
    array([ 3, 32, 59])
    >>> np.around(log_encoding_SLog2(x) * (2 ** 10 - 1)).astype(np.int)
    array([ 90, 347, 582])
    """

    out_normalised_code_value = handle_arguments_deprecation({
        'ArgumentRenamed': [['out_legal', 'out_normalised_code_value']],
    }, **kwargs).get('out_normalised_code_value', out_normalised_code_value)

    return log_encoding_SLog(x * 155 / 219, bit_depth,
                             out_normalised_code_value, in_reflection)


def log_decoding_SLog2(y,
                       bit_depth=10,
                       in_normalised_code_value=True,
                       out_reflection=True,
                       **kwargs):
    """
    Defines the *Sony S-Log2* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear *Sony S-Log2* data :math:`y`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_normalised_code_value : bool, optional
        Whether the non-linear *Sony S-Log2* data :math:`y` is encoded as
        normalised code values.
    out_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    numeric or ndarray
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_decoding_SLog2(0.339532524633774)  # doctest: +ELLIPSIS
    0.1...
    """

    in_normalised_code_value = handle_arguments_deprecation({
        'ArgumentRenamed': [['in_legal', 'in_normalised_code_value']],
    }, **kwargs).get('in_normalised_code_value', in_normalised_code_value)

    return 219 * log_decoding_SLog(y, bit_depth, in_normalised_code_value,
                                   out_reflection) / 155


def log_encoding_SLog3(x,
                       bit_depth=10,
                       out_normalised_code_value=True,
                       in_reflection=True,
                       **kwargs):
    """
    Defines the *Sony S-Log3* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_normalised_code_value : bool, optional
        Whether the non-linear *Sony S-Log3* data :math:`y` is encoded as
        normalised code values.
    in_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    numeric or ndarray
        Non-linear *Sony S-Log3* data :math:`y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporationd`

    Examples
    --------
    >>> log_encoding_SLog3(0.18)  # doctest: +ELLIPSIS
    0.4105571...

    The values of *S-Log3 10bit code values (18%, 90%)* table in
    :cite:`SonyCorporationd` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog3(x, 10, False) * 100).astype(np.int)
    array([ 4, 41, 61])
    >>> np.around(log_encoding_SLog3(x) * (2 ** 10 - 1)).astype(np.int)
    array([ 95, 420, 598])
    """

    out_normalised_code_value = handle_arguments_deprecation({
        'ArgumentRenamed': [['out_legal', 'out_normalised_code_value']],
    }, **kwargs).get('out_normalised_code_value', out_normalised_code_value)

    x = to_domain_1(x)

    if not in_reflection:
        x = x * 0.9

    y = np.where(
        x >= 0.01125000,
        (420 + np.log10((x + 0.01) / (0.18 + 0.01)) * 261.5) / 1023,
        (x * (171.2102946929 - 95) / 0.01125000 + 95) / 1023,
    )

    y = y if out_normalised_code_value else legal_to_full(y, bit_depth)

    return as_float(from_range_1(y))


def log_decoding_SLog3(y,
                       bit_depth=10,
                       in_normalised_code_value=True,
                       out_reflection=True,
                       **kwargs):
    """
    Defines the *Sony S-Log3* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear *Sony S-Log3* data :math:`y`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_normalised_code_value : bool, optional
        Whether the non-linear *Sony S-Log3* data :math:`y` is encoded as
        normalised code values.
    out_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    numeric or ndarray
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporationd`

    Examples
    --------
    >>> log_decoding_SLog3(0.410557184750733)  # doctest: +ELLIPSIS
    0.1...
    """

    in_normalised_code_value = handle_arguments_deprecation({
        'ArgumentRenamed': [['in_legal', 'in_normalised_code_value']],
    }, **kwargs).get('in_normalised_code_value', in_normalised_code_value)

    y = to_domain_1(y)

    y = y if in_normalised_code_value else full_to_legal(y, bit_depth)

    x = np.where(
        y >= 171.2102946929 / 1023,
        ((10 ** ((y * 1023 - 420) / 261.5)) * (0.18 + 0.01) - 0.01),
        (y * 1023 - 95) * 0.01125000 / (171.2102946929 - 95),
    )

    if not out_reflection:
        x = x / 0.9

    return as_float(from_range_1(x))
