#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Log Conversion
==============

Defines various *linear* to *log* and *log* to *linear* conversion functions:

-   :attr:`linear_to_cineon`
-   :attr:`cineon_to_linear`
-   :attr:`linear_to_panalog`
-   :attr:`panalog_to_linear`
-   :attr:`linear_to_red_log`
-   :attr:`red_log_to_linear`
-   :attr:`linear_to_viper_log`
-   :attr:`viper_log_to_linear`
-   :attr:`linear_to_pivoted_log`
-   :attr:`pivoted_log_to_linear`
-   :attr:`linear_to_c_log`
-   :attr:`c_log_to_linear`
-   :attr:`linear_to_aces_cc`
-   :attr:`aces_cc_to_linear`
-   :attr:`linear_to_alexa_log_c`
-   :attr:`alexa_log_c_to_linear`
-   :attr:`linear_to_dci_p3_log`
-   :attr:`dci_p3_log_to_linear`
-   :attr:`linear_to_s_log`
-   :attr:`s_log_to_linear`
-   :attr:`linear_to_s_log2`
-   :attr:`s_log2_to_linear`
-   :attr:`linear_to_s_log3`
-   :attr:`s_log3_to_linear`

See Also
--------
`Log Conversion IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/log.ipynb>`_  # noqa

References
----------
.. [1]  Sony Imageworks. (2012). make.py. Retrieved November 27, 2014, from
        https://github.com/imageworks/OpenColorIO-Configs/blob/master/nuke-default/make.py  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.dataset.aces import (
    ACES_CC_TRANSFER_FUNCTION,
    ACES_CC_INVERSE_TRANSFER_FUNCTION)
from colour.models.dataset.alexa_wide_gamut_rgb import (
    ALEXA_WIDE_GAMUT_RGB_TRANSFER_FUNCTION,
    ALEXA_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION)
from colour.models.dataset.dci_p3 import (
    DCI_P3_TRANSFER_FUNCTION,
    DCI_P3_INVERSE_TRANSFER_FUNCTION)
from colour.models.dataset.s_gamut import (
    S_LOG_TRANSFER_FUNCTION,
    S_LOG2_TRANSFER_FUNCTION,
    S_LOG3_TRANSFER_FUNCTION,
    S_LOG_INVERSE_TRANSFER_FUNCTION,
    S_LOG2_INVERSE_TRANSFER_FUNCTION,
    S_LOG3_INVERSE_TRANSFER_FUNCTION)
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['linear_to_cineon',
           'cineon_to_linear',
           'linear_to_panalog',
           'panalog_to_linear',
           'linear_to_red_log',
           'red_log_to_linear',
           'linear_to_viper_log',
           'viper_log_to_linear',
           'linear_to_pivoted_log',
           'pivoted_log_to_linear',
           'linear_to_c_log',
           'c_log_to_linear',
           'linear_to_aces_cc',
           'aces_cc_to_linear',
           'linear_to_alexa_log_c',
           'alexa_log_c_to_linear',
           'linear_to_dci_p3_log',
           'dci_p3_log_to_linear',
           'linear_to_s_log',
           's_log_to_linear',
           'linear_to_s_log2',
           's_log2_to_linear',
           'linear_to_s_log3',
           's_log3_to_linear',
           'LINEAR_TO_LOG_METHODS',
           'LOG_TO_LINEAR_METHODS',
           'linear_to_log',
           'log_to_linear']


def linear_to_cineon(value, black_offset=10 ** ((95 - 685) / 300), **kwargs):
    """
    Defines the *linear* to *Cineon* conversion function.

    Parameters
    ----------
    value : numeric
        *Linear* value.
    black_offset : numeric
        Black offset.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *Cineon* value.

    Examples
    --------
    >>> linear_to_cineon(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    return ((685 +
             300 * np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def cineon_to_linear(value, black_offset=10 ** ((95 - 685) / 300), **kwargs):
    """
    Defines the *Cineon* to *linear* conversion function.

    Parameters
    ----------
    value : numeric
        *Cineon* value.
    black_offset : numeric
        Black offset.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *Linear* value.

    Examples
    --------
    >>> cineon_to_linear(0.45731961308541841)  # doctest: +ELLIPSIS
    0.18...
    """

    return ((10 ** ((1023 * value - 685) / 300) - black_offset) /
            (1 - black_offset))


def linear_to_panalog(value, black_offset=10 ** ((64 - 681) / 444), **kwargs):
    """
    Defines the *linear* to *Panalog* conversion function.

    Parameters
    ----------
    value : numeric
        *Linear* value.
    black_offset : numeric
        Black offset.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *Panalog* value.

    Examples
    --------
    >>> linear_to_panalog(0.18)  # doctest: +ELLIPSIS
    0.3745767...
    """

    return ((681 +
             444 * np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def panalog_to_linear(value, black_offset=10 ** ((64 - 681) / 444), **kwargs):
    """
    Defines the *Panalog* to *linear* conversion function.

    Parameters
    ----------
    value : numeric
        *Panalog* value.
    black_offset : numeric
        Black offset.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *Linear* value.

    Examples
    --------
    >>> panalog_to_linear(0.37457679138229816)  # doctest: +ELLIPSIS
    0.1...
    """

    return ((10 ** ((1023 * value - 681) / 444) - black_offset) /
            (1 - black_offset))


def linear_to_red_log(value, black_offset=10 ** ((0 - 1023) / 511), **kwargs):
    """
    Defines the *linear* to *REDLog* conversion function.

    Parameters
    ----------
    value : numeric
        *Linear* value.
    black_offset : numeric
        Black offset.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *REDLog* value.

    Examples
    --------
    >>> linear_to_red_log(0.18)  # doctest: +ELLIPSIS
    0.6376218...
    """

    return ((1023 +
             511 * np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def red_log_to_linear(value, black_offset=10 ** ((0 - 1023) / 511), **kwargs):
    """
    Defines the *REDLog* to *linear* conversion function.

    Parameters
    ----------
    value : numeric
        *REDLog* value.
    black_offset : numeric
        Black offset.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *Linear* value.

    Examples
    --------
    >>> red_log_to_linear(0.63762184598817484)  # doctest: +ELLIPSIS
    0.1...
    """

    return (((10 ** ((1023 * value - 1023) / 511)) - black_offset) /
            (1 - black_offset))


def linear_to_viper_log(value, **kwargs):
    """
    Defines the *linear* to *ViperLog* conversion function.

    Parameters
    ----------
    value : numeric
        *Linear* value.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *ViperLog* value.

    Examples
    --------
    >>> linear_to_viper_log(0.18)  # doctest: +ELLIPSIS
    0.6360080...
    """

    return (1023 + 500 * np.log10(value)) / 1023


def viper_log_to_linear(value, **kwargs):
    """
    Defines the *ViperLog* to *linear* conversion function.

    Parameters
    ----------
    value : numeric
        *ViperLog* value.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *Linear* value.

    Examples
    --------
    >>> viper_log_to_linear(0.63600806701041346)  # doctest: +ELLIPSIS
    0.1799999...
    """

    return 10 ** ((1023 * value - 1023) / 500)


def linear_to_pivoted_log(value,
                          log_reference=445,
                          linear_reference=0.18,
                          negative_gamma=0.6,
                          density_per_code_value=0.002):
    """
    Defines the *linear* to *Josh Pines* style pivoted log conversion function.

    Parameters
    ----------
    value : numeric
        *Linear* value.
    log_reference : numeric
        Log reference.
    linear_reference : numeric
        Linear reference.
    negative_gamma : numeric
        Negative gamma.
    density_per_code_value : numeric
        Density per code value.

    Returns
    -------
    numeric
        *Josh Pines* style pivoted log value.

    Examples
    --------
    >>> linear_to_pivoted_log(0.18)  # doctest: +ELLIPSIS
    0.4349951...
    """

    return ((log_reference + np.log10(value / linear_reference) /
             (density_per_code_value / negative_gamma)) / 1023)


def pivoted_log_to_linear(value,
                          log_reference=445,
                          linear_reference=0.18,
                          negative_gamma=0.6,
                          density_per_code_value=0.002):
    """
    Defines the *Josh Pines* style pivoted log to *linear* conversion function.

    Parameters
    ----------
    value : numeric
        *Josh Pines* style pivoted log value.
    log_reference : numeric
        Log reference.
    linear_reference : numeric
        Linear reference.
    negative_gamma : numeric
        Negative gamma.
    density_per_code_value : numeric
        Density per code value.

    Returns
    -------
    numeric
        *Linear* value.

    Examples
    --------
    >>> pivoted_log_to_linear(0.43499511241446726)  # doctest: +ELLIPSIS
    0.1...
    """

    return (10 ** ((value * 1023 - log_reference) *
                   (density_per_code_value / negative_gamma)) *
            linear_reference)


def linear_to_c_log(value, **kwargs):
    """
    Defines the *linear* to *Canon Log* conversion function.

    Parameters
    ----------
    value : numeric
        *Linear* value.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *Canon Log* value.

    References
    ----------
    .. [2]  Thorpe, L. (2012). CANON-LOG TRANSFER CHARACTERISTIC. Retrieved
            from http://downloads.canon.com/CDLC/Canon-Log_Transfer_Characteristic_6-20-2012.pdf  # noqa

    Examples
    --------
    >>> linear_to_c_log(0.20) * 100  # doctest: +ELLIPSIS
    32.7953896...
    """

    return 0.529136 * np.log10(10.1596 * value + 1) + 0.0730597


def c_log_to_linear(value, **kwargs):
    """
    Defines the *Canon Log* to *linear* conversion function. [2]_

    Parameters
    ----------
    value : numeric
        *Canon Log* value.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric
        *Linear* value.

    Examples
    --------
    >>> c_log_to_linear(32.795389693580908 / 100)  # doctest: +ELLIPSIS
    0.19999999...
    """

    return -0.071622555735168 * (
        1.3742747797867 - np.exp(1) ** (4.3515940948906 * value))


linear_to_aces_cc = (
    lambda x, **kwargs: ACES_CC_TRANSFER_FUNCTION(x))
aces_cc_to_linear = (
    lambda x, **kwargs: ACES_CC_INVERSE_TRANSFER_FUNCTION(x))

linear_to_alexa_log_c = (
    lambda x, **kwargs: ALEXA_WIDE_GAMUT_RGB_TRANSFER_FUNCTION(x))
alexa_log_c_to_linear = (
    lambda x, **kwargs: ALEXA_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION(x))

linear_to_dci_p3_log = (
    lambda x, **kwargs: DCI_P3_TRANSFER_FUNCTION(x))
dci_p3_log_to_linear = (
    lambda x, **kwargs: DCI_P3_INVERSE_TRANSFER_FUNCTION(x))

linear_to_s_log = lambda x, **kwargs: S_LOG_TRANSFER_FUNCTION(x)
s_log_to_linear = lambda x, **kwargs: S_LOG_INVERSE_TRANSFER_FUNCTION(x)
linear_to_s_log2 = lambda x, **kwargs: S_LOG2_TRANSFER_FUNCTION(x)
s_log2_to_linear = lambda x, **kwargs: S_LOG2_INVERSE_TRANSFER_FUNCTION(x)
linear_to_s_log3 = lambda x, **kwargs: S_LOG3_TRANSFER_FUNCTION(x)
s_log3_to_linear = lambda x, **kwargs: S_LOG3_INVERSE_TRANSFER_FUNCTION(x)

LINEAR_TO_LOG_METHODS = CaseInsensitiveMapping(
    {'Cineon': linear_to_cineon,
     'Panalog': linear_to_panalog,
     'REDLog': linear_to_red_log,
     'ViperLog': linear_to_viper_log,
     'PLog': linear_to_pivoted_log,
     'C-Log': linear_to_c_log,
     'ACEScc': linear_to_aces_cc,
     'ALEXA Log C': linear_to_alexa_log_c,
     'DCI-P3': linear_to_dci_p3_log,
     'S-Log': linear_to_s_log,
     'S-Log2': linear_to_s_log2,
     'S-Log3': linear_to_s_log3})
"""
Supported *linear* to *log* computations methods.

LINEAR_TO_LOG_METHODS : CaseInsensitiveMapping
    {'Cineon', 'Panalog', 'REDLog', 'ViperLog', 'PLog', 'C-Log',
    'ACEScc', 'ALEXA Log C', 'DCI-P3', 'S-Log', 'S-Log2', 'S-Log3'}
"""


def linear_to_log(value, method='Cineon', **kwargs):
    """
    Converts from *linear* to *log* using given method.

    Parameters
    ----------
    value : numeric
        Value.
    method : unicode, optional
        {'Cineon', 'Panalog', 'REDLog', 'ViperLog', 'PLog', 'C-Log',
        'ACEScc', 'ALEXA Log C', 'DCI-P3', 'S-Log', 'S-Log2', 'S-Log3'},
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    numeric
        *Log* value.

    Examples
    --------
    >>> linear_to_log(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    >>> linear_to_log(0.18, method='ACEScc')  # doctest: +ELLIPSIS
    0.4135884...
    >>> linear_to_log(0.18, method='PLog', log_reference=400)  # noqa # doctest: +ELLIPSIS
    0.3910068...
    >>> linear_to_log(0.18, method='S-Log')  # doctest: +ELLIPSIS
    0.3599878...
    """

    return LINEAR_TO_LOG_METHODS.get(method)(value, **kwargs)


LOG_TO_LINEAR_METHODS = CaseInsensitiveMapping(
    {'Cineon': cineon_to_linear,
     'Panalog': panalog_to_linear,
     'REDLog': red_log_to_linear,
     'ViperLog': viper_log_to_linear,
     'PLog': pivoted_log_to_linear,
     'C-Log': c_log_to_linear,
     'ACEScc': aces_cc_to_linear,
     'ALEXA Log C': alexa_log_c_to_linear,
     'DCI-P3': dci_p3_log_to_linear,
     'S-Log': s_log_to_linear,
     'S-Log2': s_log2_to_linear,
     'S-Log3': s_log3_to_linear})
"""
Supported *log* to *linear* computations methods.

LOG_TO_LINEAR_METHODS : CaseInsensitiveMapping
    {'Cineon', 'Panalog', 'REDLog', 'ViperLog', 'PLog', 'C-Log',
    'ACEScc', 'ALEXA Log C', 'DCI-P3', 'S-Log', 'S-Log2', 'S-Log3'}
"""


def log_to_linear(value, method='Cineon', **kwargs):
    """
    Converts from *log* to *linear* using given method.

    Parameters
    ----------
    value : numeric
        Value.
    method : unicode, optional
        {'Cineon', 'Panalog', 'REDLog', 'ViperLog', 'PLog', 'C-Log',
        'ACEScc', 'ALEXA Log C', 'DCI-P3', 'S-Log', 'S-Log2', 'S-Log3'},
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    numeric
        *Log* value.

    Examples
    --------
    >>> log_to_linear(0.45731961308541841)  # doctest: +ELLIPSIS
    0.18...
    >>> log_to_linear(0.41358840249244228, method='ACEScc')  # noqa # doctest: +ELLIPSIS
    0.18...
    >>> log_to_linear(0.39100684261974583, method='PLog', log_reference=400)  # noqa # doctest: +ELLIPSIS
    0.1...
    >>> log_to_linear(0.35998784642215442, method='S-Log')  # noqa # doctest: +ELLIPSIS
    0.1799999...
    """

    return LOG_TO_LINEAR_METHODS.get(method)(value, **kwargs)
