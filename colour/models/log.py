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
-   :attr:`linear_to_red_log_film`
-   :attr:`red_log_film_to_linear`
-   :attr:`linear_to_s_log`
-   :attr:`s_log_to_linear`
-   :attr:`linear_to_s_log2`
-   :attr:`s_log2_to_linear`
-   :attr:`linear_to_s_log3`
-   :attr:`s_log3_to_linear`
-   :attr:`linear_to_v_log`
-   :attr:`v_log_to_linear`

See Also
--------
`Log Conversion IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/log.ipynb>`_

References
----------
.. [1]  Sony Imageworks. (2012). make.py. Retrieved November 27, 2014, from
        https://github.com/imageworks/OpenColorIO-Configs/\
blob/master/nuke-default/make.py
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.dataset.aces import (
    ACES_CC_OECF,
    ACES_CC_EOCF)
from colour.models.dataset.alexa_wide_gamut_rgb import (
    ALEXA_LOG_C_OECF,
    ALEXA_LOG_C_EOCF)
from colour.models.dataset.dci_p3 import (
    DCI_P3_OECF,
    DCI_P3_EOCF)
from colour.models.dataset.red import (
    RED_LOG_FILM_OECF,
    RED_LOG_FILM_EOCF)
from colour.models.dataset.sony import (
    S_LOG_OECF,
    S_LOG2_OECF,
    S_LOG3_OECF,
    S_LOG_EOCF,
    S_LOG2_EOCF,
    S_LOG3_EOCF)
from colour.models.dataset.v_gamut import (
    V_LOG_OECF,
    V_LOG_EOCF)
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
           'linear_to_red_log_film',
           'red_log_film_to_linear',
           'linear_to_s_log',
           's_log_to_linear',
           'linear_to_s_log2',
           's_log2_to_linear',
           'linear_to_s_log3',
           's_log3_to_linear',
           'linear_to_v_log',
           'v_log_to_linear',
           'LINEAR_TO_LOG_METHODS',
           'LOG_TO_LINEAR_METHODS',
           'linear_to_log',
           'log_to_linear']


def linear_to_cineon(value, black_offset=10 ** ((95 - 685) / 300), **kwargs):
    """
    Defines the *linear* to *Cineon* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    black_offset : numeric or array_like
        Black offset.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Cineon* value.

    Examples
    --------
    >>> linear_to_cineon(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    value = np.asarray(value)

    return ((685 + 300 *
             np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def cineon_to_linear(value, black_offset=10 ** ((95 - 685) / 300), **kwargs):
    """
    Defines the *Cineon* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Cineon* value.
    black_offset : numeric or array_like
        Black offset.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> cineon_to_linear(0.45731961308541841)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    return ((10 ** ((1023 * value - 685) / 300) - black_offset) /
            (1 - black_offset))


def linear_to_panalog(value, black_offset=10 ** ((64 - 681) / 444), **kwargs):
    """
    Defines the *linear* to *Panalog* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    black_offset : numeric or array_like
        Black offset.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Panalog* value.

    Examples
    --------
    >>> linear_to_panalog(0.18)  # doctest: +ELLIPSIS
    0.3745767...
    """

    value = np.asarray(value)

    return ((681 + 444 *
             np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def panalog_to_linear(value, black_offset=10 ** ((64 - 681) / 444), **kwargs):
    """
    Defines the *Panalog* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Panalog* value.
    black_offset : numeric or array_like
        Black offset.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> panalog_to_linear(0.37457679138229816)  # doctest: +ELLIPSIS
    0.1...
    """

    value = np.asarray(value)

    return ((10 ** ((1023 * value - 681) / 444) - black_offset) /
            (1 - black_offset))


def linear_to_viper_log(value, **kwargs):
    """
    Defines the *linear* to *ViperLog* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *ViperLog* value.

    Examples
    --------
    >>> linear_to_viper_log(0.18)  # doctest: +ELLIPSIS
    0.6360080...
    """

    value = np.asarray(value)

    return (1023 + 500 * np.log10(value)) / 1023


def viper_log_to_linear(value, **kwargs):
    """
    Defines the *ViperLog* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *ViperLog* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> viper_log_to_linear(0.63600806701041346)  # doctest: +ELLIPSIS
    0.1799999...
    """

    value = np.asarray(value)

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
    value : numeric or array_like
        *Linear* value.
    log_reference : numeric or array_like
        Log reference.
    linear_reference : numeric or array_like
        Linear reference.
    negative_gamma : numeric or array_like
        Negative gamma.
    density_per_code_value : numeric or array_like
        Density per code value.

    Returns
    -------
    numeric or ndarray
        *Josh Pines* style pivoted log value.

    Examples
    --------
    >>> linear_to_pivoted_log(0.18)  # doctest: +ELLIPSIS
    0.4349951...
    """

    value = np.asarray(value)

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
    value : numeric or array_like
        *Josh Pines* style pivoted log value.
    log_reference : numeric or array_like
        Log reference.
    linear_reference : numeric or array_like
        Linear reference.
    negative_gamma : numeric or array_like
        Negative gamma.
    density_per_code_value : numeric or array_like
        Density per code value.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> pivoted_log_to_linear(0.43499511241446726)  # doctest: +ELLIPSIS
    0.1...
    """

    value = np.asarray(value)

    return (10 ** ((value * 1023 - log_reference) *
                   (density_per_code_value / negative_gamma)) *
            linear_reference)


def linear_to_c_log(value, **kwargs):
    """
    Defines the *linear* to *Canon Log* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Canon Log* value.

    References
    ----------
    .. [2]  Thorpe, L. (2012). CANON-LOG TRANSFER CHARACTERISTIC. Retrieved
            from http://downloads.canon.com/CDLC/\
Canon-Log_Transfer_Characteristic_6-20-2012.pdf

    Examples
    --------
    >>> linear_to_c_log(0.20) * 100  # doctest: +ELLIPSIS
    32.7953896...
    """

    value = np.asarray(value)

    return 0.529136 * np.log10(10.1596 * value + 1) + 0.0730597


def c_log_to_linear(value, **kwargs):
    """
    Defines the *Canon Log* to *linear* conversion function. [2]_

    Parameters
    ----------
    value : numeric or array_like
        *Canon Log* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> c_log_to_linear(32.795389693580908 / 100)  # doctest: +ELLIPSIS
    0.19999999...
    """

    value = np.asarray(value)

    return (-0.071622555735168 *
            (1.3742747797867 - np.exp(1) ** (4.3515940948906 * value)))


def linear_to_aces_cc(value, **kwargs):
    """
    Defines the *linear* to *ACEScc* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *ACEScc* value.

    Examples
    --------
    >>> linear_to_aces_cc(0.18)  # doctest: +ELLIPSIS
    array(0.4135884...)
    """

    return ACES_CC_OECF(value)


def aces_cc_to_linear(value, **kwargs):
    """
    Defines the *ACEScc* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *ACEScc* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> aces_cc_to_linear(0.41358840249244228)  # doctest: +ELLIPSIS
    array(0.1800000...)
    """

    return ACES_CC_EOCF(value)


def linear_to_alexa_log_c(value, **kwargs):
    """
    Defines the *linear* to *ALEXA Log C* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *ALEXA Log C* value.

    Examples
    --------
    >>> linear_to_alexa_log_c(0.18)  # doctest: +ELLIPSIS
    array(0.3910068...)
    """

    return ALEXA_LOG_C_OECF(value)


def alexa_log_c_to_linear(value, **kwargs):
    """
    Defines the *ALEXA Log C* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *ALEXA Log C* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> alexa_log_c_to_linear(0.39100683203408376)  # doctest: +ELLIPSIS
    array(0.1800000...)
    """

    return ALEXA_LOG_C_EOCF(value)


def linear_to_dci_p3_log(value, **kwargs):
    """
    Defines the *linear* to *DCI-P3* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *DCI-P3* value.

    Examples
    --------
    >>> linear_to_dci_p3_log(0.18)  # doctest: +ELLIPSIS
    461.9922059...
    """

    return DCI_P3_OECF(value)


def dci_p3_log_to_linear(value, **kwargs):
    """
    Defines the *DCI-P3* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *DCI-P3* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> dci_p3_log_to_linear(461.99220597484737)  # doctest: +ELLIPSIS
    0.1800000...
    """

    return DCI_P3_EOCF(value)


def linear_to_red_log_film(value,
                           black_offset=10 ** ((0 - 1023) / 511),
                           **kwargs):
    """
    Defines the *linear* to *REDLogFilm* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    black_offset : numeric or array_like
        Black offset.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *REDLogFilm* value.

    Examples
    --------
    >>> linear_to_red_log_film(0.18)  # doctest: +ELLIPSIS
    0.6376218...
    """

    return RED_LOG_FILM_OECF(value, black_offset)


def red_log_film_to_linear(value,
                           black_offset=10 ** ((0 - 1023) / 511),
                           **kwargs):
    """
    Defines the *REDLogFilm* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *REDLogFilm* value.
    black_offset : numeric or array_like
        Black offset.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> red_log_film_to_linear(0.63762184598817484)  # doctest: +ELLIPSIS
    0.1...
    """

    return RED_LOG_FILM_EOCF(value, black_offset)


def linear_to_s_log(value, **kwargs):
    """
    Defines the *linear* to *S-Log* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *S-Log* value.

    Examples
    --------
    >>> linear_to_s_log(0.18)  # doctest: +ELLIPSIS
    0.3599878...
    """

    return S_LOG_OECF(value)


def s_log_to_linear(value, **kwargs):
    """
    Defines the *S-Log* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *S-Log* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> s_log_to_linear(0.35998784642215442)  # doctest: +ELLIPSIS
    0.1...
    """

    return S_LOG_EOCF(value)


def linear_to_s_log2(value, **kwargs):
    """
    Defines the *linear* to *S-Log2* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *S-Log2* value.

    Examples
    --------
    >>> linear_to_s_log2(0.18)  # doctest: +ELLIPSIS
    0.3849708...
    """

    return S_LOG2_OECF(value)


def s_log2_to_linear(value, **kwargs):
    """
    Defines the *S-Log2* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *S-Log2* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> s_log2_to_linear(0.38497081592867027)  # doctest: +ELLIPSIS
    0.1...
    """

    return S_LOG2_EOCF(value)


def linear_to_s_log3(value, **kwargs):
    """
    Defines the *linear* to *S-Log3* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *S-Log3* value.

    Examples
    --------
    >>> linear_to_s_log3(0.18)  # doctest: +ELLIPSIS
    array(0.4105571...)
    """

    return S_LOG3_OECF(value)


def s_log3_to_linear(value, **kwargs):
    """
    Defines the *S-Log3* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *S-Log3* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> s_log3_to_linear(0.41055718475073316)  # doctest: +ELLIPSIS
    array(0.1...)
    """

    return S_LOG3_EOCF(value)


def linear_to_v_log(value, **kwargs):
    """
    Defines the *linear* to *V-Log* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *Linear* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *V-Log* value.

    Examples
    --------
    >>> linear_to_v_log(0.18)  # doctest: +ELLIPSIS
    array(0.4233114...)
    """

    return V_LOG_OECF(value)


def v_log_to_linear(value, **kwargs):
    """
    Defines the *V-Log* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        *V-Log* value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *linear* / *log* conversion objects.

    Returns
    -------
    numeric or ndarray
        *Linear* value.

    Examples
    --------
    >>> v_log_to_linear(0.42331144876013616)  # doctest: +ELLIPSIS
    array(0.1...)
    """

    return V_LOG_EOCF(value)


LINEAR_TO_LOG_METHODS = CaseInsensitiveMapping(
    {'Cineon': linear_to_cineon,
     'Panalog': linear_to_panalog,
     'ViperLog': linear_to_viper_log,
     'PLog': linear_to_pivoted_log,
     'C-Log': linear_to_c_log,
     'ACEScc': linear_to_aces_cc,
     'ALEXA Log C': linear_to_alexa_log_c,
     'DCI-P3': linear_to_dci_p3_log,
     'REDLogFilm': linear_to_red_log_film,
     'S-Log': linear_to_s_log,
     'S-Log2': linear_to_s_log2,
     'S-Log3': linear_to_s_log3,
     'V-Log': linear_to_v_log})
"""
Supported *linear* to *log* computations methods.

LINEAR_TO_LOG_METHODS : CaseInsensitiveMapping
    **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc',
    'ALEXA Log C', 'DCI-P3', 'REDLogFilm', 'S-Log', 'S-Log2', 'S-Log3',
    'V-Log'}**
"""


def linear_to_log(value, method='Cineon', **kwargs):
    """
    Converts from *linear* to *log* using given method.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    method : unicode, optional
        **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc',
        'ALEXA Log C', 'REDLogFilm', 'DCI-P3', 'S-Log', 'S-Log2', 'S-Log3',
        'V-Log'}**,
        Computation method.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    numeric or ndarray
        *Log* value.

    Examples
    --------
    >>> linear_to_log(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    >>> linear_to_log(0.18, method='ACEScc')  # doctest: +ELLIPSIS
    array(0.4135884...)
    >>> linear_to_log(  # doctest: +ELLIPSIS
    ...     0.18, method='PLog', log_reference=400)
    0.3910068...
    >>> linear_to_log(0.18, method='S-Log')  # doctest: +ELLIPSIS
    0.3599878...
    """

    return LINEAR_TO_LOG_METHODS.get(method)(value, **kwargs)


LOG_TO_LINEAR_METHODS = CaseInsensitiveMapping(
    {'Cineon': cineon_to_linear,
     'Panalog': panalog_to_linear,
     'ViperLog': viper_log_to_linear,
     'PLog': pivoted_log_to_linear,
     'C-Log': c_log_to_linear,
     'ACEScc': aces_cc_to_linear,
     'ALEXA Log C': alexa_log_c_to_linear,
     'DCI-P3': dci_p3_log_to_linear,
     'REDLogFilm': red_log_film_to_linear,
     'S-Log': s_log_to_linear,
     'S-Log2': s_log2_to_linear,
     'S-Log3': s_log3_to_linear,
     'V-Log': v_log_to_linear})
"""
Supported *log* to *linear* computations methods.

LOG_TO_LINEAR_METHODS : CaseInsensitiveMapping
    **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc',
    'ALEXA Log C', 'DCI-P3', 'REDLogFilm', 'S-Log', 'S-Log2', 'S-Log3',
    'V-Log'}**
"""


def log_to_linear(value, method='Cineon', **kwargs):
    """
    Converts from *log* to *linear* using given method.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    method : unicode, optional
        **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc',
        'ALEXA Log C', 'DCI-P3', 'REDLogFilm', 'S-Log', 'S-Log2', 'S-Log3',
        'V-Log'}**,
        Computation method.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    numeric or ndarray
        *Log* value.

    Examples
    --------
    >>> log_to_linear(0.45731961308541841)  # doctest: +ELLIPSIS
    0.18...
    >>> log_to_linear(0.41358840249244228,
    ...     method='ACEScc')  # doctest: +ELLIPSIS
    array(0.18...)
    >>> log_to_linear(  # doctest: +ELLIPSIS
    ...     0.39100684261974583, method='PLog', log_reference=400)
    0.1...
    >>> log_to_linear(  # doctest: +ELLIPSIS
    ...     0.35998784642215442, method='S-Log')
    0.1799999...
    """

    return LOG_TO_LINEAR_METHODS.get(method)(value, **kwargs)
