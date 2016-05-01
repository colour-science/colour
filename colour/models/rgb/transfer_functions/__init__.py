#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from colour.utilities import CaseInsensitiveMapping, filter_kwargs

from .aces import (
    log_encoding_ACESproxy,
    log_decoding_ACESproxy,
    log_encoding_ACEScc,
    log_decoding_ACEScc)
from .alexa_log_c import (
    log_encoding_ALEXALogC,
    log_decoding_ALEXALogC)
from .bt_709 import oecf_BT709, eocf_BT709
from .bt_1886 import eocf_BT1886
from .bt_2020 import oecf_BT2020, eocf_BT2020
from .canon_clog import log_encoding_CLog, log_decoding_CLog
from .cineon import log_encoding_Cineon, log_decoding_Cineon
from .dci_p3 import log_encoding_DCIP3, log_decoding_DCIP3
from .gamma import gamma_function
from .linear import linear_function
from .panalog import log_encoding_Panalog, log_decoding_Panalog
from .panasonic_vlog import log_encoding_VLog, log_decoding_VLog
from .pivoted_log import log_encoding_PivotedLog, log_decoding_PivotedLog
from .prophoto_rgb import log_encoding_ProPhotoRGB, log_decoding_ProPhotoRGB
from .red_log import (
    log_encoding_REDLog,
    log_decoding_REDLog,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm)
from .srgb import oecf_sRGB, eocf_sRGB
from .sony_slog import (
    log_encoding_SLog,
    log_decoding_SLog,
    log_encoding_SLog2,
    log_decoding_SLog2,
    log_encoding_SLog3,
    log_decoding_SLog3)
from .st_2084 import oecf_ST2084, eocf_ST2084
from .viper_log import log_encoding_ViperLog, log_decoding_ViperLog

__all__ = ['log_encoding_ACESproxy',
           'log_decoding_ACESproxy',
           'log_encoding_ACEScc',
           'log_decoding_ACEScc']
__all__ += ['log_encoding_ALEXALogC', 'log_decoding_ALEXALogC']
__all__ += ['oecf_BT709', 'eocf_BT709']
__all__ += ['eocf_BT1886']
__all__ += ['oecf_BT2020', 'eocf_BT2020']
__all__ += ['log_encoding_CLog', 'log_decoding_CLog']
__all__ += ['log_encoding_Cineon', 'log_decoding_Cineon']
__all__ += ['log_encoding_DCIP3', 'log_decoding_DCIP3']
__all__ += ['gamma_function']
__all__ += ['linear_function']
__all__ += ['log_encoding_Panalog', 'log_decoding_Panalog']
__all__ += ['log_decoding_VLog', 'log_decoding_VLog']
__all__ += ['log_encoding_PivotedLog', 'log_decoding_PivotedLog']
__all__ += ['log_encoding_ProPhotoRGB', 'log_decoding_ProPhotoRGB']
__all__ += ['log_encoding_REDLog', 'log_decoding_REDLog']
__all__ += ['log_encoding_SLog',
            'log_decoding_SLog',
            'log_encoding_SLog2',
            'log_decoding_SLog2',
            'log_encoding_SLog3',
            'log_decoding_SLog3']
__all__ += ['oecf_sRGB', 'eocf_sRGB']
__all__ += ['oecf_ST2084', 'eocf_ST2084']
__all__ += ['log_decoding_ViperLog', 'log_decoding_ViperLog']

LOG_ENCODING_METHODS = CaseInsensitiveMapping(
    {'Cineon': log_encoding_Cineon,
     'Panalog': log_encoding_Panalog,
     'ViperLog': log_encoding_ViperLog,
     'PLog': log_encoding_PivotedLog,
     'C-Log': log_encoding_CLog,
     'ACEScc': log_encoding_ACEScc,
     'ACESproxy': log_encoding_ACESproxy,
     'ALEXA Log C': log_encoding_ALEXALogC,
     'DCI-P3': log_encoding_DCIP3,
     'REDLog': log_encoding_REDLog,
     'REDLogFilm': log_encoding_REDLogFilm,
     'S-Log': log_encoding_SLog,
     'S-Log2': log_encoding_SLog2,
     'S-Log3': log_encoding_SLog3,
     'V-Log': log_encoding_VLog})
"""
Supported *log* encoding computations methods.

LOG_ENCODING_METHODS : CaseInsensitiveMapping
    **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc', 'ACESproxy'
    'ALEXA Log C', 'DCI-P3', 'REDLog', 'REDLogFilm', 'S-Log', 'S-Log2',
    'S-Log3', 'V-Log'}**
"""


def log_encoding_curve(value, method='Cineon', **kwargs):
    """
    Encodes *log* values using given curve / method.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    method : unicode, optional
        **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc',
        'ACESproxy', 'ALEXA Log C', 'REDLog', 'REDLogFilm', 'DCI-P3', 'S-Log',
        'S-Log2', 'S-Log3', 'V-Log'}**,
        Computation method.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    numeric or ndarray
        *Log* value.

    Examples
    --------
    >>> log_encoding_curve(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    >>> log_encoding_curve(0.18, method='ACEScc')  # doctest: +ELLIPSIS
    0.4135884...
    >>> log_encoding_curve(  # doctest: +ELLIPSIS
    ...     0.18, method='PLog', log_reference=400)
    0.3910068...
    >>> log_encoding_curve(0.18, method='S-Log')  # doctest: +ELLIPSIS
    0.3599878...
    """

    func = LOG_ENCODING_METHODS[method]

    filter_kwargs(func, **kwargs)

    return func(value, **kwargs)


LOG_DECODING_METHODS = CaseInsensitiveMapping(
    {'Cineon': log_decoding_Cineon,
     'Panalog': log_decoding_Panalog,
     'ViperLog': log_decoding_ViperLog,
     'PLog': log_decoding_PivotedLog,
     'C-Log': log_decoding_CLog,
     'ACEScc': log_decoding_ACEScc,
     'ACESproxy': log_decoding_ACESproxy,
     'ALEXA Log C': log_decoding_ALEXALogC,
     'DCI-P3': log_decoding_DCIP3,
     'REDLog': log_decoding_REDLog,
     'REDLogFilm': log_decoding_REDLogFilm,
     'S-Log': log_decoding_SLog,
     'S-Log2': log_decoding_SLog2,
     'S-Log3': log_decoding_SLog3,
     'V-Log': log_decoding_VLog})
"""
Supported *log* decoding computations methods.

LOG_DECODING_METHODS : CaseInsensitiveMapping
    **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc', 'ACESproxy'
    'ALEXA Log C', 'DCI-P3', 'REDLog', 'REDLogFilm', 'S-Log', 'S-Log2',
    'S-Log3', 'V-Log'}**
"""


def log_decoding_curve(value, method='Cineon', **kwargs):
    """
    Decodes *log* values using given curve / method.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    method : unicode, optional
        **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc',
        'ACESproxy', 'ALEXA Log C', 'REDLog', 'REDLogFilm', 'DCI-P3', 'S-Log',
        'S-Log2', 'S-Log3', 'V-Log'}**,
        Computation method.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    numeric or ndarray
        *Log* value.

    Examples
    --------
    >>> log_decoding_curve(0.45731961308541841)  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_curve(0.41358840249244228,
    ...     method='ACEScc')  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_curve(  # doctest: +ELLIPSIS
    ...     0.39100684261974583, method='PLog', log_reference=400)
    0.1...
    >>> log_decoding_curve(  # doctest: +ELLIPSIS
    ...     0.35998784642215442, method='S-Log')
    0.1...
    """

    func = LOG_DECODING_METHODS[method]

    filter_kwargs(func, **kwargs)

    return func(value, **kwargs)

__all__ += ['LOG_ENCODING_METHODS', 'LOG_DECODING_METHODS']
__all__ += ['log_encoding_curve', 'log_decoding_curve']
