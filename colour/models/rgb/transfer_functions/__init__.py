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
from .bt_709 import oetf_BT709, eotf_BT709
from .bt_1886 import oetf_BT1886, eotf_BT1886
from .bt_2020 import oetf_BT2020, eotf_BT2020
from .canon_clog import log_encoding_CLog, log_decoding_CLog
from .cineon import log_encoding_Cineon, log_decoding_Cineon
from .dci_p3 import oetf_DCIP3, eotf_DCIP3
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
from .srgb import oetf_sRGB, eotf_sRGB
from .sony_slog import (
    log_encoding_SLog,
    log_decoding_SLog,
    log_encoding_SLog2,
    log_decoding_SLog2,
    log_encoding_SLog3,
    log_decoding_SLog3)
from .st_2084 import oetf_ST2084, eotf_ST2084
from .viper_log import log_encoding_ViperLog, log_decoding_ViperLog

__all__ = ['log_encoding_ACESproxy',
           'log_decoding_ACESproxy',
           'log_encoding_ACEScc',
           'log_decoding_ACEScc']
__all__ += ['log_encoding_ALEXALogC', 'log_decoding_ALEXALogC']
__all__ += ['oetf_BT709', 'eotf_BT709']
__all__ += ['oetf_BT1886', 'eotf_BT1886']
__all__ += ['oetf_BT2020', 'eotf_BT2020']
__all__ += ['log_encoding_CLog', 'log_decoding_CLog']
__all__ += ['log_encoding_Cineon', 'log_decoding_Cineon']
__all__ += ['oetf_DCIP3', 'eotf_DCIP3']
__all__ += ['gamma_function']
__all__ += ['linear_function']
__all__ += ['log_encoding_Panalog', 'log_decoding_Panalog']
__all__ += ['log_encoding_VLog', 'log_decoding_VLog']
__all__ += ['log_encoding_PivotedLog', 'log_decoding_PivotedLog']
__all__ += ['log_encoding_ProPhotoRGB', 'log_decoding_ProPhotoRGB']
__all__ += ['log_encoding_REDLog', 'log_decoding_REDLog']
__all__ += ['log_encoding_SLog',
            'log_decoding_SLog',
            'log_encoding_SLog2',
            'log_decoding_SLog2',
            'log_encoding_SLog3',
            'log_decoding_SLog3']
__all__ += ['oetf_sRGB', 'eotf_sRGB']
__all__ += ['oetf_ST2084', 'eotf_ST2084']
__all__ += ['log_decoding_ViperLog', 'log_decoding_ViperLog']

LOG_ENCODING_CURVES = CaseInsensitiveMapping(
    {'Cineon': log_encoding_Cineon,
     'Panalog': log_encoding_Panalog,
     'ViperLog': log_encoding_ViperLog,
     'PLog': log_encoding_PivotedLog,
     'C-Log': log_encoding_CLog,
     'ACEScc': log_encoding_ACEScc,
     'ACESproxy': log_encoding_ACESproxy,
     'ALEXA Log C': log_encoding_ALEXALogC,
     'REDLog': log_encoding_REDLog,
     'REDLogFilm': log_encoding_REDLogFilm,
     'S-Log': log_encoding_SLog,
     'S-Log2': log_encoding_SLog2,
     'S-Log3': log_encoding_SLog3,
     'V-Log': log_encoding_VLog})
"""
Supported *log* encoding curves.

LOG_ENCODING_CURVES : CaseInsensitiveMapping
    **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc', 'ACESproxy'
    'ALEXA Log C', 'REDLog', 'REDLogFilm', 'S-Log', 'S-Log2', 'S-Log3',
    'V-Log'}**
"""


def log_encoding_curve(value, curve='Cineon', **kwargs):
    """
    Encodes linear-light values to :math:`R'G'B'` video component signal
    value using given *log* curve.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    curve : unicode, optional
        **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc',
        'ACESproxy', 'ALEXA Log C', 'REDLog', 'REDLogFilm', 'S-Log', 'S-Log2',
        'S-Log3', 'V-Log'}**,
        Computation curve.
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
    >>> log_encoding_curve(0.18, curve='ACEScc')  # doctest: +ELLIPSIS
    0.4135884...
    >>> log_encoding_curve(  # doctest: +ELLIPSIS
    ...     0.18, curve='PLog', log_reference=400)
    0.3910068...
    >>> log_encoding_curve(0.18, curve='S-Log')  # doctest: +ELLIPSIS
    0.3599878...
    """

    function = LOG_ENCODING_CURVES[curve]

    filter_kwargs(function, **kwargs)

    return function(value, **kwargs)


LOG_DECODING_CURVES = CaseInsensitiveMapping(
    {'Cineon': log_decoding_Cineon,
     'Panalog': log_decoding_Panalog,
     'ViperLog': log_decoding_ViperLog,
     'PLog': log_decoding_PivotedLog,
     'C-Log': log_decoding_CLog,
     'ACEScc': log_decoding_ACEScc,
     'ACESproxy': log_decoding_ACESproxy,
     'ALEXA Log C': log_decoding_ALEXALogC,
     'REDLog': log_decoding_REDLog,
     'REDLogFilm': log_decoding_REDLogFilm,
     'S-Log': log_decoding_SLog,
     'S-Log2': log_decoding_SLog2,
     'S-Log3': log_decoding_SLog3,
     'V-Log': log_decoding_VLog})
"""
Supported *log* decoding curves.

LOG_DECODING_CURVES : CaseInsensitiveMapping
    **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc', 'ACESproxy'
    'ALEXA Log C', 'REDLog', 'REDLogFilm', 'S-Log', 'S-Log2', 'S-Log3',
    'V-Log'}**
"""


def log_decoding_curve(value, curve='Cineon', **kwargs):
    """
    Decodes :math:`R'G'B'` video component signal value to linear-light values
    using given *log* curve.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    curve : unicode, optional
        **{'Cineon', 'Panalog', 'ViperLog', 'PLog', 'C-Log', 'ACEScc',
        'ACESproxy', 'ALEXA Log C', 'REDLog', 'REDLogFilm', 'S-Log', 'S-Log2',
        'S-Log3', 'V-Log'}**,
        Computation curve.
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
    ...     curve='ACEScc')  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_curve(  # doctest: +ELLIPSIS
    ...     0.39100684261974583, curve='PLog', log_reference=400)
    0.1...
    >>> log_decoding_curve(  # doctest: +ELLIPSIS
    ...     0.35998784642215442, curve='S-Log')
    0.1...
    """

    function = LOG_DECODING_CURVES[curve]

    filter_kwargs(function, **kwargs)

    return function(value, **kwargs)


__all__ += ['LOG_ENCODING_CURVES', 'LOG_DECODING_CURVES']
__all__ += ['log_encoding_curve', 'log_decoding_curve']

OETFS = CaseInsensitiveMapping(
    {'BT.1886': oetf_BT1886,
     'BT.2020': oetf_BT2020,
     'BT.709': oetf_BT709,
     'DCI-P3': oetf_DCIP3,
     'ST 2084': oetf_ST2084,
     'sRGB': oetf_sRGB})
"""
Supported electro-optical transfer functions OETF (OECF).

OETFS : CaseInsensitiveMapping
    **{'sRGB', 'BT.1886', 'BT.2020', 'BT.709', 'DCI-P3', 'ST 2084'}**
"""


def oetf(value, function='sRGB', **kwargs):
    """
    Encodes estimated tristimulus values in a scene to :math:`R'G'B'` video
    component signal value using given OETF (OECF) opto-electronic transfer
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    function : unicode, optional
        **{'sRGB', 'BT.1886', 'BT.2020', 'BT.709', 'DCI-P3', 'ST 2084'}**,
        Computation function.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    numeric or ndarray
        :math:`R'G'B'` video component signal value.

    Examples
    --------
    >>> oetf(0.18)  # doctest: +ELLIPSIS
    0.4613561...
    >>> oetf(0.18, function='BT.2020')  # doctest: +ELLIPSIS
    0.4090077...
    >>> oetf(  # doctest: +ELLIPSIS
    ...     0.18, function='ST 2084', L_p=1000)
    0.1820115...
    """

    function = OETFS[function]

    filter_kwargs(function, **kwargs)

    return function(value, **kwargs)


EOTFS = CaseInsensitiveMapping(
    {'BT.1886': eotf_BT1886,
     'BT.2020': eotf_BT2020,
     'BT.709': eotf_BT709,
     'DCI-P3': eotf_DCIP3,
     'ST 2084': eotf_ST2084,
     'sRGB': eotf_sRGB})
"""
Supported opto-electrical transfer functions EOTF (EOCF).

EOTFS : CaseInsensitiveMapping
    **{'sRGB', 'BT.1886', 'BT.2020', 'BT.709', 'DCI-P3', 'ST 2084'}**
"""


def eotf(value, function='sRGB', **kwargs):
    """
    Decodes :math:`R'G'B'` video component signal value to tristimulus values
    at the display using given EOTF (EOCF) electro-optical transfer function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    function : unicode, optional
        **{'sRGB', 'BT.1886', 'BT.2020', 'BT.709', 'DCI-P3', 'ST 2084'}**,
        Computation function.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    numeric or ndarray
        Tristimulus values at the display.

    Examples
    --------
    >>> eotf(0.46135612950044164)  # doctest: +ELLIPSIS
    0.1...
    >>> eotf(0.4090077288641504,
    ...     function='BT.2020')  # doctest: +ELLIPSIS
    0.1...
    >>> eotf(  # doctest: +ELLIPSIS
    ...     0.18201153285000843, function='ST 2084', L_p=1000)
    0.1...
    """

    function = EOTFS[function]

    filter_kwargs(function, **kwargs)

    return function(value, **kwargs)


__all__ += ['OETFS', 'EOTFS']
__all__ += ['oetf', 'eotf']
