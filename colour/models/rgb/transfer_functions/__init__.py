# -*- coding: utf-8 -*-

from __future__ import absolute_import

from colour.utilities import CaseInsensitiveMapping, filter_kwargs

from .common import CV_range, legal_to_full, full_to_legal
from .aces import (log_encoding_ACESproxy, log_decoding_ACESproxy,
                   log_encoding_ACEScc, log_decoding_ACEScc,
                   log_encoding_ACEScct, log_decoding_ACEScct)
from .alexa_log_c import (log_encoding_ALEXALogC, log_decoding_ALEXALogC)
from .arib_std_b67 import oetf_ARIBSTDB67, oetf_reverse_ARIBSTDB67
from .canon_log import (log_encoding_CanonLog, log_decoding_CanonLog,
                        log_encoding_CanonLog2, log_decoding_CanonLog2,
                        log_encoding_CanonLog3, log_decoding_CanonLog3)
from .cineon import log_encoding_Cineon, log_decoding_Cineon
from .dci_p3 import oetf_DCIP3, eotf_DCIP3
from .dicom_gsdf import oetf_DICOMGSDF, eotf_DICOMGSDF
from .gamma import function_gamma
from .gopro import log_encoding_Protune, log_decoding_Protune
from .itur_bt_601 import oetf_BT601, oetf_reverse_BT601
from .itur_bt_709 import oetf_BT709, oetf_reverse_BT709
from .itur_bt_1886 import eotf_reverse_BT1886, eotf_BT1886
from .itur_bt_2020 import oetf_BT2020, eotf_BT2020
from .st_2084 import oetf_ST2084, eotf_ST2084
from .itur_bt_2100 import (
    oetf_BT2100_PQ, oetf_reverse_BT2100_PQ, eotf_BT2100_PQ,
    eotf_reverse_BT2100_PQ, ootf_BT2100_PQ, ootf_reverse_BT2100_PQ,
    oetf_BT2100_HLG, oetf_reverse_BT2100_HLG, eotf_BT2100_HLG,
    eotf_reverse_BT2100_HLG, ootf_BT2100_HLG, ootf_reverse_BT2100_HLG)
from .linear import function_linear
from .panalog import log_encoding_Panalog, log_decoding_Panalog
from .panasonic_vlog import log_encoding_VLog, log_decoding_VLog
from .pivoted_log import log_encoding_PivotedLog, log_decoding_PivotedLog
from .red_log import (log_encoding_REDLog, log_decoding_REDLog,
                      log_encoding_REDLogFilm, log_decoding_REDLogFilm,
                      log_encoding_Log3G10, log_decoding_Log3G10,
                      log_encoding_Log3G12, log_decoding_Log3G12)
from .rimm_romm_rgb import (oetf_ROMMRGB, eotf_ROMMRGB, oetf_ProPhotoRGB,
                            eotf_ProPhotoRGB, oetf_RIMMRGB, eotf_RIMMRGB,
                            log_encoding_ERIMMRGB, log_decoding_ERIMMRGB)
from .smpte_240m import oetf_SMPTE240M, eotf_SMPTE240M
from .sony_slog import (log_encoding_SLog, log_decoding_SLog,
                        log_encoding_SLog2, log_decoding_SLog2,
                        log_encoding_SLog3, log_decoding_SLog3)
from .srgb import oetf_sRGB, oetf_reverse_sRGB
from .viper_log import log_encoding_ViperLog, log_decoding_ViperLog

__all__ = ['CV_range', 'legal_to_full', 'full_to_legal']
__all__ += [
    'log_encoding_ACESproxy', 'log_decoding_ACESproxy', 'log_encoding_ACEScc',
    'log_decoding_ACEScc', 'log_encoding_ACEScct', 'log_decoding_ACEScct'
]
__all__ += ['log_encoding_ALEXALogC', 'log_decoding_ALEXALogC']
__all__ += ['oetf_ARIBSTDB67', 'oetf_reverse_ARIBSTDB67']
__all__ += [
    'log_encoding_CanonLog', 'log_decoding_CanonLog', 'log_encoding_CanonLog2',
    'log_decoding_CanonLog2', 'log_encoding_CanonLog3',
    'log_decoding_CanonLog3'
]
__all__ += ['log_encoding_Cineon', 'log_decoding_Cineon']
__all__ += ['oetf_DCIP3', 'eotf_DCIP3']
__all__ += ['oetf_DICOMGSDF', 'eotf_DICOMGSDF']
__all__ += ['function_gamma']
__all__ += ['log_encoding_Protune', 'log_decoding_Protune']
__all__ += ['oetf_BT601', 'oetf_reverse_BT601']
__all__ += ['oetf_BT709', 'oetf_reverse_BT709']
__all__ += ['eotf_reverse_BT1886', 'eotf_BT1886']
__all__ += ['oetf_BT2020', 'eotf_BT2020']
__all__ += ['oetf_ST2084', 'eotf_ST2084']
__all__ += [
    'oetf_BT2100_PQ', 'oetf_reverse_BT2100_PQ', 'eotf_BT2100_PQ',
    'eotf_reverse_BT2100_PQ', 'ootf_BT2100_PQ', 'ootf_reverse_BT2100_PQ',
    'oetf_BT2100_HLG', 'oetf_reverse_BT2100_HLG', 'eotf_BT2100_HLG',
    'eotf_reverse_BT2100_HLG', 'ootf_BT2100_HLG', 'ootf_reverse_BT2100_HLG'
]
__all__ += ['function_linear']
__all__ += ['log_encoding_Panalog', 'log_decoding_Panalog']
__all__ += ['log_encoding_VLog', 'log_decoding_VLog']
__all__ += ['log_encoding_PivotedLog', 'log_decoding_PivotedLog']
__all__ += [
    'log_encoding_REDLog', 'log_decoding_REDLog', 'log_encoding_REDLogFilm',
    'log_decoding_REDLogFilm', 'log_encoding_Log3G10', 'log_decoding_Log3G10',
    'log_encoding_Log3G12', 'log_decoding_Log3G12'
]
__all__ += [
    'oetf_ROMMRGB', 'eotf_ROMMRGB', 'oetf_ProPhotoRGB', 'eotf_ProPhotoRGB',
    'oetf_RIMMRGB', 'eotf_RIMMRGB', 'log_encoding_ERIMMRGB',
    'log_decoding_ERIMMRGB'
]
__all__ += ['oetf_SMPTE240M', 'eotf_SMPTE240M']
__all__ += [
    'log_encoding_SLog', 'log_decoding_SLog', 'log_encoding_SLog2',
    'log_decoding_SLog2', 'log_encoding_SLog3', 'log_decoding_SLog3'
]
__all__ += ['oetf_sRGB', 'oetf_reverse_sRGB']
__all__ += ['log_encoding_ViperLog', 'log_decoding_ViperLog']

LOG_ENCODING_CURVES = CaseInsensitiveMapping({
    'ACEScc': log_encoding_ACEScc,
    'ACEScct': log_encoding_ACEScct,
    'ACESproxy': log_encoding_ACESproxy,
    'ALEXA Log C': log_encoding_ALEXALogC,
    'Canon Log 2': log_encoding_CanonLog2,
    'Canon Log 3': log_encoding_CanonLog3,
    'Canon Log': log_encoding_CanonLog,
    'Cineon': log_encoding_Cineon,
    'ERIMM RGB': log_encoding_ERIMMRGB,
    'Log3G10': log_encoding_Log3G10,
    'Log3G12': log_encoding_Log3G12,
    'Panalog': log_encoding_Panalog,
    'PLog': log_encoding_PivotedLog,
    'Protune': log_encoding_Protune,
    'REDLog': log_encoding_REDLog,
    'REDLogFilm': log_encoding_REDLogFilm,
    'S-Log': log_encoding_SLog,
    'S-Log2': log_encoding_SLog2,
    'S-Log3': log_encoding_SLog3,
    'V-Log': log_encoding_VLog,
    'ViperLog': log_encoding_ViperLog
})
LOG_ENCODING_CURVES.__doc__ = """
Supported *log* encoding curves.

LOG_ENCODING_CURVES : CaseInsensitiveMapping
    **{'ACEScc', 'ACEScct', 'ACESproxy', 'ALEXA Log C', 'Canon Log 2',
    'Canon Log 3', 'Canon Log', 'Cineon', 'ERIMM RGB', 'Log3G10', 'Log3G12',
    'Panalog', 'PLog', 'Protune', 'REDLog', 'REDLogFilm', 'S-Log', 'S-Log2',
    'S-Log3', 'V-Log', 'ViperLog'}**
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
        **{'ACEScc', 'ACEScct', 'ACESproxy', 'ALEXA Log C', 'Canon Log 2',
        'Canon Log 3', 'Canon Log', 'Cineon', 'ERIMM RGB', 'Log3G10',
        'Log3G12', 'Panalog', 'PLog', 'Protune', 'REDLog', 'REDLogFilm',
        'S-Log', 'S-Log2', 'S-Log3', 'V-Log', 'ViperLog'}**, Computation curve.

    Other Parameters
    ----------------
    EI : int,  optional
        {:func:`colour.models.log_encoding_ALEXALogC`},
        Ei.
    E_clip : numeric, optional
        {:func:`colour.models.log_encoding_ERIMMRGB`},
        Maximum exposure limit.
    E_min : numeric, optional
        {:func:`colour.models.log_encoding_ERIMMRGB`},
        Minimum exposure limit.
    I_max : numeric, optional
        {:func:`colour.models.log_encoding_ERIMMRGB`},
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    bit_depth : unicode, optional
        {:func:`colour.models.log_encoding_ACESproxy`,
        :func:`colour.models.log_encoding_SLog`,
        :func:`colour.models.log_encoding_SLog2`},
        **{8, 10, 12}**,
        Bit depth used for conversion, *ACESproxy* uses **{10, 12}**.
    black_offset : numeric or array_like
        {:func:`colour.models.log_encoding_Cineon`,
        :func:`colour.models.log_encoding_Panalog`,
        :func:`colour.models.log_encoding_REDLog`,
        :func:`colour.models.log_encoding_REDLogFilm`},
        Black offset.
    density_per_code_value : numeric or array_like
        {:func:`colour.models.log_encoding_PivotedLog`},
        Density per code value.
    firmware : unicode, optional
        {:func:`colour.models.log_encoding_ALEXALogC`},
        **{'SUP 3.x', 'SUP 2.x'}**,
        Alexa firmware version.
    in_reflection : bool, optional
        {:func:`colour.models.log_encoding_SLog`,
        :func:`colour.models.log_encoding_SLog2`},
        Whether the light level :math:`x` to a camera is reflection.
    linear_reference : numeric or array_like
        {:func:`colour.models.log_encoding_PivotedLog`},
        Linear reference.
    log_reference : numeric or array_like
        {:func:`colour.models.log_encoding_PivotedLog`},
        Log reference.
    out_legal : bool, optional
        {:func:`colour.models.log_encoding_SLog`,
        :func:`colour.models.log_encoding_SLog2`,
        :func:`colour.models.log_encoding_SLog3`},
        Whether the non-linear *Sony S-Log*, *Sony S-Log2* or *Sony S-Log3*
        data :math:`y` is encoded in legal range.
    negative_gamma : numeric or array_like
        {:func:`colour.models.log_encoding_PivotedLog`},
        Negative gamma.
    method : unicode, optional
        {:func:`colour.models.log_encoding_ALEXALogC`},
        **{'Linear Scene Exposure Factor', 'Normalised Sensor Signal'}**,
        Conversion method.

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
    >>> log_encoding_curve(0.18, curve='PLog', log_reference=400)
    ... # doctest: +ELLIPSIS
    0.3910068...
    >>> log_encoding_curve(0.18, curve='S-Log')  # doctest: +ELLIPSIS
    0.3849708...
    """

    function = LOG_ENCODING_CURVES[curve]

    return function(value, **filter_kwargs(function, **kwargs))


LOG_DECODING_CURVES = CaseInsensitiveMapping({
    'ACEScc': log_decoding_ACEScc,
    'ACEScct': log_decoding_ACEScct,
    'ACESproxy': log_decoding_ACESproxy,
    'ALEXA Log C': log_decoding_ALEXALogC,
    'Canon Log 2': log_decoding_CanonLog2,
    'Canon Log 3': log_decoding_CanonLog3,
    'Canon Log': log_decoding_CanonLog,
    'Cineon': log_decoding_Cineon,
    'ERIMM RGB': log_decoding_ERIMMRGB,
    'Log3G10': log_decoding_Log3G10,
    'Log3G12': log_decoding_Log3G12,
    'Panalog': log_decoding_Panalog,
    'PLog': log_decoding_PivotedLog,
    'Protune': log_decoding_Protune,
    'REDLog': log_decoding_REDLog,
    'REDLogFilm': log_decoding_REDLogFilm,
    'S-Log': log_decoding_SLog,
    'S-Log2': log_decoding_SLog2,
    'S-Log3': log_decoding_SLog3,
    'V-Log': log_decoding_VLog,
    'ViperLog': log_decoding_ViperLog
})
LOG_DECODING_CURVES.__doc__ = """
Supported *log* decoding curves.

LOG_DECODING_CURVES : CaseInsensitiveMapping
    **{'ACEScc', 'ACEScct', 'ACESproxy', 'ALEXA Log C', 'Canon Log 2',
    'Canon Log 3', 'Canon Log', 'Cineon', 'ERIMM RGB', 'Log3G10', 'Log3G12',
    'Panalog', 'PLog', 'Protune', 'REDLog', 'REDLogFilm', 'S-Log', 'S-Log2',
    'S-Log3', 'V-Log', 'ViperLog'}**
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
        **{'ACEScc', 'ACEScct', 'ACESproxy', 'ALEXA Log C', 'Canon Log 2',
        'Canon Log 3', 'Canon Log', 'Cineon', 'ERIMM RGB', 'Log3G10',
        'Log3G12', 'Panalog', 'PLog', 'Protune', 'REDLog', 'REDLogFilm',
        'S-Log', 'S-Log2', 'S-Log3', 'V-Log', 'ViperLog'}**, Computation curve.

    Other Parameters
    ----------------
    EI : int,  optional
        {:func:`colour.models.log_decoding_ALEXALogC`},
        Ei.
    E_clip : numeric, optional
        {:func:`colour.models.log_decoding_ERIMMRGB`},
        Maximum exposure limit.
    E_min : numeric, optional
        {:func:`colour.models.log_decoding_ERIMMRGB`},
        Minimum exposure limit.
    I_max : numeric, optional
        {:func:`colour.models.log_decoding_ERIMMRGB`},
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    bit_depth : int, optional
        {:func:`colour.models.log_decoding_ACESproxy`,
        :func:`colour.models.log_decoding_SLog`,
        :func:`colour.models.log_decoding_SLog2`},
        **{8, 10, 12}**,
        Bit depth used for conversion, *ACESproxy* uses **{10, 12}**.
    black_offset : numeric or array_like
        {:func:`colour.models.log_decoding_Cineon`,
        :func:`colour.models.log_decoding_Panalog`,
        :func:`colour.models.log_decoding_REDLog`,
        :func:`colour.models.log_decoding_REDLogFilm`},
        Black offset.
    density_per_code_value : numeric or array_like
        {:func:`colour.models.log_decoding_PivotedLog`},
        Density per code value.
    firmware : unicode, optional
        {:func:`colour.models.log_decoding_ALEXALogC`},
        **{'SUP 3.x', 'SUP 2.x'}**,
        Alexa firmware version.
    in_legal : bool, optional
        {:func:`colour.models.log_decoding_SLog`,
        :func:`colour.models.log_decoding_SLog2`,
        :func:`colour.models.log_decoding_SLog3`},
        Whether the non-linear *Sony S-Log*, *Sony S-Log2* or *Sony S-Log3*
        data :math:`y` is encoded in legal range.
    linear_reference : numeric or array_like
        {:func:`colour.models.log_decoding_PivotedLog`},
        Linear reference.
    log_reference : numeric or array_like
        {:func:`colour.models.log_decoding_PivotedLog`},
        Log reference.
    negative_gamma : numeric or array_like
        {:func:`colour.models.log_decoding_PivotedLog`},
        Negative gamma.
    out_reflection : bool, optional
        {:func:`colour.models.log_decoding_SLog`,
        :func:`colour.models.log_decoding_SLog2`},
        Whether the light level :math:`x` to a camera is reflection.
    method : unicode, optional
        {:func:`colour.models.log_decoding_ALEXALogC`},
        **{'Linear Scene Exposure Factor', 'Normalised Sensor Signal'}**,
        Conversion method.

    Returns
    -------
    numeric or ndarray
        *Log* value.

    Examples
    --------
    >>> log_decoding_curve(0.457319613085418)  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_curve(0.413588402492442, curve='ACEScc')
    ... # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_curve(0.391006842619746, curve='PLog', log_reference=400)
    ... # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_curve(0.376512722254600, curve='S-Log')
    ... # doctest: +ELLIPSIS
    0.1...
    """

    function = LOG_DECODING_CURVES[curve]

    return function(value, **filter_kwargs(function, **kwargs))


__all__ += ['LOG_ENCODING_CURVES', 'LOG_DECODING_CURVES']
__all__ += ['log_encoding_curve', 'log_decoding_curve']

OETFS = CaseInsensitiveMapping({
    'ARIB STD-B67': oetf_ARIBSTDB67,
    'DCI-P3': oetf_DCIP3,
    'DICOM GSDF': oetf_DICOMGSDF,
    'ITU-R BT.2020': oetf_BT2020,
    'ITU-R BT.2100 HLG': oetf_BT2100_HLG,
    'ITU-R BT.2100 PQ': oetf_BT2100_PQ,
    'ITU-R BT.601': oetf_BT601,
    'ITU-R BT.709': oetf_BT709,
    'ProPhoto RGB': oetf_ProPhotoRGB,
    'RIMM RGB': oetf_RIMMRGB,
    'ROMM RGB': oetf_ROMMRGB,
    'SMPTE 240M': oetf_SMPTE240M,
    'ST 2084': oetf_ST2084,
    'sRGB': oetf_sRGB
})
OETFS.__doc__ = """
Supported opto-electrical transfer functions (OETFs / OECFs).

OETFS : CaseInsensitiveMapping
    **{'sRGB', 'ARIB STD-B67', 'DCI-P3', 'DICOM GSDF', 'ITU-R BT.2020',
    'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ', 'ITU-R BT.601', 'ITU-R BT.709',
    'ProPhoto RGB', 'RIMM RGB', 'ROMM RGB', 'SMPTE 240M', 'ST 2084'}**
"""


def oetf(value, function='sRGB', **kwargs):
    """
    Encodes estimated tristimulus values in a scene to :math:`R'G'B'` video
    component signal value using given opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    value : numeric or array_like
        Value.
    function : unicode, optional
        **{'sRGB', 'ARIB STD-B67', 'DCI-P3', 'DICOM GSDF', 'ITU-R BT.2020',
        'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ', 'ITU-R BT.601',
        'ITU-R BT.709', 'ProPhoto RGB', 'RIMM RGB', 'ROMM RGB', 'SMPTE 240M',
        'ST 2084'}**,
        Opto-electronic transfer function (OETF / OECF).

    Other Parameters
    ----------------
    E_clip : numeric, optional
        {:func:`colour.models.oetf_RIMMRGB`},
        Maximum exposure level.
    I_max : numeric, optional
        {:func:`colour.models.oetf_ROMMRGB`,
        :func:`colour.models.oetf_RIMMRGB`},
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    L_p : numeric, optional
        {:func:`colour.models.oetf_ST2084`},
        Display peak luminance :math:`cd/m^2`.
    is_12_bits_system : bool
        {:func:`colour.models.oetf_BT2020`},
        *ITU-R BT.2020* *alpha* and *beta* constants are used
        if system is not
        12-bit.
    r : numeric, optional
        {:func:`colour.models.oetf_ARIBSTDB67`},
        Video level corresponding to reference white level.

    Returns
    -------
    numeric or ndarray
        :math:`R'G'B'` video component signal value.

    Examples
    --------
    >>> oetf(0.18)  # doctest: +ELLIPSIS
    0.4613561...
    >>> oetf(0.18, function='ITU-R BT.2020')  # doctest: +ELLIPSIS
    0.4090077...
    >>> oetf(0.18, function='ST 2084', L_p=1000)
    ... # doctest: +ELLIPSIS
    0.1820115...
    """

    function = OETFS[function]

    return function(value, **filter_kwargs(function, **kwargs))


OETFS_REVERSE = CaseInsensitiveMapping({
    'ARIB STD-B67': oetf_reverse_ARIBSTDB67,
    'ITU-R BT.2100 HLD': oetf_reverse_BT2100_HLG,
    'ITU-R BT.2100 PQ': oetf_reverse_BT2100_PQ,
    'ITU-R BT.601': oetf_reverse_BT601,
    'ITU-R BT.709': oetf_reverse_BT709,
    'sRGB': oetf_reverse_sRGB
})
OETFS_REVERSE.__doc__ = """
Supported reverse opto-electrical transfer functions (OETFs / OECFs).

OETFS_REVERSE : CaseInsensitiveMapping
    **{'sRGB', 'ARIB STD-B67', 'ITU-R BT.2100 HLD', 'ITU-R BT.2100 PQ',
    'ITU-R BT.601', 'ITU-R BT.709'}**
"""


def oetf_reverse(value, function='sRGB', **kwargs):
    """
    Decodes :math:`R'G'B'` video component signal value to tristimulus values
    at the display using given reverse opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    value : numeric or array_like
        Value.
    function : unicode, optional
        **{'sRGB', 'ARIB STD-B67', 'ITU-R BT.2100 HLD', 'ITU-R BT.2100 PQ',
        'ITU-R BT.601', 'ITU-R BT.709'}**,
        Reverse opto-electronic transfer function (OETF / OECF).

    Other Parameters
    ----------------
    r : numeric, optional
        {:func:`colour.models.oetf_ARIBSTDB67`},
        Video level corresponding to reference white level.

    Returns
    -------
    numeric or ndarray
        Tristimulus values at the display.

    Examples
    --------
    >>> oetf_reverse(0.461356129500442)  # doctest: +ELLIPSIS
    0.1...
    >>> oetf_reverse(  # doctest: +ELLIPSIS
    ...     0.409007728864150, function='ITU-R BT.601')
    0.1...
    """

    function = OETFS_REVERSE[function]

    return function(value, **filter_kwargs(function, **kwargs))


EOTFS = CaseInsensitiveMapping({
    'DCI-P3': eotf_DCIP3,
    'DICOM GSDF': eotf_DICOMGSDF,
    'ITU-R BT.1886': eotf_BT1886,
    'ITU-R BT.2020': eotf_BT2020,
    'ITU-R BT.2100 HLG': eotf_BT2100_HLG,
    'ITU-R BT.2100 PQ': eotf_BT2100_PQ,
    'ProPhoto RGB': eotf_ProPhotoRGB,
    'RIMM RGB': eotf_RIMMRGB,
    'ROMM RGB': eotf_ROMMRGB,
    'SMPTE 240M': eotf_SMPTE240M,
    'ST 2084': eotf_ST2084,
})
EOTFS.__doc__ = """
Supported electro-optical transfer functions (EOTFs / EOCFs).

EOTFS : CaseInsensitiveMapping
    **{'DCI-P3', 'DICOM GSDF', 'ITU-R BT.1886', 'ITU-R BT.2020',
    'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ', 'ProPhoto RGB', 'RIMM RGB',
    'ROMM RGB', 'SMPTE 240M', 'ST 2084'}**
"""


def eotf(value, function='ITU-R BT.1886', **kwargs):
    """
    Decodes :math:`R'G'B'` video component signal value to tristimulus values
    at the display using given electro-optical transfer function (EOTF / EOCF).

    Parameters
    ----------
    value : numeric or array_like
        Value.
    function : unicode, optional
        **{'ITU-R BT.1886', 'DCI-P3', 'DICOM GSDF', 'ITU-R BT.2020',
        'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ', 'ProPhoto RGB', 'RIMM RGB',
        'ROMM RGB', 'SMPTE 240M', 'ST 2084'}**,
        Electro-optical transfer function (EOTF / EOCF).

    Other Parameters
    ----------------
    E_clip : numeric, optional
        {:func:`colour.models.eotf_RIMMRGB`},
        Maximum exposure level.
    I_max : numeric, optional
        {:func:`colour.models.eotf_ROMMRGB`,
        :func:`colour.models.eotf_RIMMRGB`},
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    L_B : numeric, optional
        {:func:`colour.models.eotf_BT1886`,
        :func:`colour.models.eotf_BT2100_HLG`},
        Screen luminance for black.
    L_W : numeric, optional
        {:func:`colour.models.eotf_BT1886`,
        :func:`colour.models.eotf_BT2100_HLG`},
        Screen luminance for white.
    L_p : numeric, optional
        {:func:`colour.models.eotf_ST2084`},
        Display peak luminance :math:`cd/m^2`.
    gamma : numeric, optional
        {:func:`colour.models.eotf_BT2100_HLG`},
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    is_12_bits_system : bool
        {:func:`colour.models.eotf_BT2020`},
        *ITU-R BT.2020* *alpha* and *beta* constants are used if system is not
        12-bit.

    Returns
    -------
    numeric or ndarray
        Tristimulus values at the display.

    Examples
    --------
    >>> eotf(0.461356129500442)  # doctest: +ELLIPSIS
    0.1...
    >>> eotf(0.409007728864150, function='ITU-R BT.2020')
    ... # doctest: +ELLIPSIS
    0.1...
    >>> eotf(0.182011532850008, function='ST 2084', L_p=1000)
    ... # doctest: +ELLIPSIS
    0.1...
    """

    function = EOTFS[function]

    return function(value, **filter_kwargs(function, **kwargs))


EOTFS_REVERSE = CaseInsensitiveMapping({
    'ITU-R BT.1886': eotf_reverse_BT1886,
    'ITU-R BT.2100 HLG': eotf_reverse_BT2100_HLG,
    'ITU-R BT.2100 PQ': eotf_reverse_BT2100_PQ,
})
EOTFS_REVERSE.__doc__ = """
Supported reverse electro-optical transfer functions (EOTFs / EOCFs).

EOTFS_REVERSE : CaseInsensitiveMapping
    **{'ITU-R BT.1886', 'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ'}**
"""


def eotf_reverse(value, function='ITU-R BT.1886', **kwargs):
    """
    Encodes estimated tristimulus values in a scene to :math:`R'G'B'` video
    component signal value using given reverse electro-optical transfer
    function (EOTF / EOCF).

    Parameters
    ----------
    value : numeric or array_like
        Value.
    function : unicode, optional
        **{'ITU-R BT.1886', 'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ'}**,
        Reverse electro-optical transfer function (EOTF / EOCF).

    Other Parameters
    ----------------
    L_B : numeric, optional
        {:func:`colour.models.eotf_reverse_BT1886`,
        :func:`colour.models.eotf_reverse_BT2100_HLG`},
        Screen luminance for black.
    L_W : numeric, optional
        {:func:`colour.models.eotf_reverse_BT1886`,
        :func:`colour.models.eotf_reverse_BT2100_HLG`},
        Screen luminance for white.
    gamma : numeric, optional
        {:func:`colour.models.eotf_BT2100_HLG`},
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    numeric or ndarray
        :math:`R'G'B'` video component signal value.

    Examples
    --------
    >>> eotf_reverse(0.11699185725296059)  # doctest: +ELLIPSIS
    0.4090077...
    >>> eotf_reverse(  # doctest: +ELLIPSIS
    ...     0.11699185725296059, function='ITU-R BT.1886')
    0.4090077...
    """

    function = EOTFS_REVERSE[function]

    return function(value, **filter_kwargs(function, **kwargs))


__all__ += ['OETFS', 'OETFS_REVERSE', 'EOTFS', 'EOTFS_REVERSE']
__all__ += ['oetf', 'oetf_reverse', 'eotf', 'eotf_reverse']

OOTFS = CaseInsensitiveMapping({
    'ITU-R BT.2100 HLG': ootf_BT2100_HLG,
    'ITU-R BT.2100 PQ': ootf_BT2100_PQ,
})
OOTFS.__doc__ = """
Supported opto-optical transfer functions (OOTFs / OOCFs).

OOTFS : CaseInsensitiveMapping
    **{'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ'}**
"""


def ootf(value, function='ITU-R BT.2100 PQ', **kwargs):
    """
    Maps relative scene linear light to display linear light using given
    opto-optical transfer function (OOTF / OOCF).

    Parameters
    ----------
    value : numeric or array_like
        Value.
    function : unicode, optional
        **{'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ'}**
        Opto-optical transfer function (OOTF / OOCF).

    Returns
    -------
    numeric or ndarray
        Luminance of a displayed linear component.

    Examples
    --------
    >>> ootf(0.1)  # doctest: +ELLIPSIS
    779.9883608...
    >>> ootf(0.1, function='ITU-R BT.2100 HLG')  # doctest: +ELLIPSIS
    63.0957344...
    """

    function = OOTFS[function]

    return function(value, **filter_kwargs(function, **kwargs))


OOTFS_REVERSE = CaseInsensitiveMapping({
    'ITU-R BT.2100 HLG': ootf_reverse_BT2100_HLG,
    'ITU-R BT.2100 PQ': ootf_reverse_BT2100_PQ,
})
OOTFS_REVERSE.__doc__ = """
Supported reverse opto-optical transfer functions (OOTFs / OOCFs).

OOTFS_REVERSE : CaseInsensitiveMapping
    **{'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ'}**
"""


def ootf_reverse(value, function='ITU-R BT.2100 PQ', **kwargs):
    """
    Maps relative display linear light to scene linear light using given
    reverse opto-optical transfer function (OOTF / OOCF).

    Parameters
    ----------
    value : numeric or array_like
        Value.
    function : unicode, optional
        **{'ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ'}**
        Reverse opto-optical transfer function (OOTF / OOCF).

    Other Parameters
    ----------------
    L_B : numeric, optional
        {:func:`colour.models.ootf_reverse_BT2100_HLG`},
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W : numeric, optional
        {:func:`colour.models.ootf_reverse_BT2100_HLG`},
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma : numeric, optional
        {:func:`colour.models.ootf_reverse_BT2100_HLG`},
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    numeric or ndarray
        Luminance of scene linear light.

    Examples
    --------
    >>> ootf_reverse(779.988360834115840)  # doctest: +ELLIPSIS
    0.1000000...
    >>> ootf_reverse(  # doctest: +ELLIPSIS
    ...     63.095734448019336, function='ITU-R BT.2100 HLG')
    0.1000000...
    """

    function = OOTFS_REVERSE[function]

    return function(value, **filter_kwargs(function, **kwargs))


__all__ += ['OOTFS', 'OOTFS_REVERSE']
__all__ += ['ootf', 'ootf_reverse']
