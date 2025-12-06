from __future__ import annotations

import typing
from functools import partial

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        NDArrayFloat,
        NDArrayInt,
        LiteralLogEncoding,
        LiteralLogDecoding,
        LiteralOETF,
        LiteralOETFInverse,
        LiteralEOTF,
        LiteralEOTFInverse,
        LiteralCCTFEncoding,
        LiteralCCTFDecoding,
        LiteralOOTF,
        LiteralOOTFInverse,
    )

from colour.utilities import (
    CanonicalMapping,
    filter_kwargs,
    usage_warning,
    validate_method,
)

# isort: split

from .common import CV_range, full_to_legal, legal_to_full
from .gamma import gamma_function

# isort: split

from .aces import (
    log_decoding_ACEScc,
    log_decoding_ACEScct,
    log_decoding_ACESproxy,
    log_encoding_ACEScc,
    log_encoding_ACEScct,
    log_encoding_ACESproxy,
)
from .apple_log_profile import (
    log_decoding_AppleLogProfile,
    log_encoding_AppleLogProfile,
)
from .arib_std_b67 import oetf_ARIBSTDB67, oetf_inverse_ARIBSTDB67
from .arri import (
    log_decoding_ARRILogC3,
    log_decoding_ARRILogC4,
    log_encoding_ARRILogC3,
    log_encoding_ARRILogC4,
)
from .blackmagic_design import (
    oetf_BlackmagicFilmGeneration5,
    oetf_inverse_BlackmagicFilmGeneration5,
)
from .canon import (
    CANON_LOG_2_DECODING_METHODS,
    CANON_LOG_2_ENCODING_METHODS,
    CANON_LOG_3_DECODING_METHODS,
    CANON_LOG_3_ENCODING_METHODS,
    CANON_LOG_DECODING_METHODS,
    CANON_LOG_ENCODING_METHODS,
    log_decoding_CanonLog,
    log_decoding_CanonLog2,
    log_decoding_CanonLog3,
    log_encoding_CanonLog,
    log_encoding_CanonLog2,
    log_encoding_CanonLog3,
)
from .cineon import log_decoding_Cineon, log_encoding_Cineon
from .davinci_intermediate import (
    oetf_DaVinciIntermediate,
    oetf_inverse_DaVinciIntermediate,
)
from .dcdm import eotf_DCDM, eotf_inverse_DCDM
from .dicom_gsdf import eotf_DICOMGSDF, eotf_inverse_DICOMGSDF
from .dji_d_log import log_decoding_DJIDLog, log_encoding_DJIDLog
from .exponent import exponent_function_basic, exponent_function_monitor_curve
from .filmic_pro import log_decoding_FilmicPro6, log_encoding_FilmicPro6
from .filmlight_t_log import (
    log_decoding_FilmLightTLog,
    log_encoding_FilmLightTLog,
)
from .fujifilm_f_log import (
    log_decoding_FLog,
    log_decoding_FLog2,
    log_encoding_FLog,
    log_encoding_FLog2,
)
from .gopro import log_decoding_Protune, log_encoding_Protune
from .itur_bt_601 import oetf_BT601, oetf_inverse_BT601
from .itur_bt_709 import oetf_BT709, oetf_inverse_BT709
from .itur_bt_1361 import oetf_BT1361, oetf_inverse_BT1361
from .itur_bt_1886 import eotf_BT1886, eotf_inverse_BT1886
from .itur_bt_2020 import oetf_BT2020, oetf_inverse_BT2020

# isort: split

from .st_2084 import eotf_inverse_ST2084, eotf_ST2084

# isort: split

from .itur_bt_2100 import (
    BT2100_HLG_EOTF_INVERSE_METHODS,
    BT2100_HLG_EOTF_METHODS,
    BT2100_HLG_OOTF_INVERSE_METHODS,
    BT2100_HLG_OOTF_METHODS,
    eotf_BT2100_HLG,
    eotf_BT2100_PQ,
    eotf_inverse_BT2100_HLG,
    eotf_inverse_BT2100_PQ,
    oetf_BT2100_HLG,
    oetf_BT2100_PQ,
    oetf_inverse_BT2100_HLG,
    oetf_inverse_BT2100_PQ,
    ootf_BT2100_HLG,
    ootf_BT2100_PQ,
    ootf_inverse_BT2100_HLG,
    ootf_inverse_BT2100_PQ,
)
from .leica_l_log import log_decoding_LLog, log_encoding_LLog
from .linear import linear_function
from .log import (
    log_decoding_Log2,
    log_encoding_Log2,
    logarithmic_function_basic,
    logarithmic_function_camera,
    logarithmic_function_quasilog,
)
from .nikon_n_log import log_decoding_NLog, log_encoding_NLog
from .panalog import log_decoding_Panalog, log_encoding_Panalog
from .panasonic_v_log import log_decoding_VLog, log_encoding_VLog
from .pivoted_log import log_decoding_PivotedLog, log_encoding_PivotedLog
from .red import (
    LOG3G10_DECODING_METHODS,
    LOG3G10_ENCODING_METHODS,
    log_decoding_Log3G10,
    log_decoding_Log3G12,
    log_decoding_REDLog,
    log_decoding_REDLogFilm,
    log_encoding_Log3G10,
    log_encoding_Log3G12,
    log_encoding_REDLog,
    log_encoding_REDLogFilm,
)
from .rimm_romm_rgb import (
    cctf_decoding_ProPhotoRGB,
    cctf_decoding_RIMMRGB,
    cctf_decoding_ROMMRGB,
    cctf_encoding_ProPhotoRGB,
    cctf_encoding_RIMMRGB,
    cctf_encoding_ROMMRGB,
    log_decoding_ERIMMRGB,
    log_encoding_ERIMMRGB,
)
from .smpte_240m import eotf_SMPTE240M, oetf_SMPTE240M
from .sony import (
    log_decoding_SLog,
    log_decoding_SLog2,
    log_decoding_SLog3,
    log_encoding_SLog,
    log_encoding_SLog2,
    log_encoding_SLog3,
)
from .srgb import eotf_inverse_sRGB, eotf_sRGB
from .viper_log import log_decoding_ViperLog, log_encoding_ViperLog
from .xiaomi_mi_log import log_decoding_MiLog, log_encoding_MiLog

# isort: split

from .itut_h_273 import (
    eotf_H273_ST428_1,
    eotf_inverse_H273_ST428_1,
    oetf_H273_IEC61966_2,
    oetf_H273_Log,
    oetf_H273_LogSqrt,
    oetf_inverse_H273_IEC61966_2,
    oetf_inverse_H273_Log,
    oetf_inverse_H273_LogSqrt,
)

__all__ = [
    "CV_range",
    "full_to_legal",
    "legal_to_full",
]
__all__ += [
    "gamma_function",
]
__all__ += [
    "log_decoding_ACEScc",
    "log_decoding_ACEScct",
    "log_decoding_ACESproxy",
    "log_encoding_ACEScc",
    "log_encoding_ACEScct",
    "log_encoding_ACESproxy",
]
__all__ += [
    "log_decoding_AppleLogProfile",
    "log_encoding_AppleLogProfile",
]
__all__ += [
    "oetf_ARIBSTDB67",
    "oetf_inverse_ARIBSTDB67",
]
__all__ += [
    "log_decoding_ARRILogC3",
    "log_decoding_ARRILogC4",
    "log_encoding_ARRILogC3",
    "log_encoding_ARRILogC4",
]
__all__ += [
    "oetf_BlackmagicFilmGeneration5",
    "oetf_inverse_BlackmagicFilmGeneration5",
]
__all__ += [
    "CANON_LOG_2_DECODING_METHODS",
    "CANON_LOG_2_ENCODING_METHODS",
    "CANON_LOG_3_DECODING_METHODS",
    "CANON_LOG_3_ENCODING_METHODS",
    "CANON_LOG_DECODING_METHODS",
    "CANON_LOG_ENCODING_METHODS",
    "log_decoding_CanonLog",
    "log_decoding_CanonLog2",
    "log_decoding_CanonLog3",
    "log_encoding_CanonLog",
    "log_encoding_CanonLog2",
    "log_encoding_CanonLog3",
]
__all__ += [
    "log_decoding_Cineon",
    "log_encoding_Cineon",
]
__all__ += [
    "oetf_DaVinciIntermediate",
    "oetf_inverse_DaVinciIntermediate",
]
__all__ += [
    "eotf_DCDM",
    "eotf_inverse_DCDM",
]
__all__ += [
    "eotf_DICOMGSDF",
    "eotf_inverse_DICOMGSDF",
]
__all__ += [
    "log_decoding_DJIDLog",
    "log_encoding_DJIDLog",
]
__all__ += [
    "exponent_function_basic",
    "exponent_function_monitor_curve",
]
__all__ += [
    "log_decoding_FilmicPro6",
    "log_encoding_FilmicPro6",
]
__all__ += [
    "log_decoding_FilmLightTLog",
    "log_encoding_FilmLightTLog",
]
__all__ += [
    "log_decoding_FLog",
    "log_decoding_FLog2",
    "log_encoding_FLog",
    "log_encoding_FLog2",
]
__all__ += [
    "log_decoding_Protune",
    "log_encoding_Protune",
]
__all__ += [
    "oetf_BT601",
    "oetf_inverse_BT601",
]
__all__ += [
    "oetf_BT709",
    "oetf_inverse_BT709",
]
__all__ += [
    "oetf_BT1361",
    "oetf_inverse_BT1361",
]
__all__ += [
    "eotf_BT1886",
    "eotf_inverse_BT1886",
]
__all__ += [
    "oetf_BT2020",
    "oetf_inverse_BT2020",
]
__all__ += [
    "eotf_inverse_ST2084",
    "eotf_ST2084",
]
__all__ += [
    "BT2100_HLG_EOTF_INVERSE_METHODS",
    "BT2100_HLG_EOTF_METHODS",
    "BT2100_HLG_OOTF_INVERSE_METHODS",
    "BT2100_HLG_OOTF_METHODS",
    "eotf_BT2100_HLG",
    "eotf_BT2100_PQ",
    "eotf_inverse_BT2100_HLG",
    "eotf_inverse_BT2100_PQ",
    "oetf_BT2100_HLG",
    "oetf_BT2100_PQ",
    "oetf_inverse_BT2100_HLG",
    "oetf_inverse_BT2100_PQ",
    "ootf_BT2100_HLG",
    "ootf_BT2100_PQ",
    "ootf_inverse_BT2100_HLG",
    "ootf_inverse_BT2100_PQ",
]
__all__ += [
    "log_decoding_LLog",
    "log_encoding_LLog",
]
__all__ += [
    "linear_function",
]
__all__ += [
    "log_decoding_Log2",
    "log_encoding_Log2",
    "logarithmic_function_basic",
    "logarithmic_function_camera",
    "logarithmic_function_quasilog",
]
__all__ += [
    "log_decoding_NLog",
    "log_encoding_NLog",
]
__all__ += [
    "log_decoding_Panalog",
    "log_encoding_Panalog",
]
__all__ += [
    "log_decoding_VLog",
    "log_encoding_VLog",
]
__all__ += [
    "log_decoding_PivotedLog",
    "log_encoding_PivotedLog",
]
__all__ += [
    "LOG3G10_DECODING_METHODS",
    "LOG3G10_ENCODING_METHODS",
    "log_decoding_Log3G10",
    "log_decoding_Log3G12",
    "log_decoding_REDLog",
    "log_decoding_REDLogFilm",
    "log_encoding_Log3G10",
    "log_encoding_Log3G12",
    "log_encoding_REDLog",
    "log_encoding_REDLogFilm",
]
__all__ += [
    "cctf_decoding_ProPhotoRGB",
    "cctf_decoding_RIMMRGB",
    "cctf_decoding_ROMMRGB",
    "cctf_encoding_ProPhotoRGB",
    "cctf_encoding_RIMMRGB",
    "cctf_encoding_ROMMRGB",
    "log_decoding_ERIMMRGB",
    "log_encoding_ERIMMRGB",
]
__all__ += [
    "eotf_SMPTE240M",
    "oetf_SMPTE240M",
]
__all__ += [
    "log_decoding_SLog",
    "log_decoding_SLog2",
    "log_decoding_SLog3",
    "log_encoding_SLog",
    "log_encoding_SLog2",
    "log_encoding_SLog3",
]
__all__ += [
    "eotf_inverse_sRGB",
    "eotf_sRGB",
]
__all__ += [
    "log_decoding_ViperLog",
    "log_encoding_ViperLog",
]
__all__ += [
    "log_decoding_MiLog",
    "log_encoding_MiLog",
]
__all__ += [
    "eotf_H273_ST428_1",
    "eotf_inverse_H273_ST428_1",
    "oetf_H273_IEC61966_2",
    "oetf_H273_Log",
    "oetf_H273_LogSqrt",
    "oetf_inverse_H273_IEC61966_2",
    "oetf_inverse_H273_Log",
    "oetf_inverse_H273_LogSqrt",
]

LOG_ENCODINGS: CanonicalMapping = CanonicalMapping(
    {
        "ACEScc": log_encoding_ACEScc,
        "ACEScct": log_encoding_ACEScct,
        "ACESproxy": log_encoding_ACESproxy,
        "Apple Log Profile": log_encoding_AppleLogProfile,
        "ARRI LogC3": log_encoding_ARRILogC3,
        "ARRI LogC4": log_encoding_ARRILogC4,
        "Canon Log 2": log_encoding_CanonLog2,
        "Canon Log 3": log_encoding_CanonLog3,
        "Canon Log": log_encoding_CanonLog,
        "Cineon": log_encoding_Cineon,
        "D-Log": log_encoding_DJIDLog,
        "ERIMM RGB": log_encoding_ERIMMRGB,
        "F-Log": log_encoding_FLog,
        "F-Log2": log_encoding_FLog2,
        "Filmic Pro 6": log_encoding_FilmicPro6,
        "L-Log": log_encoding_LLog,
        "Log2": log_encoding_Log2,
        "Log3G10": log_encoding_Log3G10,
        "Log3G12": log_encoding_Log3G12,
        "Mi-Log": log_encoding_MiLog,
        "N-Log": log_encoding_NLog,
        "PLog": log_encoding_PivotedLog,
        "Panalog": log_encoding_Panalog,
        "Protune": log_encoding_Protune,
        "REDLog": log_encoding_REDLog,
        "REDLogFilm": log_encoding_REDLogFilm,
        "S-Log": log_encoding_SLog,
        "S-Log2": log_encoding_SLog2,
        "S-Log3": log_encoding_SLog3,
        "T-Log": log_encoding_FilmLightTLog,
        "V-Log": log_encoding_VLog,
        "ViperLog": log_encoding_ViperLog,
    }
)
LOG_ENCODINGS.__doc__ = """
Supported *log* encoding functions.
"""


def log_encoding(
    value: ArrayLike, function: LiteralLogEncoding | str = "Cineon", **kwargs: Any
) -> NDArrayFloat | NDArrayInt:
    """
    Apply the specified log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    value
        Scene-linear value.
    function
        *Log* encoding function.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.log_encoding_ACEScc`,
        :func:`colour.models.log_encoding_ACEScct`,
        :func:`colour.models.log_encoding_ACESproxy`,
        :func:`colour.models.log_encoding_AppleLogProfile`,
        :func:`colour.models.log_encoding_ARRILogC3`,
        :func:`colour.models.log_encoding_ARRILogC4`,
        :func:`colour.models.log_encoding_CanonLog2`,
        :func:`colour.models.log_encoding_CanonLog3`,
        :func:`colour.models.log_encoding_CanonLog`,
        :func:`colour.models.log_encoding_Cineon`,
        :func:`colour.models.log_encoding_DJIDLog`,
        :func:`colour.models.log_encoding_ERIMMRGB`,
        :func:`colour.models.log_encoding_FLog`,
        :func:`colour.models.log_encoding_FLog2`,
        :func:`colour.models.log_encoding_FilmicPro6`,
        :func:`colour.models.log_encoding_LLog`,
        :func:`colour.models.log_encoding_Log2`,
        :func:`colour.models.log_encoding_Log3G10`,
        :func:`colour.models.log_encoding_Log3G12`,
        :func:`colour.models.log_encoding_MiLog`,
        :func:`colour.models.log_encoding_NLog`,
        :func:`colour.models.log_encoding_PivotedLog`,
        :func:`colour.models.log_encoding_Panalog`,
        :func:`colour.models.log_encoding_Protune`,
        :func:`colour.models.log_encoding_REDLog`,
        :func:`colour.models.log_encoding_REDLogFilm`,
        :func:`colour.models.log_encoding_SLog`,
        :func:`colour.models.log_encoding_SLog2`,
        :func:`colour.models.log_encoding_SLog3`,
        :func:`colour.models.log_encoding_FilmLightTLog`,
        :func:`colour.models.log_encoding_VLog`,
        :func:`colour.models.log_encoding_ViperLog`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Logarithmic encoded value.

    Examples
    --------
    >>> log_encoding(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    >>> log_encoding(0.18, function="ACEScc")  # doctest: +ELLIPSIS
    0.4135884...
    >>> log_encoding(0.18, function="PLog", log_reference=400)
    ... # doctest: +ELLIPSIS
    0.3910068...
    >>> log_encoding(0.18, function="S-Log")  # doctest: +ELLIPSIS
    0.3849708...
    """

    function = validate_method(
        function,
        tuple(LOG_ENCODINGS),
        '"{0}" "log" encoding function is invalid, it must be one of {1}!',
    )

    callable_ = LOG_ENCODINGS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


LOG_DECODINGS: CanonicalMapping = CanonicalMapping(
    {
        "ACEScc": log_decoding_ACEScc,
        "ACEScct": log_decoding_ACEScct,
        "ACESproxy": log_decoding_ACESproxy,
        "Apple Log Profile": log_decoding_AppleLogProfile,
        "ARRI LogC3": log_decoding_ARRILogC3,
        "ARRI LogC4": log_decoding_ARRILogC4,
        "Canon Log 2": log_decoding_CanonLog2,
        "Canon Log 3": log_decoding_CanonLog3,
        "Canon Log": log_decoding_CanonLog,
        "Cineon": log_decoding_Cineon,
        "D-Log": log_decoding_DJIDLog,
        "ERIMM RGB": log_decoding_ERIMMRGB,
        "F-Log": log_decoding_FLog,
        "F-Log2": log_decoding_FLog2,
        "Filmic Pro 6": log_decoding_FilmicPro6,
        "L-Log": log_decoding_LLog,
        "Log2": log_decoding_Log2,
        "Log3G10": log_decoding_Log3G10,
        "Log3G12": log_decoding_Log3G12,
        "Mi-Log": log_decoding_MiLog,
        "N-Log": log_decoding_NLog,
        "PLog": log_decoding_PivotedLog,
        "Panalog": log_decoding_Panalog,
        "Protune": log_decoding_Protune,
        "REDLog": log_decoding_REDLog,
        "REDLogFilm": log_decoding_REDLogFilm,
        "S-Log": log_decoding_SLog,
        "S-Log2": log_decoding_SLog2,
        "S-Log3": log_decoding_SLog3,
        "T-Log": log_decoding_FilmLightTLog,
        "V-Log": log_decoding_VLog,
        "ViperLog": log_decoding_ViperLog,
    }
)
LOG_DECODINGS.__doc__ = """
Supported *log* decoding functions.
"""


def log_decoding(
    value: ArrayLike,
    function: LiteralLogDecoding | str = "Cineon",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Apply the specified log decoding inverse opto-electronic transfer function (OETF).

    Parameters
    ----------
    value
        Logarithmic encoded value.
    function
        *Log* decoding function.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.log_decoding_ACEScc`,
        :func:`colour.models.log_decoding_ACEScct`,
        :func:`colour.models.log_decoding_ACESproxy`,
        :func:`colour.models.log_decoding_AppleLogProfile`,
        :func:`colour.models.log_decoding_ARRILogC3`,
        :func:`colour.models.log_decoding_ARRILogC4`,
        :func:`colour.models.log_decoding_CanonLog2`,
        :func:`colour.models.log_decoding_CanonLog3`,
        :func:`colour.models.log_decoding_CanonLog`,
        :func:`colour.models.log_decoding_Cineon`,
        :func:`colour.models.log_decoding_DJIDLog`,
        :func:`colour.models.log_decoding_ERIMMRGB`,
        :func:`colour.models.log_decoding_FLog`,
        :func:`colour.models.log_decoding_FLog2`,
        :func:`colour.models.log_decoding_FilmicPro6`,
        :func:`colour.models.log_decoding_LLog`,
        :func:`colour.models.log_decoding_Log2`,
        :func:`colour.models.log_decoding_Log3G10`,
        :func:`colour.models.log_decoding_Log3G12`,
        :func:`colour.models.log_decoding_MiLog`,
        :func:`colour.models.log_decoding_NLog`,
        :func:`colour.models.log_decoding_PivotedLog`,
        :func:`colour.models.log_decoding_Panalog`,
        :func:`colour.models.log_decoding_Protune`,
        :func:`colour.models.log_decoding_REDLog`,
        :func:`colour.models.log_decoding_REDLogFilm`,
        :func:`colour.models.log_decoding_SLog`,
        :func:`colour.models.log_decoding_SLog2`,
        :func:`colour.models.log_decoding_SLog3`,
        :func:`colour.models.log_decoding_FilmLightTLog`,
        :func:`colour.models.log_decoding_VLog`,
        :func:`colour.models.log_decoding_ViperLog`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Scene-linear value.

    Examples
    --------
    >>> log_decoding(0.457319613085418)  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding(0.413588402492442, function="ACEScc")
    ... # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding(0.391006842619746, function="PLog", log_reference=400)
    ... # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding(0.376512722254600, function="S-Log")
    ... # doctest: +ELLIPSIS
    0.1...
    """

    function = validate_method(
        function,
        tuple(LOG_DECODINGS),
        '"{0}" "log" decoding function is invalid, it must be one of {1}!',
    )

    callable_ = LOG_DECODINGS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


__all__ += [
    "LOG_ENCODINGS",
    "LOG_DECODINGS",
]
__all__ += [
    "log_encoding",
    "log_decoding",
]

OETFS: CanonicalMapping = CanonicalMapping(
    {
        "ARIB STD-B67": oetf_ARIBSTDB67,
        "Blackmagic Film Generation 5": oetf_BlackmagicFilmGeneration5,
        "DaVinci Intermediate": oetf_DaVinciIntermediate,
        "ITU-R BT.2020": oetf_BT2020,
        "ITU-R BT.2100 HLG": oetf_BT2100_HLG,
        "ITU-R BT.2100 PQ": oetf_BT2100_PQ,
        "ITU-R BT.601": oetf_BT601,
        "ITU-R BT.709": oetf_BT709,
        "ITU-T H.273 Log": oetf_H273_Log,
        "ITU-T H.273 Log Sqrt": oetf_H273_LogSqrt,
        "ITU-T H.273 IEC 61966-2": oetf_H273_IEC61966_2,
        "SMPTE 240M": oetf_SMPTE240M,
    }
)
OETFS.__doc__ = """
Supported opto-electrical transfer functions (OETFs / OECFs).
"""


def oetf(
    value: ArrayLike, function: LiteralOETF | str = "ITU-R BT.709", **kwargs: Any
) -> NDArrayFloat:
    """
    Apply the specified opto-electronic transfer function (OETF).

    Parameters
    ----------
    value
        Scene-linear value.
    function
        Opto-electronic transfer function (OETF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.oetf_ARIBSTDB67`,
        :func:`colour.models.oetf_BlackmagicFilmGeneration5`,
        :func:`colour.models.oetf_DaVinciIntermediate`,
        :func:`colour.models.oetf_BT2020`,
        :func:`colour.models.oetf_BT2100_HLG`,
        :func:`colour.models.oetf_BT2100_PQ`,
        :func:`colour.models.oetf_BT601`,
        :func:`colour.models.oetf_BT709`,
        :func:`colour.models.oetf_SMPTE240M`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Non-linear signal value.

    Examples
    --------
    >>> oetf(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    >>> oetf(0.18, function="ITU-R BT.601")  # doctest: +ELLIPSIS
    0.4090077...
    """

    function = validate_method(
        function,
        tuple(OETFS),
        '"{0}" "OETF" is invalid, it must be one of {1}!',
    )

    callable_ = OETFS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


OETF_INVERSES: CanonicalMapping = CanonicalMapping(
    {
        "ARIB STD-B67": oetf_inverse_ARIBSTDB67,
        "Blackmagic Film Generation 5": oetf_inverse_BlackmagicFilmGeneration5,
        "DaVinci Intermediate": oetf_inverse_DaVinciIntermediate,
        "ITU-R BT.2020": oetf_inverse_BT2020,
        "ITU-R BT.2100 HLG": oetf_inverse_BT2100_HLG,
        "ITU-R BT.2100 PQ": oetf_inverse_BT2100_PQ,
        "ITU-R BT.601": oetf_inverse_BT601,
        "ITU-R BT.709": oetf_inverse_BT709,
        "ITU-T H.273 Log": oetf_inverse_H273_Log,
        "ITU-T H.273 Log Sqrt": oetf_inverse_H273_LogSqrt,
        "ITU-T H.273 IEC 61966-2": oetf_inverse_H273_IEC61966_2,
    }
)
OETF_INVERSES.__doc__ = """
Supported inverse opto-electrical transfer functions (OETFs / OECFs).
"""


def oetf_inverse(
    value: ArrayLike,
    function: LiteralOETFInverse | str = "ITU-R BT.709",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Apply the specified inverse opto-electronic transfer function (OETF).

    Parameters
    ----------
    value
        Non-linear signal value.
    function
        Inverse opto-electronic transfer function (OETF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.oetf_inverse_ARIBSTDB67`,
        :func:`colour.models.oetf_inverse_BlackmagicFilmGeneration5`,
        :func:`colour.models.oetf_inverse_DaVinciIntermediate`,
        :func:`colour.models.oetf_inverse_BT2020`,
        :func:`colour.models.oetf_inverse_BT2100_HLG`,
        :func:`colour.models.oetf_inverse_BT2100_PQ`,
        :func:`colour.models.oetf_inverse_BT601`,
        :func:`colour.models.oetf_inverse_BT709`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Scene-linear value.

    Examples
    --------
    >>> oetf_inverse(0.409007728864150)  # doctest: +ELLIPSIS
    0.1...
    >>> oetf_inverse(  # doctest: +ELLIPSIS
    ...     0.409007728864150, function="ITU-R BT.601"
    ... )
    0.1...
    """

    function = validate_method(
        function,
        tuple(OETF_INVERSES),
        '"{0}" inverse "OETF" is invalid, it must be one of {1}!',
    )

    callable_ = OETF_INVERSES[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


EOTFS: CanonicalMapping = CanonicalMapping(
    {
        "DCDM": eotf_DCDM,
        "DICOM GSDF": eotf_DICOMGSDF,
        "ITU-R BT.1886": eotf_BT1886,
        "ITU-R BT.2100 HLG": eotf_BT2100_HLG,
        "ITU-R BT.2100 PQ": eotf_BT2100_PQ,
        "ITU-T H.273 ST.428-1": eotf_H273_ST428_1,
        "SMPTE 240M": eotf_SMPTE240M,
        "ST 2084": eotf_ST2084,
        "sRGB": eotf_sRGB,
    }
)
EOTFS.__doc__ = """
Supported electro-optical transfer functions (EOTFs / EOCFs).
"""


def eotf(
    value: ArrayLike,
    function: LiteralEOTF | str = "ITU-R BT.1886",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Apply the specified electro-optical transfer function (EOTF).

    Parameters
    ----------
    value
        Non-linear signal value.
    function
        Electro-optical transfer function (EOTF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.eotf_DCDM`,
        :func:`colour.models.eotf_DICOMGSDF`,
        :func:`colour.models.eotf_BT1886`,
        :func:`colour.models.eotf_BT2100_HLG`,
        :func:`colour.models.eotf_BT2100_PQ`,
        :func:`colour.models.eotf_SMPTE240M`,
        :func:`colour.models.eotf_ST2084`,
        :func:`colour.models.eotf_sRGB`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Display-linear value.

    Examples
    --------
    >>> eotf(0.461356129500442)  # doctest: +ELLIPSIS
    0.1...
    >>> eotf(0.182011532850008, function="ST 2084", L_p=1000)
    ... # doctest: +ELLIPSIS
    0.1...
    """

    function = validate_method(
        function,
        tuple(EOTFS),
        '"{0}" "EOTF" is invalid, it must be one of {1}!',
    )

    callable_ = EOTFS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


EOTF_INVERSES: CanonicalMapping = CanonicalMapping(
    {
        "DCDM": eotf_inverse_DCDM,
        "DICOM GSDF": eotf_inverse_DICOMGSDF,
        "ITU-R BT.1886": eotf_inverse_BT1886,
        "ITU-R BT.2100 HLG": eotf_inverse_BT2100_HLG,
        "ITU-R BT.2100 PQ": eotf_inverse_BT2100_PQ,
        "ITU-T H.273 ST.428-1": eotf_inverse_H273_ST428_1,
        "ST 2084": eotf_inverse_ST2084,
        "sRGB": eotf_inverse_sRGB,
    }
)
EOTF_INVERSES.__doc__ = """
Supported inverse electro-optical transfer functions (EOTFs / EOCFs).
"""


def eotf_inverse(
    value: ArrayLike,
    function: LiteralEOTFInverse | str = "ITU-R BT.1886",
    **kwargs: Any,
) -> NDArrayFloat | NDArrayInt:
    """
    Apply the specified inverse electro-optical transfer function (EOTF).

    Parameters
    ----------
    value
        Display-linear value.
    function
        Inverse electro-optical transfer function (EOTF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.eotf_inverse_DCDM`,
        :func:`colour.models.eotf_inverse_DICOMGSDF`,
        :func:`colour.models.eotf_inverse_BT1886`,
        :func:`colour.models.eotf_inverse_BT2100_HLG`,
        :func:`colour.models.eotf_inverse_BT2100_PQ`,
        :func:`colour.models.eotf_inverse_ST2084`,
        :func:`colour.models.eotf_inverse_sRGB`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Non-linear signal value.

    Examples
    --------
    >>> eotf_inverse(0.11699185725296059)  # doctest: +ELLIPSIS
    0.4090077...
    >>> eotf_inverse(  # doctest: +ELLIPSIS
    ...     0.11699185725296059, function="ITU-R BT.1886"
    ... )
    0.4090077...
    """

    function = validate_method(
        function,
        tuple(EOTF_INVERSES),
        '"{0}" inverse "EOTF" is invalid, it must be one of {1}!',
    )

    callable_ = EOTF_INVERSES[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


__all__ += [
    "OETFS",
    "OETF_INVERSES",
    "EOTFS",
    "EOTF_INVERSES",
]
__all__ += [
    "oetf",
    "oetf_inverse",
    "eotf",
    "eotf_inverse",
]

CCTF_ENCODINGS: CanonicalMapping = CanonicalMapping(
    {
        "Gamma 2.2": partial(gamma_function, exponent=1 / 2.2),
        "Gamma 2.4": partial(gamma_function, exponent=1 / 2.4),
        "Gamma 2.6": partial(gamma_function, exponent=1 / 2.6),
        "ProPhoto RGB": cctf_encoding_ProPhotoRGB,
        "RIMM RGB": cctf_encoding_RIMMRGB,
        "ROMM RGB": cctf_encoding_ROMMRGB,
    }
)
CCTF_ENCODINGS.update(LOG_ENCODINGS)
CCTF_ENCODINGS.update(OETFS)
CCTF_ENCODINGS.update(EOTF_INVERSES)
CCTF_ENCODINGS.__doc__ = """
Supported encoding colour component transfer functions (encoding CCTFs), a
collection comprising functions from :attr:`colour.LOG_ENCODINGS`,
:attr:`colour.OETFS`, :attr:`colour.EOTF_INVERSES`,
:func:`colour.models.cctf_encoding_ProPhotoRGB`,
:func:`colour.models.cctf_encoding_RIMMRGB`,
:func:`colour.models.cctf_encoding_ROMMRGB`, and three gamma encoding
functions (1/2.2, 1/2.4, 1/2.6).

Warnings
--------
For *ITU-R BT.2100*, only the inverse electro-optical transfer functions
(EOTFs) are exposed by this definition, See the :func:`colour.oetf`
definition for the opto-electronic transfer functions (OETF).
"""


def cctf_encoding(
    value: ArrayLike, function: LiteralCCTFEncoding | str = "sRGB", **kwargs: Any
) -> NDArrayFloat | NDArrayInt:
    """
    Apply the specified encoding colour component transfer function (Encoding
    CCTF).

    Parameters
    ----------
    value
        Linear RGB value.
    function
        Encoding colour component transfer function.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments for the relevant encoding *CCTF* of the
        :attr:`colour.CCTF_ENCODINGS` attribute collection.

    Warnings
    --------
    For *ITU-R BT.2100*, only the inverse electro-optical transfer functions
    (EOTFs) are exposed by this definition, See the :func:`colour.oetf`
    definition for the opto-electronic transfer functions (OETF).

    Returns
    -------
    :class:`numpy.ndarray`
        Non-linear RGB value.

    Examples
    --------
    >>> cctf_encoding(0.18, function="PLog", log_reference=400)
    ... # doctest: +ELLIPSIS
    0.3910068...
    >>> cctf_encoding(0.18, function="ST 2084", L_p=1000)
    ... # doctest: +ELLIPSIS
    0.1820115...
    >>> cctf_encoding(  # doctest: +ELLIPSIS
    ...     0.11699185725296059, function="ITU-R BT.1886"
    ... )
    0.4090077...
    """

    function = validate_method(
        function,
        tuple(CCTF_ENCODINGS),
        '"{0}" encoding "CCTF" is invalid, it must be one of {1}!',
    )

    if "itu-r bt.2100" in function:
        usage_warning(
            'With the "ITU-R BT.2100" method, only the inverse '
            "electro-optical transfer functions (EOTFs / EOCFs) are exposed "
            'by this definition, See the "colour.oetf" definition '
            "for the opto-electronic transfer functions (OETF)."
        )

    callable_ = CCTF_ENCODINGS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


CCTF_DECODINGS: CanonicalMapping = CanonicalMapping(
    {
        "Gamma 2.2": partial(gamma_function, exponent=2.2),
        "Gamma 2.4": partial(gamma_function, exponent=2.4),
        "Gamma 2.6": partial(gamma_function, exponent=2.6),
        "ProPhoto RGB": cctf_decoding_ProPhotoRGB,
        "RIMM RGB": cctf_decoding_RIMMRGB,
        "ROMM RGB": cctf_decoding_ROMMRGB,
    }
)
CCTF_DECODINGS.update(LOG_DECODINGS)
CCTF_DECODINGS.update(OETF_INVERSES)
CCTF_DECODINGS.update(EOTFS)
CCTF_DECODINGS.__doc__ = """
Supported decoding colour component transfer functions (decoding CCTFs), a
collection comprising functions from :attr:`colour.LOG_DECODINGS`,
:attr:`colour.OETF_INVERSES`, :attr:`colour.EOTFS`,
:func:`colour.models.cctf_decoding_ProPhotoRGB`,
:func:`colour.models.cctf_decoding_RIMMRGB`,
:func:`colour.models.cctf_decoding_ROMMRGB`, and three gamma decoding
functions (2.2, 2.4, 2.6).

Warnings
--------
For *ITU-R BT.2100*, only the electro-optical transfer functions
(EOTFs) are exposed by this attribute. See :attr:`colour.OETF_INVERSES`
for the inverse opto-electronic transfer functions (OETFs).
"""


def cctf_decoding(
    value: ArrayLike,
    function: LiteralCCTFDecoding | str = "sRGB",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Apply the specified decoding colour component transfer function (Decoding
    CCTF).

    Parameters
    ----------
    value
        Non-linear RGB value.
    function
        Decoding colour component transfer function.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments for the relevant decoding *CCTF* of the
        :attr:`colour.CCTF_DECODINGS` attribute collection.

    Warnings
    --------
    For *ITU-R BT.2100*, only the electro-optical transfer functions
    (EOTFs) are exposed by this attribute. See :attr:`colour.OETF_INVERSES`
    for the inverse opto-electronic transfer functions (OETFs).

    Returns
    -------
    :class:`numpy.ndarray`
        Linear RGB value.

    Examples
    --------
    >>> cctf_decoding(0.391006842619746, function="PLog", log_reference=400)
    ... # doctest: +ELLIPSIS
    0.1...
    >>> cctf_decoding(0.182011532850008, function="ST 2084", L_p=1000)
    ... # doctest: +ELLIPSIS
    0.1...
    >>> cctf_decoding(  # doctest: +ELLIPSIS
    ...     0.461356129500442, function="ITU-R BT.1886"
    ... )
    0.1...
    """

    function = validate_method(
        function,
        tuple(CCTF_DECODINGS),
        '"{0}" decoding "CCTF" is invalid, it must be one of {1}!',
    )

    if "itu-r bt.2100" in function:
        usage_warning(
            'With the "ITU-R BT.2100" method, only the electro-optical '
            "transfer functions (EOTFs / EOCFs) are exposed by this "
            'definition, See the "colour.oetf_inverse" definition '
            "for the inverse opto-electronic transfer functions (OETF)."
        )

    callable_ = CCTF_DECODINGS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


__all__ += [
    "CCTF_ENCODINGS",
    "CCTF_DECODINGS",
]
__all__ += [
    "cctf_encoding",
    "cctf_decoding",
]

OOTFS: CanonicalMapping = CanonicalMapping(
    {
        "ITU-R BT.2100 HLG": ootf_BT2100_HLG,
        "ITU-R BT.2100 PQ": ootf_BT2100_PQ,
    }
)
OOTFS.__doc__ = """
Supported opto-optical transfer functions (OOTFs / OOCFs).
"""


def ootf(
    value: ArrayLike,
    function: LiteralOOTF | str = "ITU-R BT.2100 PQ",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Apply the specified opto-optical transfer function (OOTF).

    Parameters
    ----------
    value
        Scene-linear value.
    function
        Opto-optical transfer function (OOTF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.ootf_BT2100_HLG`,
        :func:`colour.models.ootf_BT2100_PQ`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Display-linear value.

    Examples
    --------
    >>> ootf(0.1)  # doctest: +ELLIPSIS
    779.9883608...
    >>> ootf(0.1, function="ITU-R BT.2100 HLG")  # doctest: +ELLIPSIS
    63.0957344...
    """

    function = validate_method(
        function,
        tuple(OOTFS),
        '"{0}" "OOTF" is invalid, it must be one of {1}!',
    )

    callable_ = OOTFS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


OOTF_INVERSES: CanonicalMapping = CanonicalMapping(
    {
        "ITU-R BT.2100 HLG": ootf_inverse_BT2100_HLG,
        "ITU-R BT.2100 PQ": ootf_inverse_BT2100_PQ,
    }
)
OOTF_INVERSES.__doc__ = """
Supported inverse opto-optical transfer functions (OOTFs / OOCFs).
"""


def ootf_inverse(
    value: ArrayLike,
    function: LiteralOOTFInverse | str = "ITU-R BT.2100 PQ",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Apply the specified inverse opto-optical transfer function (OOTF).

    Parameters
    ----------
    value
        Display-linear value.
    function
        Inverse opto-optical transfer function (OOTF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.ootf_inverse_BT2100_HLG`,
        :func:`colour.models.ootf_inverse_BT2100_PQ`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Scene-linear value.

    Examples
    --------
    >>> ootf_inverse(779.988360834115840)  # doctest: +ELLIPSIS
    0.1000000...
    >>> ootf_inverse(  # doctest: +ELLIPSIS
    ...     63.095734448019336, function="ITU-R BT.2100 HLG"
    ... )
    0.1000000...
    """

    function = validate_method(
        function,
        tuple(OOTF_INVERSES),
        '"{0}" inverse "OOTF" is invalid, it must be one of {1}!',
    )

    callable_ = OOTF_INVERSES[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


__all__ += [
    "OOTFS",
    "OOTF_INVERSES",
]
__all__ += [
    "ootf",
    "ootf_inverse",
]
