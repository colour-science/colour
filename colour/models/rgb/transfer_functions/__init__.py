from functools import partial

from colour.hints import (
    Any,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    IntegerOrArrayLike,
    IntegerOrNDArray,
    Literal,
    Union,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    filter_kwargs,
    usage_warning,
    validate_method,
)

from .common import CV_range, legal_to_full, full_to_legal
from .gamma import gamma_function
from .aces import (
    log_encoding_ACESproxy,
    log_decoding_ACESproxy,
    log_encoding_ACEScc,
    log_decoding_ACEScc,
    log_encoding_ACEScct,
    log_decoding_ACEScct,
)
from .arib_std_b67 import oetf_ARIBSTDB67, oetf_inverse_ARIBSTDB67
from .arri_alexa_log_c import log_encoding_ALEXALogC, log_decoding_ALEXALogC
from .blackmagic_design import (
    oetf_BlackmagicFilmGeneration5,
    oetf_inverse_BlackmagicFilmGeneration5,
)
from .canon_log import (
    log_encoding_CanonLog,
    log_decoding_CanonLog,
    log_encoding_CanonLog2,
    log_decoding_CanonLog2,
    log_encoding_CanonLog3,
    log_decoding_CanonLog3,
)
from .cineon import log_encoding_Cineon, log_decoding_Cineon
from .davinci_intermediate import (
    oetf_DaVinciIntermediate,
    oetf_inverse_DaVinciIntermediate,
)
from .dcdm import eotf_inverse_DCDM, eotf_DCDM
from .dicom_gsdf import eotf_inverse_DICOMGSDF, eotf_DICOMGSDF
from .dji_dlog import log_encoding_DJIDLog, log_decoding_DJIDLog
from .exponent import exponent_function_basic, exponent_function_monitor_curve
from .filmic_pro import log_encoding_FilmicPro6, log_decoding_FilmicPro6
from .filmlight_tlog import (
    log_encoding_FilmLightTLog,
    log_decoding_FilmLightTLog,
)
from .gopro import log_encoding_Protune, log_decoding_Protune
from .itur_bt_601 import oetf_BT601, oetf_inverse_BT601
from .itur_bt_709 import oetf_BT709, oetf_inverse_BT709
from .itur_bt_1886 import eotf_inverse_BT1886, eotf_BT1886
from .itur_bt_2020 import eotf_inverse_BT2020, eotf_BT2020
from .st_2084 import eotf_inverse_ST2084, eotf_ST2084
from .itur_bt_2100 import (
    oetf_PQ_BT2100,
    oetf_inverse_PQ_BT2100,
    eotf_PQ_BT2100,
    eotf_inverse_PQ_BT2100,
    ootf_PQ_BT2100,
    ootf_inverse_PQ_BT2100,
    oetf_HLG_BT2100,
    oetf_inverse_HLG_BT2100,
    BT2100_HLG_EOTF_METHODS,
    eotf_HLG_BT2100,
    BT2100_HLG_EOTF_INVERSE_METHODS,
    eotf_inverse_HLG_BT2100,
    BT2100_HLG_OOTF_METHODS,
    ootf_HLG_BT2100,
    BT2100_HLG_OOTF_INVERSE_METHODS,
    ootf_inverse_HLG_BT2100,
)
from .linear import linear_function
from .log import (
    logarithmic_function_basic,
    logarithmic_function_quasilog,
    logarithmic_function_camera,
    log_encoding_Log2,
    log_decoding_Log2,
)
from .panalog import log_encoding_Panalog, log_decoding_Panalog
from .panasonic_vlog import log_encoding_VLog, log_decoding_VLog
from .fujifilm_flog import log_encoding_FLog, log_decoding_FLog
from .nikon_nlog import log_encoding_NLog, log_decoding_NLog
from .pivoted_log import log_encoding_PivotedLog, log_decoding_PivotedLog
from .red_log import (
    log_encoding_REDLog,
    log_decoding_REDLog,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
    LOG3G10_ENCODING_METHODS,
    LOG3G10_DECODING_METHODS,
    log_encoding_Log3G10,
    log_decoding_Log3G10,
    log_encoding_Log3G12,
    log_decoding_Log3G12,
)
from .rimm_romm_rgb import (
    cctf_encoding_ROMMRGB,
    cctf_decoding_ROMMRGB,
    cctf_encoding_ProPhotoRGB,
    cctf_decoding_ProPhotoRGB,
    cctf_encoding_RIMMRGB,
    cctf_decoding_RIMMRGB,
    log_encoding_ERIMMRGB,
    log_decoding_ERIMMRGB,
)
from .smpte_240m import oetf_SMPTE240M, eotf_SMPTE240M
from .sony_slog import (
    log_encoding_SLog,
    log_decoding_SLog,
    log_encoding_SLog2,
    log_decoding_SLog2,
    log_encoding_SLog3,
    log_decoding_SLog3,
)
from .srgb import eotf_inverse_sRGB, eotf_sRGB
from .viper_log import log_encoding_ViperLog, log_decoding_ViperLog

__all__ = [
    "CV_range",
    "legal_to_full",
    "full_to_legal",
]
__all__ += [
    "gamma_function",
]
__all__ += [
    "log_encoding_ACESproxy",
    "log_decoding_ACESproxy",
    "log_encoding_ACEScc",
    "log_decoding_ACEScc",
    "log_encoding_ACEScct",
    "log_decoding_ACEScct",
]
__all__ += [
    "oetf_ARIBSTDB67",
    "oetf_inverse_ARIBSTDB67",
]
__all__ += [
    "log_encoding_ALEXALogC",
    "log_decoding_ALEXALogC",
]
__all__ += [
    "oetf_BlackmagicFilmGeneration5",
    "oetf_inverse_BlackmagicFilmGeneration5",
]
__all__ += [
    "log_encoding_CanonLog",
    "log_decoding_CanonLog",
    "log_encoding_CanonLog2",
    "log_decoding_CanonLog2",
    "log_encoding_CanonLog3",
    "log_decoding_CanonLog3",
]
__all__ += [
    "log_encoding_Cineon",
    "log_decoding_Cineon",
]
__all__ += [
    "oetf_DaVinciIntermediate",
    "oetf_inverse_DaVinciIntermediate",
]
__all__ += [
    "eotf_inverse_DCDM",
    "eotf_DCDM",
]
__all__ += [
    "eotf_inverse_DICOMGSDF",
    "eotf_DICOMGSDF",
]
__all__ += [
    "log_encoding_DJIDLog",
    "log_decoding_DJIDLog",
]
__all__ += [
    "exponent_function_basic",
    "exponent_function_monitor_curve",
]
__all__ += [
    "log_encoding_FilmicPro6",
    "log_decoding_FilmicPro6",
]
__all__ += [
    "log_encoding_FilmLightTLog",
    "log_decoding_FilmLightTLog",
]
__all__ += [
    "log_encoding_Protune",
    "log_decoding_Protune",
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
    "eotf_inverse_BT1886",
    "eotf_BT1886",
]
__all__ += [
    "eotf_inverse_BT2020",
    "eotf_BT2020",
]
__all__ += [
    "eotf_inverse_ST2084",
    "eotf_ST2084",
]
__all__ += [
    "oetf_PQ_BT2100",
    "oetf_inverse_PQ_BT2100",
    "eotf_PQ_BT2100",
    "eotf_inverse_PQ_BT2100",
    "ootf_PQ_BT2100",
    "ootf_inverse_PQ_BT2100",
    "oetf_HLG_BT2100",
    "oetf_inverse_HLG_BT2100",
    "BT2100_HLG_EOTF_METHODS",
    "eotf_HLG_BT2100",
    "BT2100_HLG_EOTF_INVERSE_METHODS",
    "eotf_inverse_HLG_BT2100",
    "BT2100_HLG_OOTF_METHODS",
    "ootf_HLG_BT2100",
    "BT2100_HLG_OOTF_INVERSE_METHODS",
    "ootf_inverse_HLG_BT2100",
]
__all__ += [
    "linear_function",
]
__all__ += [
    "logarithmic_function_basic",
    "logarithmic_function_quasilog",
    "logarithmic_function_camera",
    "log_encoding_Log2",
    "log_decoding_Log2",
]
__all__ += [
    "log_encoding_Panalog",
    "log_decoding_Panalog",
]
__all__ += [
    "log_encoding_VLog",
    "log_decoding_VLog",
]
__all__ += [
    "log_encoding_FLog",
    "log_decoding_FLog",
]
__all__ += [
    "log_encoding_NLog",
    "log_decoding_NLog",
]
__all__ += [
    "log_encoding_PivotedLog",
    "log_decoding_PivotedLog",
]
__all__ += [
    "log_encoding_REDLog",
    "log_decoding_REDLog",
    "log_encoding_REDLogFilm",
    "log_decoding_REDLogFilm",
    "LOG3G10_ENCODING_METHODS",
    "LOG3G10_DECODING_METHODS",
    "log_encoding_Log3G10",
    "log_decoding_Log3G10",
    "log_encoding_Log3G12",
    "log_decoding_Log3G12",
]
__all__ += [
    "cctf_encoding_ROMMRGB",
    "cctf_decoding_ROMMRGB",
    "cctf_encoding_ProPhotoRGB",
    "cctf_decoding_ProPhotoRGB",
    "cctf_encoding_RIMMRGB",
    "cctf_decoding_RIMMRGB",
    "log_encoding_ERIMMRGB",
    "log_decoding_ERIMMRGB",
]
__all__ += [
    "oetf_SMPTE240M",
    "eotf_SMPTE240M",
]
__all__ += [
    "log_encoding_SLog",
    "log_decoding_SLog",
    "log_encoding_SLog2",
    "log_decoding_SLog2",
    "log_encoding_SLog3",
    "log_decoding_SLog3",
]
__all__ += [
    "eotf_inverse_sRGB",
    "eotf_sRGB",
]
__all__ += [
    "log_encoding_ViperLog",
    "log_decoding_ViperLog",
]

LOG_ENCODINGS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "ACEScc": log_encoding_ACEScc,
        "ACEScct": log_encoding_ACEScct,
        "ACESproxy": log_encoding_ACESproxy,
        "ALEXA Log C": log_encoding_ALEXALogC,
        "Canon Log 2": log_encoding_CanonLog2,
        "Canon Log 3": log_encoding_CanonLog3,
        "Canon Log": log_encoding_CanonLog,
        "Cineon": log_encoding_Cineon,
        "D-Log": log_encoding_DJIDLog,
        "ERIMM RGB": log_encoding_ERIMMRGB,
        "F-Log": log_encoding_FLog,
        "Filmic Pro 6": log_encoding_FilmicPro6,
        "Log2": log_encoding_Log2,
        "Log3G10": log_encoding_Log3G10,
        "Log3G12": log_encoding_Log3G12,
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
    value: FloatingOrArrayLike,
    function: Union[
        Literal[
            "ACEScc",
            "ACEScct",
            "ACESproxy",
            "ALEXA Log C",
            "Canon Log 2",
            "Canon Log 3",
            "Canon Log",
            "Cineon",
            "D-Log",
            "ERIMM RGB",
            "F-Log",
            "Filmic Pro 6",
            "Log2",
            "Log3G10",
            "Log3G12",
            "N-Log",
            "PLog",
            "Panalog",
            "Protune",
            "REDLog",
            "REDLogFilm",
            "S-Log",
            "S-Log2",
            "S-Log3",
            "T-Log",
            "V-Log",
            "ViperLog",
        ],
        str,
    ] = "Cineon",
    **kwargs: Any
) -> Union[FloatingOrNDArray, IntegerOrNDArray]:
    """
    Encode *scene-referred* exposure values to :math:`R'G'B'` video component
    signal value using given *log* encoding function.

    Parameters
    ----------
    value
        *Scene-referred* exposure values.
    function
        *Log* encoding function.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.log_encoding_ACEScc`,
        :func:`colour.models.log_encoding_ACEScct`,
        :func:`colour.models.log_encoding_ACESproxy`,
        :func:`colour.models.log_encoding_ALEXALogC`,
        :func:`colour.models.log_encoding_CanonLog2`,
        :func:`colour.models.log_encoding_CanonLog3`,
        :func:`colour.models.log_encoding_CanonLog`,
        :func:`colour.models.log_encoding_Cineon`,
        :func:`colour.models.log_encoding_DJIDLog`,
        :func:`colour.models.log_encoding_ERIMMRGB`,
        :func:`colour.models.log_encoding_FLog`,
        :func:`colour.models.log_encoding_FilmicPro6`,
        :func:`colour.models.log_encoding_Log2`,
        :func:`colour.models.log_encoding_Log3G10`,
        :func:`colour.models.log_encoding_Log3G12`,
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
    :class:`numpy.floating` or :class:`numpy.integer` or :class:`numpy.ndarray`
        *Log* values.

    Examples
    --------
    >>> log_encoding(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    >>> log_encoding(0.18, function='ACEScc')  # doctest: +ELLIPSIS
    0.4135884...
    >>> log_encoding(0.18, function='PLog', log_reference=400)
    ... # doctest: +ELLIPSIS
    0.3910068...
    >>> log_encoding(0.18, function='S-Log')  # doctest: +ELLIPSIS
    0.3849708...
    """

    function = validate_method(
        function,
        LOG_ENCODINGS,
        '"{0}" "log" encoding function is invalid, it must be one of {1}!',
    )

    callable_ = LOG_ENCODINGS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


LOG_DECODINGS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "ACEScc": log_decoding_ACEScc,
        "ACEScct": log_decoding_ACEScct,
        "ACESproxy": log_decoding_ACESproxy,
        "ALEXA Log C": log_decoding_ALEXALogC,
        "Canon Log 2": log_decoding_CanonLog2,
        "Canon Log 3": log_decoding_CanonLog3,
        "Canon Log": log_decoding_CanonLog,
        "Cineon": log_decoding_Cineon,
        "D-Log": log_decoding_DJIDLog,
        "ERIMM RGB": log_decoding_ERIMMRGB,
        "F-Log": log_decoding_FLog,
        "Filmic Pro 6": log_decoding_FilmicPro6,
        "Log2": log_decoding_Log2,
        "Log3G10": log_decoding_Log3G10,
        "Log3G12": log_decoding_Log3G12,
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
    value: Union[FloatingOrArrayLike, IntegerOrArrayLike],
    function: Union[
        Literal[
            "ACEScc",
            "ACEScct",
            "ACESproxy",
            "ALEXA Log C",
            "Canon Log 2",
            "Canon Log 3",
            "Canon Log",
            "Cineon",
            "D-Log",
            "ERIMM RGB",
            "F-Log",
            "Filmic Pro 6",
            "Log2",
            "Log3G10",
            "Log3G12",
            "N-Log",
            "PLog",
            "Panalog",
            "Protune",
            "REDLog",
            "REDLogFilm",
            "S-Log",
            "S-Log2",
            "S-Log3",
            "T-Log",
            "V-Log",
            "ViperLog",
        ],
        str,
    ] = "Cineon",
    **kwargs: Any
) -> FloatingOrNDArray:
    """
    Decode :math:`R'G'B'` video component signal value to *scene-referred*
    exposure values using given *log* decoding function.

    Parameters
    ----------
    value
        *Log* values.
    function
        *Log* decoding function.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.log_decoding_ACEScc`,
        :func:`colour.models.log_decoding_ACEScct`,
        :func:`colour.models.log_decoding_ACESproxy`,
        :func:`colour.models.log_decoding_ALEXALogC`,
        :func:`colour.models.log_decoding_CanonLog2`,
        :func:`colour.models.log_decoding_CanonLog3`,
        :func:`colour.models.log_decoding_CanonLog`,
        :func:`colour.models.log_decoding_Cineon`,
        :func:`colour.models.log_decoding_DJIDLog`,
        :func:`colour.models.log_decoding_ERIMMRGB`,
        :func:`colour.models.log_decoding_FLog`,
        :func:`colour.models.log_decoding_FilmicPro6`,
        :func:`colour.models.log_decoding_Log2`,
        :func:`colour.models.log_decoding_Log3G10`,
        :func:`colour.models.log_decoding_Log3G12`,
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
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *Scene-referred* exposure values.

    Examples
    --------
    >>> log_decoding(0.457319613085418)  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding(0.413588402492442, function='ACEScc')
    ... # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding(0.391006842619746, function='PLog', log_reference=400)
    ... # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding(0.376512722254600, function='S-Log')
    ... # doctest: +ELLIPSIS
    0.1...
    """

    function = validate_method(
        function,
        LOG_DECODINGS,
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

OETFS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "ARIB STD-B67": oetf_ARIBSTDB67,
        "Blackmagic Film Generation 5": oetf_BlackmagicFilmGeneration5,
        "DaVinci Intermediate": oetf_DaVinciIntermediate,
        "ITU-R BT.2100 HLG": oetf_HLG_BT2100,
        "ITU-R BT.2100 PQ": oetf_PQ_BT2100,
        "ITU-R BT.601": oetf_BT601,
        "ITU-R BT.709": oetf_BT709,
        "SMPTE 240M": oetf_SMPTE240M,
    }
)
OETFS.__doc__ = """
Supported opto-electrical transfer functions (OETFs / OECFs).
"""


def oetf(
    value: FloatingOrArrayLike,
    function: Union[
        Literal[
            "ARIB STD-B67",
            "Blackmagic Film Generation 5",
            "DaVinci Intermediate",
            "ITU-R BT.2100 HLG",
            "ITU-R BT.2100 PQ",
            "ITU-R BT.601",
            "ITU-R BT.709",
            "SMPTE 240M",
        ],
        str,
    ] = "ITU-R BT.709",
    **kwargs: Any
) -> FloatingOrNDArray:
    """
    Encode estimated tristimulus values in a scene to :math:`R'G'B'` video
    component signal value using given opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    value
        Value.
    function
        Opto-electronic transfer function (OETF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.oetf_ARIBSTDB67`,
        :func:`colour.models.oetf_BlackmagicFilmGeneration5`,
        :func:`colour.models.oetf_DaVinciIntermediate`,
        :func:`colour.models.oetf_HLG_BT2100`,
        :func:`colour.models.oetf_PQ_BT2100`,
        :func:`colour.models.oetf_BT601`,
        :func:`colour.models.oetf_BT709`,
        :func:`colour.models.oetf_SMPTE240M`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        :math:`R'G'B'` video component signal value.

    Examples
    --------
    >>> oetf(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    >>> oetf(0.18, function='ITU-R BT.601')  # doctest: +ELLIPSIS
    0.4090077...
    """

    function = validate_method(
        function, OETFS, '"{0}" "OETF" is invalid, it must be one of {1}!'
    )

    callable_ = OETFS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


OETF_INVERSES: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "ARIB STD-B67": oetf_inverse_ARIBSTDB67,
        "Blackmagic Film Generation 5": oetf_inverse_BlackmagicFilmGeneration5,
        "DaVinci Intermediate": oetf_inverse_DaVinciIntermediate,
        "ITU-R BT.2100 HLG": oetf_inverse_HLG_BT2100,
        "ITU-R BT.2100 PQ": oetf_inverse_PQ_BT2100,
        "ITU-R BT.601": oetf_inverse_BT601,
        "ITU-R BT.709": oetf_inverse_BT709,
    }
)
OETF_INVERSES.__doc__ = """
Supported inverse opto-electrical transfer functions (OETFs / OECFs).
"""


def oetf_inverse(
    value: FloatingOrArrayLike,
    function: Union[
        Literal[
            "ARIB STD-B67",
            "Blackmagic Film Generation 5",
            "DaVinci Intermediate",
            "ITU-R BT.2100 HLG",
            "ITU-R BT.2100 PQ",
            "ITU-R BT.601",
            "ITU-R BT.709",
        ],
        str,
    ] = "ITU-R BT.709",
    **kwargs: Any
) -> FloatingOrNDArray:
    """
    Decode :math:`R'G'B'` video component signal value to tristimulus values
    at the display using given inverse opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    value
        Value.
    function
        Inverse opto-electronic transfer function (OETF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.oetf_inverse_ARIBSTDB67`,
        :func:`colour.models.oetf_inverse_BlackmagicFilmGeneration5`,
        :func:`colour.models.oetf_inverse_DaVinciIntermediate`,
        :func:`colour.models.oetf_inverse_HLG_BT2100`,
        :func:`colour.models.oetf_inverse_PQ_BT2100`,
        :func:`colour.models.oetf_inverse_BT601`,
        :func:`colour.models.oetf_inverse_BT709`},
        See the documentation of the previously listed definitions.


    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Tristimulus values at the display.

    Examples
    --------
    >>> oetf_inverse(0.409007728864150)  # doctest: +ELLIPSIS
    0.1...
    >>> oetf_inverse(  # doctest: +ELLIPSIS
    ...     0.409007728864150, function='ITU-R BT.601')
    0.1...
    """

    function = validate_method(
        function,
        OETF_INVERSES,
        '"{0}" inverse "OETF" is invalid, it must be one of {1}!',
    )

    callable_ = OETF_INVERSES[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


EOTFS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "DCDM": eotf_DCDM,
        "DICOM GSDF": eotf_DICOMGSDF,
        "ITU-R BT.1886": eotf_BT1886,
        "ITU-R BT.2020": eotf_BT2020,
        "ITU-R BT.2100 HLG": eotf_HLG_BT2100,
        "ITU-R BT.2100 PQ": eotf_PQ_BT2100,
        "SMPTE 240M": eotf_SMPTE240M,
        "ST 2084": eotf_ST2084,
        "sRGB": eotf_sRGB,
    }
)
EOTFS.__doc__ = """
Supported electro-optical transfer functions (EOTFs / EOCFs).
"""


def eotf(
    value: Union[FloatingOrArrayLike, IntegerOrArrayLike],
    function: Union[
        Literal[
            "DCDM",
            "DICOM GSDF",
            "ITU-R BT.1886",
            "ITU-R BT.2020",
            "ITU-R BT.2100 HLG",
            "ITU-R BT.2100 PQ",
            "SMPTE 240M",
            "ST 2084",
            "sRGB",
        ],
        str,
    ] = "ITU-R BT.1886",
    **kwargs: Any
) -> FloatingOrNDArray:
    """
    Decode :math:`R'G'B'` video component signal value to tristimulus values
    at the display using given electro-optical transfer function (EOTF).

    Parameters
    ----------
    value
        Value.
    function
        Electro-optical transfer function (EOTF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.eotf_DCDM`,
        :func:`colour.models.eotf_DICOMGSDF`,
        :func:`colour.models.eotf_BT1886`,
        :func:`colour.models.eotf_BT2020`,
        :func:`colour.models.eotf_HLG_BT2100`,
        :func:`colour.models.eotf_PQ_BT2100`,
        :func:`colour.models.eotf_SMPTE240M`,
        :func:`colour.models.eotf_ST2084`,
        :func:`colour.models.eotf_sRGB`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
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

    function = validate_method(
        function, EOTFS, '"{0}" "EOTF" is invalid, it must be one of {1}!'
    )

    callable_ = EOTFS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


EOTF_INVERSES: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "DCDM": eotf_inverse_DCDM,
        "DICOM GSDF": eotf_inverse_DICOMGSDF,
        "ITU-R BT.1886": eotf_inverse_BT1886,
        "ITU-R BT.2020": eotf_inverse_BT2020,
        "ITU-R BT.2100 HLG": eotf_inverse_HLG_BT2100,
        "ITU-R BT.2100 PQ": eotf_inverse_PQ_BT2100,
        "ST 2084": eotf_inverse_ST2084,
        "sRGB": eotf_inverse_sRGB,
    }
)
EOTF_INVERSES.__doc__ = """
Supported inverse electro-optical transfer functions (EOTFs / EOCFs).
"""


def eotf_inverse(
    value: FloatingOrArrayLike,
    function: Union[
        Literal[
            "DCDM",
            "DICOM GSDF",
            "ITU-R BT.1886",
            "ITU-R BT.2020",
            "ITU-R BT.2100 HLG",
            "ITU-R BT.2100 PQ",
            "ST 2084",
            "sRGB",
        ],
        str,
    ] = "ITU-R BT.1886",
    **kwargs
) -> Union[FloatingOrNDArray, IntegerOrNDArray]:
    """
    Encode estimated tristimulus values in a scene to :math:`R'G'B'` video
    component signal value using given inverse electro-optical transfer
    function (EOTF).

    Parameters
    ----------
    value
        Value.
    function
        Inverse electro-optical transfer function (EOTF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.eotf_inverse_DCDM`,
        :func:`colour.models.eotf_inverse_DICOMGSDF`,
        :func:`colour.models.eotf_inverse_BT1886`,
        :func:`colour.models.eotf_inverse_BT2020`,
        :func:`colour.models.eotf_inverse_HLG_BT2100`,
        :func:`colour.models.eotf_inverse_PQ_BT2100`,
        :func:`colour.models.eotf_inverse_ST2084`,
        :func:`colour.models.eotf_inverse_sRGB`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.integer` or :class:`numpy.ndarray`
        :math:`R'G'B'` video component signal value.

    Examples
    --------
    >>> eotf_inverse(0.11699185725296059)  # doctest: +ELLIPSIS
    0.4090077...
    >>> eotf_inverse(  # doctest: +ELLIPSIS
    ...     0.11699185725296059, function='ITU-R BT.1886')
    0.4090077...
    """

    function = validate_method(
        function,
        EOTF_INVERSES,
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

CCTF_ENCODINGS: CaseInsensitiveMapping = CaseInsensitiveMapping(
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
Supported encoding colour component transfer functions (Encoding CCTFs), a
collection of the functions defined by :attr:`colour.LOG_ENCODINGS`,
:attr:`colour.OETFS`, :attr:`colour.EOTF_INVERSES` attributes, the
:func:`colour.models.cctf_encoding_ProPhotoRGB`,
:func:`colour.models.cctf_encoding_RIMMRGB`,
:func:`colour.models.cctf_encoding_ROMMRGB` definitions and 3 gamma encoding
functions (1 / 2.2, 1 / 2.4, 1 / 2.6).

Warnings
--------
For *ITU-R BT.2100*, only the inverse electro-optical transfer functions
(EOTFs / EOCFs) are exposed by this attribute, See the
:attr:`colour.OETFS` attribute for the opto-electronic transfer functions
(OETF).
"""


def cctf_encoding(
    value: FloatingOrArrayLike,
    function: Union[
        Literal[
            "ACEScc",
            "ACEScct",
            "ACESproxy",
            "ALEXA Log C",
            "ARIB STD-B67",
            "Blackmagic Film Generation 5",
            "Canon Log 2",
            "Canon Log 3",
            "Canon Log",
            "Cineon",
            "D-Log",
            "DCDM",
            "DICOM GSDF",
            "DaVinci Intermediate",
            "ERIMM RGB",
            "F-Log",
            "Filmic Pro 6",
            "Gamma 2.2",
            "Gamma 2.4",
            "Gamma 2.6",
            "ITU-R BT.1886",
            "ITU-R BT.2020",
            "ITU-R BT.2100 HLG",
            "ITU-R BT.2100 PQ",
            "ITU-R BT.601",
            "ITU-R BT.709",
            "Log2",
            "Log3G10",
            "Log3G12",
            "N-Log",
            "PLog",
            "Panalog",
            "ProPhoto RGB",
            "Protune",
            "REDLog",
            "REDLogFilm",
            "RIMM RGB",
            "ROMM RGB",
            "S-Log",
            "S-Log2",
            "S-Log3",
            "SMPTE 240M",
            "ST 2084",
            "T-Log",
            "V-Log",
            "ViperLog",
            "sRGB",
        ],
        str,
    ] = "sRGB",
    **kwargs: Any
) -> Union[FloatingOrNDArray, IntegerOrNDArray]:
    """
    Encode linear :math:`RGB` values to non-linear :math:`R'G'B'` values using
    given encoding colour component transfer function (Encoding CCTF).

    Parameters
    ----------
    value
        Linear :math:`RGB` values.
    function
        {:attr:`colour.CCTF_ENCODINGS`},
        Encoding colour component transfer function.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments for the relevant encoding *CCTF* of the
        :attr:`colour.CCTF_ENCODINGS` attribute collection.

    Warnings
    --------
    For *ITU-R BT.2100*, only the inverse electro-optical transfer functions
    (EOTFs / EOCFs) are exposed by this definition, See the
    :func:`colour.oetf` definition for the opto-electronic transfer functions
    (OETF).

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear :math:`R'G'B'` values.

    Examples
    --------
    >>> cctf_encoding(0.18, function='PLog', log_reference=400)
    ... # doctest: +ELLIPSIS
    0.3910068...
    >>> cctf_encoding(0.18, function='ST 2084', L_p=1000)
    ... # doctest: +ELLIPSIS
    0.1820115...
    >>> cctf_encoding(  # doctest: +ELLIPSIS
    ...     0.11699185725296059, function='ITU-R BT.1886')
    0.4090077...
    """

    function = validate_method(
        function,
        CCTF_ENCODINGS,
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


CCTF_DECODINGS: CaseInsensitiveMapping = CaseInsensitiveMapping(
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
Supported decoding colour component transfer functions (Decoding CCTFs), a
collection of the functions defined by :attr:`colour.LOG_DECODINGS`,
:attr:`colour.EOTFS`, :attr:`colour.OETF_INVERSES` attributes, the
:func:`colour.models.cctf_decoding_ProPhotoRGB`,
:func:`colour.models.cctf_decoding_RIMMRGB`,
:func:`colour.models.cctf_decoding_ROMMRGB` definitions and 3 gamma decoding
functions (2.2, 2.4, 2.6).

Warnings
--------
For *ITU-R BT.2100*, only the electro-optical transfer functions
(EOTFs / EOCFs) are exposed by this attribute, See the
:attr:`colour.OETF_INVERSES` attribute for the inverse opto-electronic
transfer functions (OETF).

Notes
-----
-   The order by which this attribute is defined and updated is critically
    important to ensure that *ITU-R BT.2100* definitions are reciprocal.
"""


def cctf_decoding(
    value: Union[FloatingOrArrayLike, IntegerOrArrayLike],
    function: Union[
        Literal[
            "ACEScc",
            "ACEScct",
            "ACESproxy",
            "ALEXA Log C",
            "ARIB STD-B67",
            "Blackmagic Film Generation 5",
            "Canon Log 2",
            "Canon Log 3",
            "Canon Log",
            "Cineon",
            "D-Log",
            "DCDM",
            "DICOM GSDF",
            "DaVinci Intermediate",
            "ERIMM RGB",
            "F-Log",
            "Filmic Pro 6",
            "Gamma 2.2",
            "Gamma 2.4",
            "Gamma 2.6",
            "ITU-R BT.1886",
            "ITU-R BT.2020",
            "ITU-R BT.2100 HLG",
            "ITU-R BT.2100 PQ",
            "ITU-R BT.601",
            "ITU-R BT.709",
            "Log2",
            "Log3G10",
            "Log3G12",
            "N-Log",
            "PLog",
            "Panalog",
            "ProPhoto RGB",
            "Protune",
            "REDLog",
            "REDLogFilm",
            "RIMM RGB",
            "ROMM RGB",
            "S-Log",
            "S-Log2",
            "S-Log3",
            "SMPTE 240M",
            "ST 2084",
            "T-Log",
            "V-Log",
            "ViperLog",
            "sRGB",
        ],
        str,
    ] = "sRGB",
    **kwargs: Any
) -> FloatingOrNDArray:
    """
    Decode non-linear :math:`R'G'B'` values to linear :math:`RGB` values using
    given decoding colour component transfer function (Decoding CCTF).

    Parameters
    ----------
    value
        Non-linear :math:`R'G'B'` values.
    function
        {:attr:`colour.CCTF_DECODINGS`},
        Decoding colour component transfer function.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments for the relevant decoding *CCTF* of the
        :attr:`colour.CCTF_DECODINGS` attribute collection.

    Warnings
    --------
    For *ITU-R BT.2100*, only the electro-optical transfer functions
    (EOTFs / EOCFs) are exposed by this definition, See the
    :func:`colour.oetf_inverse` definition for the inverse opto-electronic
    transfer functions (OETF).

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear :math:`RGB` values.

    Examples
    --------
    >>> cctf_decoding(0.391006842619746, function='PLog', log_reference=400)
    ... # doctest: +ELLIPSIS
    0.1...
    >>> cctf_decoding(0.182011532850008, function='ST 2084', L_p=1000)
    ... # doctest: +ELLIPSIS
    0.1...
    >>> cctf_decoding(  # doctest: +ELLIPSIS
    ...     0.461356129500442, function='ITU-R BT.1886')
    0.1...
    """

    function = validate_method(
        function,
        CCTF_DECODINGS,
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

OOTFS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "ITU-R BT.2100 HLG": ootf_HLG_BT2100,
        "ITU-R BT.2100 PQ": ootf_PQ_BT2100,
    }
)
OOTFS.__doc__ = """
Supported opto-optical transfer functions (OOTFs / OOCFs).
"""


def ootf(
    value: FloatingOrArrayLike,
    function: Union[
        Literal["ITU-R BT.2100 HLG", "ITU-R BT.2100 PQ"], str
    ] = "ITU-R BT.2100 PQ",
    **kwargs: Any
) -> FloatingOrNDArray:
    """
    Map relative scene linear light to display linear light using given
    opto-optical transfer function (OOTF / OOCF).

    Parameters
    ----------
    value
        Value.
    function
        Opto-optical transfer function (OOTF / OOCF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.ootf_HLG_BT2100`,
        :func:`colour.models.ootf_PQ_BT2100`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Luminance of a displayed linear component.

    Examples
    --------
    >>> ootf(0.1)  # doctest: +ELLIPSIS
    779.9883608...
    >>> ootf(0.1, function='ITU-R BT.2100 HLG')  # doctest: +ELLIPSIS
    63.0957344...
    """

    function = validate_method(
        function, OOTFS, '"{0}" "OOTF" is invalid, it must be one of {1}!'
    )

    callable_ = OOTFS[function]

    return callable_(value, **filter_kwargs(callable_, **kwargs))


OOTF_INVERSES: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "ITU-R BT.2100 HLG": ootf_inverse_HLG_BT2100,
        "ITU-R BT.2100 PQ": ootf_inverse_PQ_BT2100,
    }
)
OOTF_INVERSES.__doc__ = """
Supported inverse opto-optical transfer functions (OOTFs / OOCFs).
"""


def ootf_inverse(
    value: FloatingOrArrayLike,
    function: Union[
        Literal["ITU-R BT.2100 HLG", "ITU-R BT.2100 PQ"], str
    ] = "ITU-R BT.2100 PQ",
    **kwargs: Any
) -> FloatingOrNDArray:
    """
    Map relative display linear light to scene linear light using given
    inverse opto-optical transfer function (OOTF / OOCF).

    Parameters
    ----------
    value
        Value.
    function
        Inverse opto-optical transfer function (OOTF / OOCF).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.models.ootf_inverse_HLG_BT2100`,
        :func:`colour.models.ootf_inverse_PQ_BT2100`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Luminance of scene linear light.

    Examples
    --------
    >>> ootf_inverse(779.988360834115840)  # doctest: +ELLIPSIS
    0.1000000...
    >>> ootf_inverse(  # doctest: +ELLIPSIS
    ...     63.095734448019336, function='ITU-R BT.2100 HLG')
    0.1000000...
    """

    function = validate_method(
        function,
        OOTF_INVERSES,
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
