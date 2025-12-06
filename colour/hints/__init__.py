"""
Annotation Type Hints
=====================

Define the annotation type hints, the module exposes many aliases from
:mod:`typing` and :mod:`numpy.typing` to avoid having to handle multiple
imports.
"""

from __future__ import annotations

import re
import typing
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from os import PathLike
from types import ModuleType
from typing import (  # noqa: UP035
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    NewType,
    NoReturn,
    Protocol,
    Self,
    Set,
    SupportsIndex,
    TextIO,
    Tuple,
    Type,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Annotated",
    "ArrayLike",
    "NDArray",
    "ModuleType",
    "Any",
    "Callable",
    "ClassVar",
    "Dict",
    "Generator",
    "Iterable",
    "Iterator",
    "List",
    "Literal",
    "Mapping",
    "NoReturn",
    "NewType",
    "Protocol",
    "Sequence",
    "PathLike",
    "Set",
    "SupportsIndex",
    "TextIO",
    "Tuple",
    "Type",
    "TypeVar",
    "TypedDict",
    "cast",
    "overload",
    "runtime_checkable",
    "Self",
    "RegexFlag",
    "DTypeInt",
    "DTypeFloat",
    "DTypeReal",
    "DTypeComplex",
    "DTypeBoolean",
    "DType",
    "Real",
    "Dataclass",
    "NDArrayInt",
    "NDArrayFloat",
    "NDArrayReal",
    "NDArrayComplex",
    "NDArrayBoolean",
    "NDArrayStr",
    "Domain1",
    "Domain10",
    "Domain100",
    "Domain360",
    "Domain100_100_360",
    "Range1",
    "Range10",
    "Range100",
    "Range360",
    "Range100_100_360",
    "ProtocolInterpolator",
    "ProtocolExtrapolator",
    "ProtocolLUTSequenceItem",
    "LiteralWarning",
    "LiteralChromaticAdaptationTransform",
    "LiteralColourspaceModel",
    "LiteralRGBColourspace",
    "LiteralLogEncoding",
    "LiteralLogDecoding",
    "LiteralOETF",
    "LiteralOETFInverse",
    "LiteralEOTF",
    "LiteralEOTFInverse",
    "LiteralCCTFEncoding",
    "LiteralCCTFDecoding",
    "LiteralOOTF",
    "LiteralOOTFInverse",
    "LiteralLUTReadMethod",
    "LiteralLUTWriteMethod",
    "LiteralFontScaling",
]

RegexFlag = NewType("RegexFlag", re.RegexFlag)

DTypeInt: TypeAlias = (
    np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
)
DTypeFloat: TypeAlias = np.float16 | np.float32 | np.float64
DTypeReal: TypeAlias = DTypeInt | DTypeFloat
DTypeComplex: TypeAlias = np.complex64 | np.complex128
DTypeBoolean: TypeAlias = np.bool_
DType: TypeAlias = DTypeBoolean | DTypeReal | DTypeComplex

Real: TypeAlias = int | float

# TODO: Revisit to use Protocol.
Dataclass: TypeAlias = Any

NDArrayInt: TypeAlias = NDArray[DTypeInt]
NDArrayFloat: TypeAlias = NDArray[DTypeFloat]
NDArrayReal: TypeAlias = NDArray[DTypeInt | DTypeFloat]
NDArrayComplex: TypeAlias = NDArray[DTypeComplex]
NDArrayBoolean: TypeAlias = NDArray[DTypeBoolean]
NDArrayStr: TypeAlias = NDArray[np.str_]

# Domain-Range Scale Type Aliases
Domain1: TypeAlias = Annotated[ArrayLike, 1]
Domain10: TypeAlias = Annotated[ArrayLike, 10]
Domain100: TypeAlias = Annotated[ArrayLike, 100]
Domain360: TypeAlias = Annotated[ArrayLike, 360]
Domain100_100_360: TypeAlias = Annotated[ArrayLike, (100, 100, 360)]

Range1: TypeAlias = Annotated[NDArrayFloat, 1]
Range10: TypeAlias = Annotated[NDArrayFloat, 10]
Range100: TypeAlias = Annotated[NDArrayFloat, 100]
Range360: TypeAlias = Annotated[NDArrayFloat, 360]
Range100_100_360: TypeAlias = Annotated[NDArrayFloat, (100, 100, 360)]


class ProtocolInterpolator(Protocol):  # noqa: D101  # pragma: no cover
    @property
    def x(self) -> NDArray:  # noqa: D102
        ...

    @x.setter
    def x(self, value: ArrayLike, /) -> None: ...

    @property
    def y(self) -> NDArray:  # noqa: D102
        ...

    @y.setter
    def y(self, value: ArrayLike, /) -> None: ...

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # pragma: no cover

    def __call__(self, x: NDArrayFloat) -> NDArray:  # noqa: D102
        ...  # pragma: no cover


class ProtocolExtrapolator(Protocol):  # noqa: D101  # pragma: no cover
    @property
    def interpolator(self) -> ProtocolInterpolator:  # noqa: D102
        ...

    @interpolator.setter
    def interpolator(self, value: ProtocolInterpolator, /) -> None: ...

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # pragma: no cover

    def __call__(self, x: NDArrayFloat) -> NDArray:  # noqa: D102
        ...  # pragma: no cover


@runtime_checkable
class ProtocolLUTSequenceItem(Protocol):  # noqa: D101  # pragma: no cover
    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: D102
        ...  # pragma: no cover


LiteralWarning = Literal["default", "error", "ignore", "always", "module", "once"]

# NOTE: The following literals are automatically generated by the *invoke*
# *literalise* task. Please do not edit this section manually!

# LITERALISE::BEGIN
LiteralChromaticAdaptationTransform = Literal[
    "Bianco 2010",
    "Bianco PC 2010",
    "Bradford",
    "CAT02",
    "CAT02 Brill 2008",
    "CAT16",
    "CMCCAT2000",
    "CMCCAT97",
    "Fairchild",
    "Sharp",
    "Von Kries",
    "XYZ Scaling",
]

LiteralColourspaceModel = Literal[
    "CAM02LCD",
    "CAM02SCD",
    "CAM02UCS",
    "CAM16LCD",
    "CAM16SCD",
    "CAM16UCS",
    "CIE 1931",
    "CIE 1960 UCS",
    "CIE 1976 UCS",
    "CIE Lab",
    "CIE Luv",
    "CIE UCS",
    "CIE UVW",
    "CIE XYZ",
    "CIE xyY",
    "DIN99",
    "HCL",
    "HSL",
    "HSV",
    "Hunter Lab",
    "Hunter Rdab",
    "ICaCb",
    "ICtCp",
    "IHLS",
    "IPT",
    "IPT Ragoo 2021",
    "IgPgTg",
    "Jzazbz",
    "OSA UCS",
    "Oklab",
    "RGB",
    "YCbCr",
    "YCoCg",
    "Yrg",
    "hdr-CIELAB",
    "hdr-IPT",
    "sUCS",
]

LiteralRGBColourspace = Literal[
    "ACES2065-1",
    "ACEScc",
    "ACEScct",
    "ACEScg",
    "ACESproxy",
    "ARRI Wide Gamut 3",
    "ARRI Wide Gamut 4",
    "Adobe RGB (1998)",
    "Adobe Wide Gamut RGB",
    "Apple RGB",
    "Best RGB",
    "Beta RGB",
    "Blackmagic Wide Gamut",
    "CIE RGB",
    "CIE XYZ-D65 - Scene-referred",
    "Cinema Gamut",
    "ColorMatch RGB",
    "DCDM XYZ",
    "DCI-P3",
    "DCI-P3-P",
    "DJI D-Gamut",
    "DRAGONcolor",
    "DRAGONcolor2",
    "DaVinci Wide Gamut",
    "Display P3",
    "Don RGB 4",
    "EBU Tech. 3213-E",
    "ECI RGB v2",
    "ERIMM RGB",
    "Ekta Space PS 5",
    "F-Gamut",
    "F-Gamut C",
    "FilmLight E-Gamut",
    "FilmLight E-Gamut 2",
    "Gamma 1.8 Encoded Rec.709",
    "Gamma 2.2 Encoded AP1",
    "Gamma 2.2 Encoded AdobeRGB",
    "Gamma 2.2 Encoded Rec.709",
    "ITU-R BT.2020",
    "ITU-R BT.470 - 525",
    "ITU-R BT.470 - 625",
    "ITU-R BT.709",
    "ITU-T H.273 - 22 Unspecified",
    "ITU-T H.273 - Generic Film",
    "Linear AdobeRGB",
    "Linear P3-D65",
    "Linear Rec.2020",
    "Linear Rec.709 (sRGB)",
    "Max RGB",
    "N-Gamut",
    "NTSC (1953)",
    "NTSC (1987)",
    "P3-D65",
    "PLASA ANSI E1.54",
    "Pal/Secam",
    "ProPhoto RGB",
    "Protune Native",
    "REDWideGamutRGB",
    "REDcolor",
    "REDcolor2",
    "REDcolor3",
    "REDcolor4",
    "RIMM RGB",
    "ROMM RGB",
    "Russell RGB",
    "S-Gamut",
    "S-Gamut3",
    "S-Gamut3.Cine",
    "SMPTE 240M",
    "SMPTE C",
    "Sharp RGB",
    "V-Gamut",
    "Venice S-Gamut3",
    "Venice S-Gamut3.Cine",
    "Xtreme RGB",
    "aces",
    "adobe1998",
    "g18_rec709_scene",
    "g22_adobergb_scene",
    "g22_ap1_scene",
    "g22_rec709_scene",
    "lin_adobergb_scene",
    "lin_ap0_scene",
    "lin_ap1_scene",
    "lin_ciexyzd65_scene",
    "lin_p3d65_scene",
    "lin_rec2020_scene",
    "lin_rec709_scene",
    "prophoto",
    "sRGB",
    "sRGB Encoded AP1",
    "sRGB Encoded P3-D65",
    "sRGB Encoded Rec.709 (sRGB)",
    "srgb_ap1_scene",
    "srgb_p3d65_scene",
    "srgb_rec709_scene",
]

LiteralLogEncoding = Literal[
    "ACEScc",
    "ACEScct",
    "ACESproxy",
    "ARRI LogC3",
    "ARRI LogC4",
    "Apple Log Profile",
    "Canon Log",
    "Canon Log 2",
    "Canon Log 3",
    "Cineon",
    "D-Log",
    "ERIMM RGB",
    "F-Log",
    "F-Log2",
    "Filmic Pro 6",
    "L-Log",
    "Log2",
    "Log3G10",
    "Log3G12",
    "Mi-Log",
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
]

LiteralLogDecoding = Literal[
    "ACEScc",
    "ACEScct",
    "ACESproxy",
    "ARRI LogC3",
    "ARRI LogC4",
    "Apple Log Profile",
    "Canon Log",
    "Canon Log 2",
    "Canon Log 3",
    "Cineon",
    "D-Log",
    "ERIMM RGB",
    "F-Log",
    "F-Log2",
    "Filmic Pro 6",
    "L-Log",
    "Log2",
    "Log3G10",
    "Log3G12",
    "Mi-Log",
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
]

LiteralOETF = Literal[
    "ARIB STD-B67",
    "Blackmagic Film Generation 5",
    "DaVinci Intermediate",
    "ITU-R BT.2020",
    "ITU-R BT.2100 HLG",
    "ITU-R BT.2100 PQ",
    "ITU-R BT.601",
    "ITU-R BT.709",
    "ITU-T H.273 IEC 61966-2",
    "ITU-T H.273 Log",
    "ITU-T H.273 Log Sqrt",
    "SMPTE 240M",
]

LiteralOETFInverse = Literal[
    "ARIB STD-B67",
    "Blackmagic Film Generation 5",
    "DaVinci Intermediate",
    "ITU-R BT.2020",
    "ITU-R BT.2100 HLG",
    "ITU-R BT.2100 PQ",
    "ITU-R BT.601",
    "ITU-R BT.709",
    "ITU-T H.273 IEC 61966-2",
    "ITU-T H.273 Log",
    "ITU-T H.273 Log Sqrt",
]

LiteralEOTF = Literal[
    "DCDM",
    "DICOM GSDF",
    "ITU-R BT.1886",
    "ITU-R BT.2100 HLG",
    "ITU-R BT.2100 PQ",
    "ITU-T H.273 ST.428-1",
    "SMPTE 240M",
    "ST 2084",
    "sRGB",
]

LiteralEOTFInverse = Literal[
    "DCDM",
    "DICOM GSDF",
    "ITU-R BT.1886",
    "ITU-R BT.2100 HLG",
    "ITU-R BT.2100 PQ",
    "ITU-T H.273 ST.428-1",
    "ST 2084",
    "sRGB",
]

LiteralCCTFEncoding = Literal[
    "ACEScc",
    "ACEScct",
    "ACESproxy",
    "ARIB STD-B67",
    "ARRI LogC3",
    "ARRI LogC4",
    "Apple Log Profile",
    "Blackmagic Film Generation 5",
    "Canon Log",
    "Canon Log 2",
    "Canon Log 3",
    "Cineon",
    "D-Log",
    "DCDM",
    "DICOM GSDF",
    "DaVinci Intermediate",
    "ERIMM RGB",
    "F-Log",
    "F-Log2",
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
    "ITU-T H.273 IEC 61966-2",
    "ITU-T H.273 Log",
    "ITU-T H.273 Log Sqrt",
    "ITU-T H.273 ST.428-1",
    "L-Log",
    "Log2",
    "Log3G10",
    "Log3G12",
    "Mi-Log",
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
]

LiteralCCTFDecoding = Literal[
    "ACEScc",
    "ACEScct",
    "ACESproxy",
    "ARIB STD-B67",
    "ARRI LogC3",
    "ARRI LogC4",
    "Apple Log Profile",
    "Blackmagic Film Generation 5",
    "Canon Log",
    "Canon Log 2",
    "Canon Log 3",
    "Cineon",
    "D-Log",
    "DCDM",
    "DICOM GSDF",
    "DaVinci Intermediate",
    "ERIMM RGB",
    "F-Log",
    "F-Log2",
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
    "ITU-T H.273 IEC 61966-2",
    "ITU-T H.273 Log",
    "ITU-T H.273 Log Sqrt",
    "ITU-T H.273 ST.428-1",
    "L-Log",
    "Log2",
    "Log3G10",
    "Log3G12",
    "Mi-Log",
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
]

LiteralOOTF = Literal["ITU-R BT.2100 HLG", "ITU-R BT.2100 PQ"]

LiteralOOTFInverse = Literal["ITU-R BT.2100 HLG", "ITU-R BT.2100 PQ"]

LiteralLUTReadMethod = Literal[
    "Cinespace",
    "Iridas Cube",
    "Resolve Cube",
    "Sony SPI1D",
    "Sony SPI3D",
    "Sony SPImtx",
]

LiteralLUTWriteMethod = Literal[
    "Cinespace",
    "Iridas Cube",
    "Resolve Cube",
    "Sony SPI1D",
    "Sony SPI3D",
    "Sony SPImtx",
]

LiteralDeltaEMethod = Literal[
    "CAM02-LCD",
    "CAM02-SCD",
    "CAM02-UCS",
    "CAM16-LCD",
    "CAM16-SCD",
    "CAM16-UCS",
    "CIE 1976",
    "CIE 1994",
    "CIE 2000",
    "CMC",
    "DIN99",
    "HyAB",
    "HyCH",
    "ITP",
    "cie1976",
    "cie1994",
    "cie2000",
]

LiteralFontScaling = Literal[
    "xx-small",
    "x-small",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
    "larger",
    "smaller",
    "xx-small-colour-science",
    "x-small-colour-science",
    "small-colour-science",
    "medium-colour-science",
    "large-colour-science",
    "x-large-colour-science",
    "xx-large-colour-science",
]
# LITERALISE::END


def arraylike(a: ArrayLike) -> NDArray: ...


def number_or_arraylike(a: ArrayLike) -> NDArray: ...


a: DTypeFloat = np.float64(1)
b: float = 1
c: float = 1
d: ArrayLike = [c, c]
e: ArrayLike = d
s_a: Sequence[DTypeFloat] = [a, a]
s_b: Sequence[float] = [b, b]
s_c: Sequence[float] = [c, c]

arraylike(a)
arraylike(b)
arraylike(c)
arraylike(d)
arraylike([d, d])
arraylike(e)
arraylike([e, e])
arraylike(s_a)
arraylike(s_b)
arraylike(s_c)

number_or_arraylike(a)
number_or_arraylike(b)
number_or_arraylike(c)
number_or_arraylike(d)
number_or_arraylike([d, d])
number_or_arraylike(e)
number_or_arraylike([e, e])
number_or_arraylike(s_a)
number_or_arraylike(s_b)
number_or_arraylike(s_c)

np.atleast_1d(a)
np.atleast_1d(b)
np.atleast_1d(c)
np.atleast_1d(arraylike(d))
np.atleast_1d(arraylike([d, d]))
np.atleast_1d(arraylike(e))
np.atleast_1d(arraylike([e, e]))
np.atleast_1d(s_a)
np.atleast_1d(s_b)
np.atleast_1d(s_c)

del a, b, c, d, e, s_a, s_b, s_c

# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
if not typing.TYPE_CHECKING:
    DTypeFloating = DTypeFloat
    DTypeInteger = DTypeInt
    DTypeNumber = DTypeReal
    Boolean = bool
    Floating = float
    Integer = int
    Number = Real
    FloatingOrArrayLike = ArrayLike
    FloatingOrNDArray = NDArrayFloat
