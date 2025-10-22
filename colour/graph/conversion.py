"""
Automatic Colour Conversion Graph
=================================

Define the automatic colour conversion graph objects.

-   :func:`colour.describe_conversion_path`
-   :func:`colour.convert`
"""

from __future__ import annotations

import inspect
import itertools
import re
import sys
import textwrap
import typing
from copy import copy
from dataclasses import dataclass
from functools import partial
from pprint import pformat

import numpy as np

import colour
import colour.models
from colour.appearance import (
    CAM16_to_XYZ,
    CAM_Specification_CAM16,
    CAM_Specification_CIECAM02,
    CAM_Specification_CIECAM16,
    CAM_Specification_Hellwig2022,
    CAM_Specification_sCAM,
    CIECAM02_to_XYZ,
    CIECAM16_to_XYZ,
    Hellwig2022_to_XYZ,
    Kim2009_to_XYZ,
    XYZ_to_ATD95,
    XYZ_to_CAM16,
    XYZ_to_CIECAM02,
    XYZ_to_CIECAM16,
    XYZ_to_Hellwig2022,
    XYZ_to_Hunt,
    XYZ_to_Kim2009,
    XYZ_to_LLAB,
    XYZ_to_Nayatani95,
    XYZ_to_RLAB,
    XYZ_to_sCAM,
    XYZ_to_ZCAM,
    ZCAM_to_XYZ,
    sCAM_to_XYZ,
)
from colour.appearance.ciecam02 import CAM_KWARGS_CIECAM02_sRGB
from colour.colorimetry import (
    CCS_ILLUMINANTS,
    TVS_ILLUMINANTS_HUNTERLAB,
    colorimetric_purity,
    complementary_wavelength,
    dominant_wavelength,
    excitation_purity,
    lightness,
    luminance,
    luminous_efficacy,
    luminous_efficiency,
    luminous_flux,
    sd_to_XYZ,
    wavelength_to_XYZ,
    whiteness,
    yellowness,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Callable,
        List,
        Literal,
    )

from colour.hints import NDArrayFloat, cast
from colour.models import (
    COLOURSPACE_MODELS_POLAR_CONVERSIONS,
    CAM02LCD_to_JMh_CIECAM02,
    CAM02SCD_to_JMh_CIECAM02,
    CAM02UCS_to_JMh_CIECAM02,
    CAM16LCD_to_JMh_CAM16,
    CAM16SCD_to_JMh_CAM16,
    CAM16UCS_to_JMh_CAM16,
    CIE1960UCS_to_XYZ,
    CIE1976UCS_to_XYZ,
    CMY_to_CMYK,
    CMY_to_RGB,
    CMYK_to_CMY,
    DIN99_to_XYZ,
    HCL_to_RGB,
    HSL_to_RGB,
    HSV_to_RGB,
    Hunter_Lab_to_XYZ,
    Hunter_Rdab_to_XYZ,
    ICaCb_to_XYZ,
    ICtCp_to_XYZ,
    IgPgTg_to_XYZ,
    IHLS_to_RGB,
    IPT_Ragoo2021_to_XYZ,
    IPT_to_XYZ,
    JMh_CAM16_to_CAM16LCD,
    JMh_CAM16_to_CAM16SCD,
    JMh_CAM16_to_CAM16UCS,
    JMh_CIECAM02_to_CAM02LCD,
    JMh_CIECAM02_to_CAM02SCD,
    JMh_CIECAM02_to_CAM02UCS,
    Jzazbz_to_XYZ,
    Lab_to_XYZ,
    Luv_to_uv,
    Luv_to_XYZ,
    Luv_uv_to_xy,
    Oklab_to_XYZ,
    OSA_UCS_to_XYZ,
    Prismatic_to_RGB,
    ProLab_to_XYZ,
    RGB_Colourspace,
    RGB_COLOURSPACE_sRGB,
    RGB_luminance,
    RGB_to_CMY,
    RGB_to_HCL,
    RGB_to_HSL,
    RGB_to_HSV,
    RGB_to_IHLS,
    RGB_to_Prismatic,
    RGB_to_RGB,
    RGB_to_XYZ,
    RGB_to_YCbCr,
    RGB_to_YcCbcCrc,
    RGB_to_YCoCg,
    UCS_to_uv,
    UCS_to_XYZ,
    UCS_uv_to_xy,
    UVW_to_XYZ,
    XYZ_to_CIE1960UCS,
    XYZ_to_CIE1976UCS,
    XYZ_to_DIN99,
    XYZ_to_hdr_CIELab,
    XYZ_to_hdr_IPT,
    XYZ_to_Hunter_Lab,
    XYZ_to_Hunter_Rdab,
    XYZ_to_ICaCb,
    XYZ_to_ICtCp,
    XYZ_to_IgPgTg,
    XYZ_to_IPT,
    XYZ_to_IPT_Ragoo2021,
    XYZ_to_Jzazbz,
    XYZ_to_Lab,
    XYZ_to_Luv,
    XYZ_to_Oklab,
    XYZ_to_OSA_UCS,
    XYZ_to_ProLab,
    XYZ_to_RGB,
    XYZ_to_sRGB,
    XYZ_to_sUCS,
    XYZ_to_UCS,
    XYZ_to_UVW,
    XYZ_to_xy,
    XYZ_to_xyY,
    XYZ_to_Yrg,
    YCbCr_to_RGB,
    YcCbcCrc_to_RGB,
    YCoCg_to_RGB,
    Yrg_to_XYZ,
    cctf_decoding,
    cctf_encoding,
    hdr_CIELab_to_XYZ,
    hdr_IPT_to_XYZ,
    sRGB_to_XYZ,
    sUCS_to_XYZ,
    uv_to_Luv,
    uv_to_UCS,
    xy_to_Luv_uv,
    xy_to_UCS_uv,
    xy_to_xyY,
    xy_to_XYZ,
    xyY_to_xy,
    xyY_to_XYZ,
)
from colour.notation import (
    HEX_to_RGB,
    RGB_to_HEX,
    keyword_to_RGB_CSSColor3,
    munsell_colour_to_xyY,
    munsell_value,
    xyY_to_munsell_colour,
)
from colour.quality import colour_quality_scale, colour_rendering_index
from colour.recovery import XYZ_to_sd
from colour.temperature import CCT_to_mired, CCT_to_uv, mired_to_CCT, uv_to_CCT
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    filter_kwargs,
    message_box,
    required,
    tsplit,
    tstack,
    validate_method,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Conversion_Specification",
    "CIECAM02_to_JMh_CIECAM02",
    "JMh_CIECAM02_to_CIECAM02",
    "CAM16_to_JMh_CAM16",
    "JMh_CAM16_to_CAM16",
    "Hellwig2022_to_JMh_Hellwig2022",
    "JMh_Hellwig2022_to_Hellwig2022",
    "sCAM_to_JMh_sCAM",
    "JMh_sCAM_to_sCAM",
    "XYZ_to_luminance",
    "RGB_luminance_to_RGB",
    "CCT_D_uv_to_mired",
    "mired_to_CCT_D_uv",
    "CONVERSION_SPECIFICATIONS_DATA",
    "CONVERSION_GRAPH_NODE_LABELS",
    "CONVERSION_SPECIFICATIONS",
    "CONVERSION_GRAPH",
    "describe_conversion_path",
    "convert",
]


@dataclass(frozen=True)
class Conversion_Specification:
    """
    Define a conversion specification for the *Colour* graph used in automatic
    colour space conversions.

    The specification describes the relationship between two nodes (colour
    spaces or representations) and the transformation function that connects
    them within the conversion graph.

    Parameters
    ----------
    source
        Source node in the graph.
    target
        Target node in the graph.
    conversion_function
        Callable converting from the ``source`` node to the ``target`` node.
    to_reference
        Tuple of scale factors for converting from domain-range scale '1' to
        'reference' scale for each component of the output. For example,
        100 for CIE Lab or (100, 100, 360) for CIE LCHab.
        None if the conversion output doesn't support automatic rescaling.
    """

    source: str
    target: str
    conversion_function: Callable
    to_reference: tuple[float, ...] | None

    def __post_init__(self) -> None:
        """
        Post-initialise the class.

        Convert the source and target attribute values to lowercase for
        consistent case-insensitive comparisons.
        """

        object.__setattr__(self, "source", self.source.lower())
        object.__setattr__(self, "target", self.target.lower())


def CIECAM02_to_JMh_CIECAM02(
    specification: CAM_Specification_CIECAM02,
) -> NDArrayFloat:
    """
    Convert from *CIECAM02* specification to *CIECAM02* :math:`JMh`
    correlates.

    Parameters
    ----------
    specification
        *CIECAM02* colour appearance model specification.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIECAM02* :math:`JMh` correlates.

    Examples
    --------
    >>> specification = CAM_Specification_CIECAM02(
    ...     J=41.731091132513917, M=0.108842175669226, h=219.048432658311780
    ... )
    >>> CIECAM02_to_JMh_CIECAM02(specification)  # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    return tstack(
        [
            cast("NDArrayFloat", specification.J),
            cast("NDArrayFloat", specification.M),
            cast("NDArrayFloat", specification.h),
        ]
    )


def JMh_CIECAM02_to_CIECAM02(JMh: ArrayLike) -> CAM_Specification_CIECAM02:
    """
    Convert from *CIECAM02* :math:`JMh` correlates to *CIECAM02*
    specification.

    Parameters
    ----------
    JMh
        *CIECAM02* :math:`JMh` correlates.

    Returns
    -------
    :class:`colour.CAM_Specification_CIECAM02`
        *CIECAM02* colour appearance model specification.

    Examples
    --------
    >>> import numpy as np
    >>> JMh = np.array([4.17310911e01, 1.08842176e-01, 2.19048433e02])
    >>> JMh_CIECAM02_to_CIECAM02(JMh)  # doctest: +ELLIPSIS
    CAM_Specification_CIECAM02(J=41.7310911..., C=None, h=219.0484329..., \
s=None, Q=None, M=0.1088421..., H=None, HC=None)
    """

    J, M, h = tsplit(JMh)

    return CAM_Specification_CIECAM02(J=J, M=M, h=h)


def CAM16_to_JMh_CAM16(specification: CAM_Specification_CAM16) -> NDArrayFloat:
    """
    Convert from *CAM16* specification to *CAM16* :math:`JMh` correlates.

    Parameters
    ----------
    specification
        *CAM16* colour appearance model specification.

    Returns
    -------
    :class:`numpy.ndarray`
        *CAM16* :math:`JMh` correlates.

    Examples
    --------
    >>> specification = CAM_Specification_CAM16(
    ...     J=41.731207905126638, M=0.107436772335905, h=217.067959767393010
    ... )
    >>> CAM16_to_JMh_CAM16(specification)  # doctest: +ELLIPSIS
    array([  4.1731207...e+01,   1.0743677...e-01,   2.1706796...e+02])
    """

    return tstack([specification.J, specification.M, specification.h])  # pyright: ignore


def JMh_CAM16_to_CAM16(JMh: ArrayLike) -> CAM_Specification_CAM16:
    """
    Convert from *CAM16* :math:`JMh` correlates to *CAM16* specification.

    Parameters
    ----------
    JMh
        *CAM16* :math:`JMh` correlates.

    Returns
    -------
    :class:`colour.CAM_Specification_CAM16`
        *CAM16* colour appearance model specification.

    Examples
    --------
    >>> import numpy as np
    >>> JMh = np.array([4.17312079e01, 1.07436772e-01, 2.17067960e02])
    >>> JMh_CAM16_to_CAM16(JMh)  # doctest: +ELLIPSIS
    CAM_Specification_CAM16(J=41.7312079..., C=None, h=217.06796..., s=None, \
Q=None, M=0.1074367..., H=None, HC=None)
    """

    J, M, h = tsplit(JMh)

    return CAM_Specification_CAM16(J=J, M=M, h=h)


def CIECAM16_to_JMh_CIECAM16(specification: CAM_Specification_CIECAM16) -> NDArrayFloat:
    """
    Convert from *CIECAM16* specification to *CIECAM16* :math:`JMh`
    correlates.

    Parameters
    ----------
    specification
        *CIECAM16* colour appearance model specification.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIECAM16* :math:`JMh` correlates.

    Examples
    --------
    >>> specification = CAM_Specification_CIECAM16(
    ...     J=41.731207905126638, M=0.107436772335905, h=217.067959767393010
    ... )
    >>> CIECAM16_to_JMh_CIECAM16(specification)  # doctest: +ELLIPSIS
    array([  4.1731207...e+01,   1.0743677...e-01,   2.1706796...e+02])
    """

    return tstack([specification.J, specification.M, specification.h])  # pyright: ignore


def JMh_CIECAM16_to_CIECAM16(JMh: ArrayLike) -> CAM_Specification_CIECAM16:
    """
    Convert from *CIECAM16* :math:`JMh` correlates to *CIECAM16*
    specification.

    Parameters
    ----------
    JMh
        *CIECAM16* :math:`JMh` correlates.

    Returns
    -------
    :class:`colour.CAM_Specification_CIECAM16`
        *CIECAM16* colour appearance model specification.

    Examples
    --------
    >>> import numpy as np
    >>> JMh = np.array([4.17312079e01, 1.07436772e-01, 2.17067960e02])
    >>> JMh_CIECAM16_to_CIECAM16(JMh)  # doctest: +ELLIPSIS
    CAM_Specification_CIECAM16(J=41.7312079..., C=None, h=217.06796..., \
s=None, Q=None, M=0.1074367..., H=None, HC=None)
    """

    J, M, h = tsplit(JMh)

    return CAM_Specification_CIECAM16(J=J, M=M, h=h)


def Hellwig2022_to_JMh_Hellwig2022(
    specification: CAM_Specification_Hellwig2022,
) -> NDArrayFloat:
    """
    Convert from *Hellwig and Fairchild (2022)* specification to
    *Hellwig and Fairchild (2022)* :math:`JMh` correlates.

    Parameters
    ----------
    specification
        *Hellwig and Fairchild (2022)* colour appearance model
        specification.

    Returns
    -------
    :class:`numpy.ndarray`
        *Hellwig and Fairchild (2022)* :math:`JMh` correlates.

    Examples
    --------
    >>> specification = CAM_Specification_Hellwig2022(
    ...     J=41.731207905126638, M=0.029382869535427687, h=217.06795976739301
    ... )
    >>> Hellwig2022_to_JMh_Hellwig2022(specification)  # doctest: +ELLIPSIS
    array([  4.1731207...e+01,   2.9382869...e-02,   2.1706796...e+02])
    """

    return tstack([specification.J, specification.M, specification.h])  # pyright: ignore


def JMh_Hellwig2022_to_Hellwig2022(
    JMh: ArrayLike,
) -> CAM_Specification_Hellwig2022:
    """
    Convert from *Hellwig and Fairchild (2022)* :math:`JMh` correlates to
    *Hellwig and Fairchild (2022)* specification.

    Parameters
    ----------
    JMh
        *Hellwig and Fairchild (2022)* :math:`JMh` correlates.

    Returns
    -------
    :class:`colour.CAM_Specification_Hellwig2022`
        *Hellwig and Fairchild (2022)* colour appearance model specification.

    Examples
    --------
    >>> import numpy as np
    >>> JMh = np.array([4.17312079e01, 2.93828695e-02, 2.17067960e02])
    >>> JMh_Hellwig2022_to_Hellwig2022(JMh)  # doctest: +ELLIPSIS
    CAM_Specification_Hellwig2022(J=41.7312079..., C=None, h=217.06796, \
s=None, Q=None, M=0.0293828..., H=None, HC=None, J_HK=None, Q_HK=None)
    """

    J, M, h = tsplit(JMh)

    return CAM_Specification_Hellwig2022(J=J, M=M, h=h)


def sCAM_to_JMh_sCAM(specification: CAM_Specification_sCAM) -> NDArrayFloat:
    """
    Convert from *sCAM* specification to *sCAM* :math:`JMh` correlates.

    Parameters
    ----------
    specification
        *sCAM* colour appearance model specification.

    Returns
    -------
    :class:`numpy.ndarray`
        *sCAM* :math:`JMh` correlates.

    Examples
    --------
    >>> specification = CAM_Specification_sCAM(
    ...     J=41.731207905126638, M=0.107436772335905, h=217.067959767393010
    ... )
    >>> sCAM_to_JMh_sCAM(specification)  # doctest: +ELLIPSIS
    array([  4.1731207...e+01,   1.0743677...e-01,   2.1706796...e+02])
    """

    return tstack([specification.J, specification.M, specification.h])  # pyright: ignore


def JMh_sCAM_to_sCAM(JMh: ArrayLike) -> CAM_Specification_sCAM:
    """
    Convert from *sCAM* :math:`JMh` correlates to *sCAM* specification.

    Parameters
    ----------
    JMh
        *sCAM* :math:`JMh` correlates.

    Returns
    -------
    :class:`colour.CAM_Specification_sCAM`
        *sCAM* colour appearance model specification.

    Examples
    --------
    >>> import numpy as np
    >>> JMh = np.array([4.17312079e01, 1.07436772e-01, 2.17067960e02])
    >>> JMh_sCAM_to_sCAM(JMh)  # doctest: +ELLIPSIS
    CAM_Specification_sCAM(J=41.7312079..., C=None, h=217.06796..., Q=None, \
M=0.1074367..., H=None, HC=None, V=None, K=None, W=None, D=None)
    """

    J, M, h = tsplit(JMh)

    return CAM_Specification_sCAM(J=J, M=M, h=h)


def XYZ_to_luminance(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert specified *CIE XYZ* tristimulus values to *luminance* :math:`Y`.

    Extract the Y component from *CIE XYZ* tristimulus values, which
    represents the *luminance* of the colour stimulus.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y`.

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_luminance(XYZ)  # doctest: +ELLIPSIS
    0.1219722...
    """

    _X, Y, _Z = tsplit(XYZ)

    return Y


def RGB_luminance_to_RGB(Y: ArrayLike) -> NDArrayFloat:
    """
    Convert from *luminance* :math:`Y` to *RGB*.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB*.

    Examples
    --------
    >>> RGB_luminance_to_RGB(0.123014562384318)  # doctest: +ELLIPSIS
    array([ 0.1230145...,  0.1230145...,  0.1230145...])
    """

    Y = as_float_array(Y)

    return tstack([Y, Y, Y])


def CCT_D_uv_to_mired(CCT_D_uv: ArrayLike) -> NDArrayFloat:
    """
    Convert correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` to micro reciprocal degree (mired).

    Parameters
    ----------
    CCT_D_uv
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    Returns
    -------
    :class:`numpy.ndarray`
        Micro reciprocal degree (mired).

    Examples
    --------
    >>> CCT_D_uv = np.array([6500.0081378199056, 0.008333331244225])
    >>> CCT_D_uv_to_mired(CCT_D_uv)  # doctest: +ELLIPSIS
    153.8459612...
    """

    CCT, _D_uv = tsplit(CCT_D_uv)

    return CCT_to_mired(CCT)


def mired_to_CCT_D_uv(mired: ArrayLike) -> NDArrayFloat:
    """
    Convert specified micro reciprocal degree (mired) to correlated colour
    temperature :math:`T_{cp}` and :math:`\\Delta_{uv}`.

    Parameters
    ----------
    mired
        Micro reciprocal degree (mired).

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    Examples
    --------
    >>> mired = 153.84596123527297
    >>> mired_to_CCT_D_uv(mired)  # doctest: +ELLIPSIS
    array([ 6500.0081378...,     0.        ])
    """

    mired = as_float_array(mired)

    return tstack([mired_to_CCT(mired), zeros(mired.shape)])


_ILLUMINANT_DEFAULT: str = "D65"
"""Default automatic colour conversion graph illuminant name."""

_CCS_ILLUMINANT_DEFAULT: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][_ILLUMINANT_DEFAULT]
"""
Default automatic colour conversion graph illuminant *CIE xy* chromaticity
coordinates.
"""

_TVS_ILLUMINANT_DEFAULT: NDArrayFloat = xy_to_XYZ(_CCS_ILLUMINANT_DEFAULT)
"""
Default automatic colour conversion graph illuminant *CIE XYZ* tristimulus
values.
"""

_RGB_COLOURSPACE_DEFAULT: RGB_Colourspace = RGB_COLOURSPACE_sRGB
"""Default automatic colour conversion graph *RGB* colourspace."""

_CAM_KWARGS_CIECAM02_sRGB: dict = CAM_KWARGS_CIECAM02_sRGB.copy()
"""
Default parameter values for the *CIECAM02* colour appearance model usage in
the context of *sRGB*.

Warnings
--------
The *CIE XYZ* tristimulus values of reference white :math:`XYZ_w` is adjusted
for the domain-range scale **'1'**.
"""

_CAM_KWARGS_CIECAM02_sRGB["XYZ_w"] = _CAM_KWARGS_CIECAM02_sRGB["XYZ_w"] / 100

# fmt: off
# ruff: noqa: E501
CONVERSION_SPECIFICATIONS_DATA: List[tuple] = [
    # Colorimetry
    ("Spectral Distribution", "CIE XYZ", sd_to_XYZ, 100,),
    ("CIE XYZ", "Spectral Distribution", XYZ_to_sd, None,),
    ("Spectral Distribution", "Luminous Flux", luminous_flux, 1,),
    ("Spectral Distribution", "Luminous Efficiency", luminous_efficiency, 1,),
    ("Spectral Distribution", "Luminous Efficacy", luminous_efficacy, 1,),
    ("CIE XYZ", "Luminance", XYZ_to_luminance, 1,),
    ("Luminance", "Lightness", lightness, 100,),
    ("Lightness", "Luminance", luminance, 1,),
    ("CIE XYZ", "Whiteness", partial(whiteness, XYZ_0=_TVS_ILLUMINANT_DEFAULT), 100,),
    ("CIE XYZ", "Yellowness", yellowness, 100,),
    ("CIE xy", "Colorimetric Purity", partial(colorimetric_purity, xy_n=_CCS_ILLUMINANT_DEFAULT), None,),
    ("CIE xy", "Complementary Wavelength", partial(complementary_wavelength, xy_n=_CCS_ILLUMINANT_DEFAULT), None,),
    ("CIE xy", "Dominant Wavelength", partial(dominant_wavelength, xy_n=_CCS_ILLUMINANT_DEFAULT), None,),
    ("CIE xy", "Excitation Purity", partial(excitation_purity, xy_n=_CCS_ILLUMINANT_DEFAULT), None,),
    ("Wavelength", "CIE XYZ", wavelength_to_XYZ, 1,),
    # Colour Models
    ("CIE XYZ", "CIE xyY", XYZ_to_xyY, 1,),
    ("CIE xyY", "CIE XYZ", xyY_to_XYZ, 1,),
    ("CIE xyY", "CIE xy", xyY_to_xy, None,),
    ("CIE xy", "CIE xyY", xy_to_xyY, 1,),
    ("CIE XYZ", "CIE xy", XYZ_to_xy, None,),
    ("CIE xy", "CIE XYZ", xy_to_XYZ, 1,),
    ("CIE XYZ", "CIE Lab", XYZ_to_Lab, 100,),
    ("CIE Lab", "CIE XYZ", Lab_to_XYZ, 1,),
    ("CIE XYZ", "CIE Luv", XYZ_to_Luv, 100,),
    ("CIE Luv", "CIE XYZ", Luv_to_XYZ, 1,),
    ("CIE Luv", "CIE Luv uv", Luv_to_uv, None,),
    ("CIE Luv uv", "CIE Luv", uv_to_Luv, 1,),
    ("CIE Luv uv", "CIE xy", Luv_uv_to_xy, None,),
    ("CIE xy", "CIE Luv uv", xy_to_Luv_uv, 1,),
    ("CIE XYZ", "CIE UCS", XYZ_to_UCS, 1,),
    ("CIE UCS", "CIE XYZ", UCS_to_XYZ, 1,),
    ("CIE UCS", "CIE UCS uv", UCS_to_uv, None,),
    ("CIE UCS uv", "CIE UCS", uv_to_UCS, 1,),
    ("CIE UCS uv", "CIE xy", UCS_uv_to_xy, None,),
    ("CIE xy", "CIE UCS uv", xy_to_UCS_uv, None,),
    ("CIE XYZ", "CIE UVW", XYZ_to_UVW, 100,),
    ("CIE UVW", "CIE XYZ", UVW_to_XYZ, 1,),
    ("CIE XYZ", "DIN99", XYZ_to_DIN99, 100,),
    ("DIN99", "CIE XYZ", DIN99_to_XYZ, 1,),
    ("CIE XYZ", "hdr-CIELAB", XYZ_to_hdr_CIELab, 100,),
    ("hdr-CIELAB", "CIE XYZ", hdr_CIELab_to_XYZ, 1,),
    ("CIE XYZ", "Hunter Lab", partial(XYZ_to_Hunter_Lab, XYZ_n=TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]["D65"].XYZ_n / 100), 100,),
    ("Hunter Lab", "CIE XYZ", partial(Hunter_Lab_to_XYZ, XYZ_n=TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]["D65"].XYZ_n / 100), 1,),
    ("CIE XYZ", "Hunter Rdab", partial(XYZ_to_Hunter_Rdab, XYZ_n=TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]["D65"].XYZ_n / 100), 100,),
    ("Hunter Rdab", "CIE XYZ", partial(Hunter_Rdab_to_XYZ, XYZ_n=TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]["D65"].XYZ_n / 100), 1,),
    ("CIE XYZ", "ICaCb", XYZ_to_ICaCb, 1,),
    ("ICaCb", "CIE XYZ", ICaCb_to_XYZ, 1,),
    ("CIE XYZ", "ICtCp", XYZ_to_ICtCp, 1,),
    ("ICtCp", "CIE XYZ", ICtCp_to_XYZ, 1,),
    ("CIE XYZ", "IgPgTg", XYZ_to_IgPgTg, 1,),
    ("IgPgTg", "CIE XYZ", IgPgTg_to_XYZ, 1,),
    ("CIE XYZ", "IPT", XYZ_to_IPT, 1,),
    ("IPT", "CIE XYZ", IPT_to_XYZ, 1,),
    ("CIE XYZ", "IPT Ragoo 2021", XYZ_to_IPT_Ragoo2021, 1,),
    ("IPT Ragoo 2021", "CIE XYZ", IPT_Ragoo2021_to_XYZ, 1,),
    ("CIE XYZ", "Jzazbz", XYZ_to_Jzazbz, 1,),
    ("Jzazbz", "CIE XYZ", Jzazbz_to_XYZ, 1,),
    ("CIE XYZ", "hdr-IPT", XYZ_to_hdr_IPT, 1,),
    ("hdr-IPT", "CIE XYZ", hdr_IPT_to_XYZ, 1,),
    ("CIE XYZ", "OSA UCS", XYZ_to_OSA_UCS, 100,),
    ("OSA UCS", "CIE XYZ", OSA_UCS_to_XYZ, 1,),
    ("CIE XYZ", "Oklab", XYZ_to_Oklab, 1,),
    ("Oklab", "CIE XYZ", Oklab_to_XYZ, 1,),
    ("CIE XYZ", "ProLab", XYZ_to_ProLab, 1,),
    ("ProLab", "CIE XYZ", ProLab_to_XYZ, 1,),
    ("CIE XYZ", "sUCS", XYZ_to_sUCS, 1,),
    ("sUCS", "CIE XYZ", sUCS_to_XYZ, 1,),
    ("CIE XYZ", "Yrg", XYZ_to_Yrg, 1,),
    ("Yrg", "CIE XYZ", Yrg_to_XYZ, 1,),
    ("CIE 1931", "CIE XYZ", xyY_to_XYZ, 1,),
    ("CIE XYZ", "CIE 1931", XYZ_to_xyY, 1,),
    ("CIE 1960 UCS", "CIE XYZ", CIE1960UCS_to_XYZ, 1,),
    ("CIE XYZ", "CIE 1960 UCS", XYZ_to_CIE1960UCS, 1,),
    ("CIE 1976 UCS", "CIE XYZ", CIE1976UCS_to_XYZ, 1,),
    ("CIE XYZ", "CIE 1976 UCS", XYZ_to_CIE1976UCS, 1,),
    # RGB Colour Models
    ("CIE XYZ", "RGB", partial(XYZ_to_RGB, colourspace=_RGB_COLOURSPACE_DEFAULT), 1,),
    ("RGB", "CIE XYZ", partial(RGB_to_XYZ, colourspace=_RGB_COLOURSPACE_DEFAULT), 1,),
    ("RGB", "Scene-Referred RGB", partial(RGB_to_RGB, input_colourspace=_RGB_COLOURSPACE_DEFAULT, output_colourspace=_RGB_COLOURSPACE_DEFAULT), 1,),
    ("Scene-Referred RGB", "RGB", partial(RGB_to_RGB, input_colourspace=_RGB_COLOURSPACE_DEFAULT, output_colourspace=_RGB_COLOURSPACE_DEFAULT), 1,),
    ("RGB", "HSV", RGB_to_HSV, 1,),
    ("HSV", "RGB", HSV_to_RGB, 1,),
    ("RGB", "HSL", RGB_to_HSL, 1,),
    ("HSL", "RGB", HSL_to_RGB, 1,),
    ("RGB", "HCL", RGB_to_HCL, 1,),
    ("HCL", "RGB", HCL_to_RGB, 1,),
    ("RGB", "IHLS", RGB_to_IHLS, 1,),
    ("IHLS", "RGB", IHLS_to_RGB, 1,),
    ("CMY", "RGB", CMY_to_RGB, 1,),
    ("RGB", "CMY", RGB_to_CMY, 1,),
    ("CMY", "CMYK", CMY_to_CMYK, 1,),
    ("CMYK", "CMY", CMYK_to_CMY, 1,),
    ("RGB", "RGB Luminance", partial(RGB_luminance, primaries=_RGB_COLOURSPACE_DEFAULT.primaries, whitepoint=_RGB_COLOURSPACE_DEFAULT.whitepoint), 1,),
    ("RGB Luminance", "RGB", RGB_luminance_to_RGB, 1,),
    ("RGB", "Prismatic", RGB_to_Prismatic, 1,),
    ("Prismatic", "RGB", Prismatic_to_RGB, 1,),
    ("Output-Referred RGB", "YCbCr", RGB_to_YCbCr, 1,),
    ("YCbCr", "Output-Referred RGB", YCbCr_to_RGB, 1,),
    ("RGB", "YcCbcCrc", RGB_to_YcCbcCrc, 1,),
    ("YcCbcCrc", "RGB", YcCbcCrc_to_RGB, 1,),
    ("Output-Referred RGB", "YCoCg", RGB_to_YCoCg, 1,),
    ("YCoCg", "Output-Referred RGB", YCoCg_to_RGB, 1,),
    ("RGB", "Output-Referred RGB", cctf_encoding, 1,),
    ("Output-Referred RGB", "RGB", cctf_decoding, 1,),
    ("Scene-Referred RGB", "Output-Referred RGB", cctf_encoding, 1,),
    ("Output-Referred RGB", "Scene-Referred RGB", cctf_decoding, 1,),
    ("CIE XYZ", "sRGB", XYZ_to_sRGB, 1,),
    ("sRGB", "CIE XYZ", sRGB_to_XYZ, 1,),
    # Colour Notation Systems
    ("Output-Referred RGB", "Hexadecimal", RGB_to_HEX, None,),
    ("Hexadecimal", "Output-Referred RGB", HEX_to_RGB, 1,),
    ("CSS Color 3", "Output-Referred RGB", keyword_to_RGB_CSSColor3, 1,),
    ("CIE xyY", "Munsell Colour", xyY_to_munsell_colour, None,),
    ("Munsell Colour", "CIE xyY", munsell_colour_to_xyY, 1,),
    ("Luminance", "Munsell Value", munsell_value, 10,),
    ("Munsell Value", "Luminance", partial(luminance, method="ASTM D1535"), 100,),
    # Colour Quality
    ("Spectral Distribution", "CRI", colour_rendering_index, 1,),
    ("Spectral Distribution", "CQS", colour_quality_scale, 1,),
    # Colour Temperature
    ("CCT", "CIE UCS uv", CCT_to_uv, None,),
    ("CIE UCS uv", "CCT", uv_to_CCT, None,),
    ("CCT", "Mired", CCT_D_uv_to_mired, None,),
    ("Mired", "CCT", mired_to_CCT_D_uv, None,),
    # Advanced Colorimetry
    ("CIE XYZ", "ATD95", partial(XYZ_to_ATD95, XYZ_0=_TVS_ILLUMINANT_DEFAULT, Y_0=80 * 0.2, k_1=0, k_2=(15 + 50) / 2), (360, 1, 1, 1, 1, 1, 1, 1, 1),),
    ("CIE XYZ", "CIECAM02", partial(XYZ_to_CIECAM02, **_CAM_KWARGS_CIECAM02_sRGB), (100, 100, 360, 100, 100, 100, 400, 1),),
    ("CIECAM02", "CIE XYZ", partial(CIECAM02_to_XYZ, **_CAM_KWARGS_CIECAM02_sRGB), 1,),
    ("CIECAM02", "CIECAM02 JMh", CIECAM02_to_JMh_CIECAM02, None,),
    ("CIECAM02 JMh", "CIECAM02", JMh_CIECAM02_to_CIECAM02, None,),
    ("CIE XYZ", "CAM16", partial(XYZ_to_CAM16, **_CAM_KWARGS_CIECAM02_sRGB), (100, 100, 360, 100, 100, 100, 400, 1),),
    ("CAM16", "CIE XYZ", partial(CAM16_to_XYZ, **_CAM_KWARGS_CIECAM02_sRGB), 1,),
    ("CAM16", "CAM16 JMh", CAM16_to_JMh_CAM16, None,),
    ("CAM16 JMh", "CAM16", JMh_CAM16_to_CAM16, None,),
    ("CIE XYZ", "CIECAM16", partial(XYZ_to_CIECAM16, **_CAM_KWARGS_CIECAM02_sRGB), (100, 100, 360, 100, 100, 100, 400, 1),),
    ("CIECAM16", "CIE XYZ", partial(CIECAM16_to_XYZ, **_CAM_KWARGS_CIECAM02_sRGB), 1,),
    ("CIECAM16", "CIECAM16 JMh", CIECAM16_to_JMh_CIECAM16, None,),
    ("CIECAM16 JMh", "CIECAM16", JMh_CIECAM16_to_CIECAM16, None,),
    ("CIE XYZ", "Hellwig 2022", partial(XYZ_to_Hellwig2022, **_CAM_KWARGS_CIECAM02_sRGB), (100, 100, 360, 100, 100, 100, 400, 1, 100, 100),),
    ("Hellwig 2022", "CIE XYZ", partial(Hellwig2022_to_XYZ, **_CAM_KWARGS_CIECAM02_sRGB), 1,),
    ("Hellwig 2022", "Hellwig 2022 JMh", Hellwig2022_to_JMh_Hellwig2022, None,),
    ("Hellwig 2022 JMh", "Hellwig 2022", JMh_Hellwig2022_to_Hellwig2022, None,),
    ("CIE XYZ", "Kim 2009", partial(XYZ_to_Kim2009, XYZ_w=_TVS_ILLUMINANT_DEFAULT, L_A=80 * 0.2), (100, 100, 360, 100, 100, 100, 400, 1),),
    ("Kim 2009", "CIE XYZ", partial(Kim2009_to_XYZ, XYZ_w=_TVS_ILLUMINANT_DEFAULT, L_A=80 * 0.2), 1,),
    ("CIE XYZ", "Hunt", partial(XYZ_to_Hunt, XYZ_w=_TVS_ILLUMINANT_DEFAULT, XYZ_b=_TVS_ILLUMINANT_DEFAULT, L_A=80 * 0.2, CCT_w=6504), (1, 1, 360, 1, 1, 1, 1, 1),),
    ("CIE XYZ", "LLAB", partial(XYZ_to_LLAB, XYZ_0=_TVS_ILLUMINANT_DEFAULT, Y_b=80 * 0.2, L=80), (1, 1, 360, 1, 1, 1, 1, 1),),
    ("CIE XYZ", "Nayatani95", partial(XYZ_to_Nayatani95, XYZ_n=_TVS_ILLUMINANT_DEFAULT, Y_o=0.2, E_o=1000, E_or=1000), (1, 1, 360, 1, 1, 1, 1, 1, 1),),
    ("CIE XYZ", "RLAB", partial(XYZ_to_RLAB, XYZ_n=_TVS_ILLUMINANT_DEFAULT, Y_n=20), (1, 1, 360, 1, 1, 1, 1),),
    ("CIE XYZ", "sCAM", partial(XYZ_to_sCAM, **_CAM_KWARGS_CIECAM02_sRGB), (100, 100, 360, 100, 100, 400, 1, 100, 100, 100, 100),),
    ("sCAM", "CIE XYZ", partial(sCAM_to_XYZ, **_CAM_KWARGS_CIECAM02_sRGB), 1,),
    ("sCAM", "sCAM JMh", sCAM_to_JMh_sCAM, None,),
    ("sCAM JMh", "sCAM", JMh_sCAM_to_sCAM, None,),
    ("CIE XYZ", "ZCAM", partial(XYZ_to_ZCAM, XYZ_w=_TVS_ILLUMINANT_DEFAULT, L_A=64 / np.pi * 0.2, Y_b=20), (1, 1, 360, 1, 1, 1, 400, 1, 1, 1, 1),),
    ("ZCAM", "CIE XYZ", partial(ZCAM_to_XYZ, XYZ_w=_TVS_ILLUMINANT_DEFAULT, L_A=64 / np.pi * 0.2, Y_b=20), 1,),
    ("CIECAM02 JMh", "CAM02LCD", JMh_CIECAM02_to_CAM02LCD, 100,),
    ("CAM02LCD", "CIECAM02 JMh", CAM02LCD_to_JMh_CIECAM02, (100, 100, 360),),
    ("CIECAM02 JMh", "CAM02SCD", JMh_CIECAM02_to_CAM02SCD, 100,),
    ("CAM02SCD", "CIECAM02 JMh", CAM02SCD_to_JMh_CIECAM02, (100, 100, 360),),
    ("CIECAM02 JMh", "CAM02UCS", JMh_CIECAM02_to_CAM02UCS, 100,),
    ("CAM02UCS", "CIECAM02 JMh", CAM02UCS_to_JMh_CIECAM02, (100, 100, 360),),
    ("CAM16 JMh", "CAM16LCD", JMh_CAM16_to_CAM16LCD, 100,),
    ("CAM16LCD", "CAM16 JMh", CAM16LCD_to_JMh_CAM16, (100, 100, 360),),
    ("CAM16 JMh", "CAM16SCD", JMh_CAM16_to_CAM16SCD, 100,),
    ("CAM16SCD", "CAM16 JMh", CAM16SCD_to_JMh_CAM16, (100, 100, 360),),
    ("CAM16 JMh", "CAM16UCS", JMh_CAM16_to_CAM16UCS, 100,),
    ("CAM16UCS", "CAM16 JMh", CAM16UCS_to_JMh_CAM16, (100, 100, 360),),
]
"""
Automatic colour conversion graph specifications data describing two nodes and
the edge in the graph.
"""
# fmt: on


# Programmatically defining the colourspace models polar conversions.
def _format_node_name(name: str) -> str:
    """
    Format the specified name by applying a series of substitutions.

    This function transforms node names according to predefined patterns,
    typically converting underscores to hyphens and applying other naming
    conventions used in the colourspace models polar conversions system.

    Parameters
    ----------
    name
        The node name to format.

    Returns
    -------
    :class:`str`
        The formatted node name with substitutions applied.
    """

    for pattern, substitution in [
        ("hdr_", "hdr-"),
        ("-CIELab", "-CIELAB"),
        ("_", " "),
        ("^Lab", "CIE Lab"),
        ("^LCHab", "CIE LCHab"),
        ("^Luv", "CIE Luv"),
        ("^LCHuv", "CIE LCHuv"),
        ("Ragoo2021", "Ragoo 2021"),
    ]:
        name = re.sub(pattern, substitution, name)

    return name


for _Jab, _JCh in COLOURSPACE_MODELS_POLAR_CONVERSIONS:
    _module = sys.modules["colour.models"]
    _Jab_name = _format_node_name(_Jab)
    _JCh_name = _format_node_name(_JCh)

    # Determine to_reference scales based on color space type
    # Lab-based spaces use (100, 100, x), others use (1, 1, x)
    if "Lab" in _Jab or "Luv" in _Jab:
        _Jab_to_JCh_scale = (100, 100, 360)
        _JCh_to_Jab_scale = 100
    else:
        _Jab_to_JCh_scale = (1, 1, 360)
        _JCh_to_Jab_scale = 1

    CONVERSION_SPECIFICATIONS_DATA.append(
        (_Jab_name, _JCh_name, getattr(_module, f"{_Jab}_to_{_JCh}"), _Jab_to_JCh_scale)
    )
    CONVERSION_SPECIFICATIONS_DATA.append(
        (_JCh_name, _Jab_name, getattr(_module, f"{_JCh}_to_{_Jab}"), _JCh_to_Jab_scale)
    )

del _format_node_name, _JCh, _Jab, _module, _Jab_name, _JCh_name

CONVERSION_SPECIFICATIONS: list = [
    Conversion_Specification(*specification)
    for specification in CONVERSION_SPECIFICATIONS_DATA
]
"""
Automatic colour conversion graph specifications describing two nodes and
the edge in the graph.
"""

CONVERSION_GRAPH_NODE_LABELS: dict = {
    specification[0].lower(): specification[0]
    for specification in CONVERSION_SPECIFICATIONS_DATA
}
"""Automatic colour conversion graph node labels."""

CONVERSION_GRAPH_NODE_LABELS.update(
    {
        specification[1].lower(): specification[1]
        for specification in CONVERSION_SPECIFICATIONS_DATA
    }
)


@required("NetworkX")
def _build_graph() -> networkx.DiGraph:  # pyright: ignore  # noqa: F821
    """
    Build the automatic colour conversion graph.

    Returns
    -------
    :class:`networkx.DiGraph`
        Automatic colour conversion graph.
    """

    import networkx as nx  # noqa: PLC0415

    graph = nx.DiGraph()

    for specification in CONVERSION_SPECIFICATIONS:
        graph.add_edge(
            specification.source,
            specification.target,
            conversion_function=specification.conversion_function,
            to_reference=specification.to_reference,
        )

    return graph


CONVERSION_GRAPH: nx.DiGraph | None = None  # pyright: ignore # noqa: F821
"""Automatic colour conversion graph."""


@required("NetworkX")
def _conversion_path(
    source: str, target: str
) -> tuple[List[Callable], tuple[float, ...] | None]:
    """
    Generate the conversion path from the source node to the target node in
    the automatic colour conversion graph.

    Parameters
    ----------
    source
        Source node.
    target
        Target node.

    Returns
    -------
    :class:`tuple`
        Tuple of (conversion_functions, to_reference) where conversion_functions
        is a list of conversion function callables and to_reference is the
        scale factors for the final conversion (or None if not available).

    Examples
    --------
    >>> functions, to_ref = _conversion_path("cie lab", "cie lchab")
    >>> len(functions)
    1
    >>> to_ref
    (100, 100, 360)
    """

    import networkx as nx  # noqa: PLC0415

    global CONVERSION_GRAPH  # noqa: PLW0603

    if CONVERSION_GRAPH is None:
        # Updating the :attr:`CONVERSION_GRAPH` attributes.
        colour.graph.CONVERSION_GRAPH = CONVERSION_GRAPH = _build_graph()

    path = nx.shortest_path(cast("nx.DiGraph", CONVERSION_GRAPH), source, target)

    conversion_functions, to_reference = [], None

    for a, b in itertools.pairwise(path):
        edge_data = CONVERSION_GRAPH.get_edge_data(a, b)  # pyright: ignore
        conversion_functions.append(edge_data["conversion_function"])
        to_reference = edge_data.get("to_reference")

    return conversion_functions, to_reference


def _lower_order_function(callable_: Callable) -> Callable:
    """
    Extract the lower-order function from the specified callable, such as the
    underlying function wrapped by a partial object.

    Parameters
    ----------
    callable_
        Callable from which to extract the lower-order function.

    Returns
    -------
    Callable
        Lower-order function if the callable is a partial object, otherwise
        the original callable.
    """

    return callable_.func if isinstance(callable_, partial) else callable_


def describe_conversion_path(
    source: str,
    target: str,
    mode: Literal["Short", "Long", "Extended"] | str = "Short",
    width: int = 79,
    padding: int = 3,
    print_callable: Callable = print,
    **kwargs: Any,
) -> None:
    """
    Describe the conversion path from the specified source colour
    representation to the specified target colour representation using the
    automatic colour conversion graph.

    Parameters
    ----------
    source
        Source colour representation, i.e., the source node in the automatic
        colour conversion graph.
    target
        Target colour representation, i.e., the target node in the automatic
        colour conversion graph.
    mode
        Verbose mode: *Short* describes the conversion path, *Long* provides
        details about the arguments, definitions signatures and output
        values, *Extended* appends the definitions' documentation.
    width
        Message box width.
    padding
        Padding on each side of the message.
    print_callable
        Callable used to print the message box.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.convert`},
        See the documentation of the previously listed definition.

    Raises
    ------
    ValueError
        If the mode is not one of the supported values.
    NetworkXNoPath
        If no conversion path exists between the source and target colour
        representations.

    Examples
    --------
    >>> describe_conversion_path("Spectral Distribution", "sRGB", width=75)
    ===========================================================================
    *                                                                         *
    *   [ Conversion Path ]                                                   *
    *                                                                         *
    *   "sd_to_XYZ" --> "XYZ_to_sRGB"                                         *
    *                                                                         *
    ===========================================================================
    """

    try:  # pragma: no cover
        signature_inspection = inspect.signature
    except AttributeError:  # pragma: no cover
        signature_inspection = inspect.getfullargspec

    source, target = source.lower(), target.lower()
    mode = validate_method(
        mode,
        (
            "Short",
            "Long",
            "Extended",
            None,
        ),  # NEEDS MANUAL REVIEW
        '"{0}" mode is invalid, it must be one of {1}!',
    )

    width = (79 + 2 + 2 * 3 - 4) if mode == "extended" else width

    conversion_path, _to_reference = _conversion_path(source, target)

    joined_conversion_path = " --> ".join(
        [
            f'"{_lower_order_function(conversion_function).__name__}"'
            for conversion_function in conversion_path
        ]
    )

    message_box(
        f"[ Conversion Path ]\n\n{joined_conversion_path}",
        width,
        padding,
        print_callable,
    )

    for conversion_function in conversion_path:
        conversion_function_name = _lower_order_function(conversion_function).__name__

        # Filtering compatible keyword arguments passed directly and
        # irrespective of any conversion function name.
        filtered_kwargs = filter_kwargs(conversion_function, **kwargs)

        # Filtering keyword arguments passed as dictionary with the
        # conversion function name.
        filtered_kwargs.update(kwargs.get(conversion_function_name, {}))

        return_value = filtered_kwargs.pop("return", None)

        if mode in ("long", "extended"):
            signature = pformat(
                signature_inspection(_lower_order_function(conversion_function))
            )
            message = (
                f'[ "{_lower_order_function(conversion_function).__name__}" ]\n\n'
                f"[ Signature ]\n\n"
                f"{signature}"
            )

            if filtered_kwargs:
                message += f"\n\n[ Filtered Arguments ]\n\n{pformat(filtered_kwargs)}"

            if mode in ("extended",):
                docstring = textwrap.dedent(
                    str(_lower_order_function(conversion_function).__doc__)
                ).strip()
                message += f"\n\n[ Documentation ]\n\n {docstring}"

            if return_value is not None:
                message += f"\n\n[ Conversion Output ]\n\n{return_value}"

            message_box(message, width, padding, print_callable)


def convert(
    a: Any, source: str, target: str, *, to_reference_scale: bool = False, **kwargs: Any
) -> Any:
    """
    Convert specified object :math:`a` from source colour representation to
    target colour representation using the automatic colour conversion
    graph.

    The conversion is performed by finding the shortest path in a
    `NetworkX <https://networkx.github.io>`__ :class:`DiGraph` class
    instance.

    The conversion path adopts the **'1'** domain-range scale and the
    object :math:`a` is expected to be *soft* normalised accordingly. For
    example, *CIE XYZ* tristimulus values arguments for use with the
    *CAM16* colour appearance model should be in domain `[0, 1]` instead
    of the domain `[0, 100]` used with the **'Reference'** domain-range
    scale. The arguments are typically converted as follows:

    -   *Scalars* in domain-range `[0, 10]`, e.g *Munsell Value* are
        scaled by *10*.
    -   *Percentages* in domain-range `[0, 100]` are scaled by *100*.
    -   *Degrees* in domain-range `[0, 360]` are scaled by *360*.
    -   *Integers* in domain-range `[0, 2**n -1]` where `n` is the bit
        depth are scaled by *2**n -1*.

    See the `Domain-Range Scales <../basics.html#domain-range-scales>`__
    page for more information.

    Parameters
    ----------
    a
        Object :math:`a` to convert. If :math:`a` represents a
        reflectance, transmittance or absorptance value, the expectation
        is that it is viewed under *CIE Standard Illuminant D Series*
        *D65*. The illuminant can be changed on a per-definition basis
        along the conversion path.
    source
        Source colour representation, i.e., the source node in the
        automatic colour conversion graph.
    target
        Target colour representation, i.e., the target node in the
        automatic colour conversion graph.
    to_reference_scale
        If True, automatically rescale the output from domain-range scale
        '1' to 'reference' scale using the to_reference metadata. For
        example, CIE LCHab output will be scaled to (100, 100, 360) instead
        of 1. Has no effect if the target conversion doesn't have
        to_reference metadata.

    Other Parameters
    ----------------
    kwargs
        See the documentation of the supported conversion definitions.

        Arguments for the conversion definitions are passed as keyword
        arguments whose names are those of the conversion definitions and
        values set as dictionaries. For example, in the conversion from
        spectral distribution to *sRGB* colourspace, passing arguments to
        the :func:`colour.sd_to_XYZ` definition is done as follows::

            convert(
                sd,
                "Spectral Distribution",
                "sRGB",
                sd_to_XYZ={"illuminant": SDS_ILLUMINANTS["FL2"]},
            )

        It is also possible to pass keyword arguments directly to the
        various conversion definitions irrespective of their name. This is
        ``dangerous`` and could cause unexpected behaviour, consider the
        following conversion::

             convert(sd, "Spectral Distribution", "sRGB", "illuminant": \
SDS_ILLUMINANTS["FL2"])

        Because both the :func:`colour.sd_to_XYZ` and
        :func:`colour.XYZ_to_sRGB` definitions have an *illuminant*
        argument, `SDS_ILLUMINANTS["FL2"]` will be passed to both of them
        and will raise an exception in the :func:`colour.XYZ_to_sRGB`
        definition. This will be addressed in the future by either
        catching the exception and trying a new time without the keyword
        argument or more elegantly via type checking.

        With that in mind, this mechanism offers some good benefits: For
        example, it allows defining a conversion from *CIE XYZ*
        colourspace to *n* different colour models while passing an
        illuminant argument but without having to explicitly define all
        the explicit conversion definition arguments::

            a = np.array([0.20654008, 0.12197225, 0.05136952])
            illuminant = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
            for model in ("CIE xyY", "CIE Lab"):
                convert(a, "CIE XYZ", model, illuminant=illuminant)

        Instead of::

            for model in ("CIE xyY", "CIE Lab"):
                convert(
                    a,
                    "CIE XYZ",
                    model,
                    XYZ_to_xyY={"illuminant": illuminant},
                    XYZ_to_Lab={"illuminant": illuminant},
                )

        Mixing both approaches is possible for the brevity benefits. It is
        made possible because the keyword arguments directly passed are
        filtered first and then the resulting dict is updated with the
        explicit conversion definition arguments::

            illuminant = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
             convert(sd, "Spectral Distribution", "sRGB", "illuminant": \
SDS_ILLUMINANTS["FL2"], XYZ_to_sRGB={"illuminant": illuminant})

        For inspection purposes, verbose is enabled by passing arguments
        to the :func:`colour.describe_conversion_path` definition via the
        ``verbose`` keyword argument as follows::

            convert(sd, "Spectral Distribution", "sRGB", verbose={"mode": "Long"})

    Returns
    -------
    Any
        Converted object :math:`a`.

    Raises
    ------
    NetworkXNoPath
        If no conversion path exists between the source and target colour
        representations.

    Warnings
    --------
    The domain-range scale is **'1'** and cannot be changed.

    Notes
    -----
    -   The **RGB** colour representation is assumed to be linear and
        representing *scene-referred* imagery, i.e., **Scene-Referred
        RGB** representation. To encode such *RGB* values as
        *output-referred* (*display-referred*) imagery, i.e., encode the
        *RGB* values using an encoding colour component transfer function
        (Encoding CCTF) / opto-electronic transfer function (OETF), the
        **Output-Referred RGB** representation must be used::

             convert(RGB, "Scene-Referred RGB", "Output-Referred RGB")

        Likewise, encoded *output-referred* *RGB* values can be decoded
        with the **Scene-Referred RGB** representation::

            convert(RGB, "Output-Referred RGB", "Scene-Referred RGB")

    -   The following defaults have been adopted:

        -   The default illuminant for the computation is *CIE Standard
            Illuminant D Series* *D65*. It can be changed on a
            per-definition basis along the conversion path. Note that the
            conversion from spectral to *CIE XYZ* tristimulus values
            remains unchanged.
        -   The default *RGB* colourspace primaries and whitepoint are
            that of the *BT.709*/*sRGB* colourspace. They can be changed
            on a per-definition basis along the conversion path.
        -   When using **sRGB** as a source or target colour
            representation, the convenient :func:`colour.sRGB_to_XYZ` and
            :func:`colour.XYZ_to_sRGB` definitions are used,
            respectively. Thus, decoding and encoding using the sRGB
            electro-optical transfer function (EOTF) and its inverse will
            be applied by default.
        -   Most of the colour appearance models have defaults set
            according to *IEC 61966-2-1:1999* viewing conditions, i.e.,
            *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
            adapting field luminance about 20% of a white object in the
            scene.

    Examples
    --------
    >>> import numpy as np
    >>> from colour import SDS_COLOURCHECKERS, SDS_ILLUMINANTS
    >>> sd = SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"]
    >>> convert(
    ...     sd,
    ...     "Spectral Distribution",
    ...     "sRGB",
    ...     verbose={"mode": "Short", "width": 75},
    ... )
    ... # doctest: +ELLIPSIS
    ===========================================================================
    *                                                                         *
    *   [ Conversion Path ]                                                   *
    *                                                                         *
    *   "sd_to_XYZ" --> "XYZ_to_sRGB"                                         *
    *                                                                         *
    ===========================================================================
    array([ 0.4903477...,  0.3018587...,  0.2358768...])
    >>> illuminant = SDS_ILLUMINANTS["FL2"]
    >>> convert(
    ...     sd,
    ...     "Spectral Distribution",
    ...     "sRGB",
    ...     sd_to_XYZ={"illuminant": illuminant},
    ... )
    ... # doctest: +ELLIPSIS
    array([ 0.4792457...,  0.3167696...,  0.1736272...])
    >>> a = np.array([0.45675795, 0.30986982, 0.24861924])
    >>> convert(a, "Output-Referred RGB", "CAM16UCS")
    ... # doctest: +ELLIPSIS
    array([ 0.3999481...,  0.0920655...,  0.0812752...])
    >>> a = np.array([0.39994811, 0.09206558, 0.08127526])
    >>> convert(a, "CAM16UCS", "sRGB", verbose={"mode": "Short", "width": 75})
    ... # doctest: +ELLIPSIS
    ===========================================================================
    *                                                                         *
    *   [ Conversion Path ]                                                   *
    *                                                                         *
    *   "UCS_Li2017_to_JMh_CAM16" --> "JMh_CAM16_to_CAM16" -->                *
    *   "CAM16_to_XYZ" --> "XYZ_to_sRGB"                                      *
    *                                                                         *
    ===========================================================================
    array([ 0.4567576...,  0.3098826...,  0.2486222...])
    """

    source, target = source.lower(), target.lower()

    conversion_path, to_reference = _conversion_path(source, target)

    verbose_kwargs = copy(kwargs)
    for conversion_function in conversion_path:
        conversion_function_name = _lower_order_function(conversion_function).__name__

        # Filtering compatible keyword arguments passed directly and
        # irrespective of any conversion function name.
        filtered_kwargs = filter_kwargs(conversion_function, **kwargs)

        # Filtering keyword arguments passed as dictionary with the
        # conversion function name.
        filtered_kwargs.update(kwargs.get(conversion_function_name, {}))

        with domain_range_scale("1"):
            a = conversion_function(a, **filtered_kwargs)

        if conversion_function_name in verbose_kwargs:
            verbose_kwargs[conversion_function_name]["return"] = a
        else:
            verbose_kwargs[conversion_function_name] = {"return": a}

    # Apply rescaling if requested and to_reference is available
    if to_reference_scale and to_reference is not None:
        a = as_float_array(a) * np.array(to_reference)

    if "verbose" in verbose_kwargs:
        verbose_kwargs.update(verbose_kwargs.pop("verbose"))
        describe_conversion_path(source, target, **verbose_kwargs)

    return a
