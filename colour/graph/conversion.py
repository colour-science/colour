# -*- coding: utf-8 -*-
"""
Automatic Colour Conversion Graph
=================================

Defines the automatic colour conversion graph objects:

-   :func:`colour.describe_conversion_path`
-   :func:`colour.convert`

See Also
--------
`Automatic Colour Conversion Graph Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/graph/conversion.ipynb>`_
"""

from __future__ import division, print_function, unicode_literals

import inspect
import numpy as np
import textwrap
from collections import namedtuple
from copy import copy
from functools import partial
from pprint import pformat

from colour.colorimetry import (ILLUMINANTS, ILLUMINANTS_SDS,
                                HUNTERLAB_ILLUMINANTS)
from colour.colorimetry import (colorimetric_purity, complementary_wavelength,
                                dominant_wavelength, excitation_purity,
                                lightness, luminance, luminous_efficacy,
                                luminous_efficiency, luminous_flux, sd_to_XYZ,
                                whiteness, yellowness, wavelength_to_XYZ)
from colour.recovery import XYZ_to_sd
from colour.models import sRGB_COLOURSPACE
from colour.models import (
    CAM02LCD_to_JMh_CIECAM02, CAM02SCD_to_JMh_CIECAM02,
    CAM02UCS_to_JMh_CIECAM02, CAM16LCD_to_JMh_CAM16, CAM16SCD_to_JMh_CAM16,
    CAM16UCS_to_JMh_CAM16, CMYK_to_CMY, CMY_to_CMYK, CMY_to_RGB, DIN99_to_Lab,
    HSL_to_RGB, HSV_to_RGB, Hunter_Lab_to_XYZ, Hunter_Rdab_to_XYZ,
    ICTCP_to_RGB, IPT_to_XYZ, JMh_CAM16_to_CAM16LCD, JMh_CAM16_to_CAM16SCD,
    JMh_CAM16_to_CAM16UCS, JMh_CIECAM02_to_CAM02LCD, JMh_CIECAM02_to_CAM02SCD,
    JMh_CIECAM02_to_CAM02UCS, JzAzBz_to_XYZ, LCHab_to_Lab, LCHuv_to_Luv,
    Lab_to_DIN99, Lab_to_LCHab, Lab_to_XYZ, Luv_to_LCHuv, Luv_to_XYZ,
    Luv_to_uv, Luv_uv_to_xy, OSA_UCS_to_XYZ, Prismatic_to_RGB, RGB_luminance,
    RGB_to_CMY, RGB_to_HSL, RGB_to_HSV, RGB_to_ICTCP, RGB_to_Prismatic,
    RGB_to_RGB, RGB_to_XYZ, RGB_to_YCbCr, RGB_to_YCoCg, RGB_to_YcCbcCrc,
    UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy, UVW_to_XYZ, XYZ_to_Hunter_Lab,
    XYZ_to_Hunter_Rdab, XYZ_to_IPT, XYZ_to_JzAzBz, XYZ_to_Lab, XYZ_to_Luv,
    XYZ_to_OSA_UCS, XYZ_to_RGB, XYZ_to_UCS, XYZ_to_UVW, XYZ_to_hdr_CIELab,
    XYZ_to_hdr_IPT, XYZ_to_sRGB, XYZ_to_xy, XYZ_to_xyY, YCbCr_to_RGB,
    YCoCg_to_RGB, YcCbcCrc_to_RGB, cctf_decoding, cctf_encoding,
    hdr_CIELab_to_XYZ, hdr_IPT_to_XYZ, sRGB_to_XYZ, uv_to_Luv, uv_to_UCS,
    xyY_to_XYZ, xyY_to_xy, xy_to_Luv_uv, xy_to_UCS_uv, xy_to_XYZ, xy_to_xyY)
from colour.notation import (HEX_to_RGB, RGB_to_HEX, munsell_value,
                             munsell_colour_to_xyY, xyY_to_munsell_colour)
from colour.quality import colour_quality_scale, colour_rendering_index
from colour.appearance import (
    CAM16_Specification, CAM16_to_XYZ, CIECAM02_Specification, CIECAM02_to_XYZ,
    XYZ_to_ATD95, XYZ_to_CAM16, XYZ_to_CIECAM02, XYZ_to_Hunt, XYZ_to_LLAB,
    XYZ_to_Nayatani95, XYZ_to_RLAB)
from colour.temperature import CCT_to_uv, CCT_to_xy, uv_to_CCT, xy_to_CCT
from colour.utilities import (domain_range_scale, filter_kwargs,
                              is_networkx_installed, message_box, tsplit,
                              tstack, usage_warning)

if is_networkx_installed():  # pragma: no cover
    import networkx as nx

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'ConversionSpecification', 'CIECAM02_to_JMh_CIECAM02',
    'JMh_CIECAM02_to_CIECAM02', 'CAM16_to_JMh_CAM16', 'JMh_CAM16_to_CAM16',
    'XYZ_to_luminance', 'RGB_luminance_to_RGB',
    'CONVERSION_SPECIFICATIONS_DATA', 'CONVERSION_GRAPH_NODE_LABELS',
    'CONVERSION_SPECIFICATIONS', 'CONVERSION_GRAPH',
    'describe_conversion_path', 'convert'
]


class ConversionSpecification(
        namedtuple('ConversionSpecification',
                   ('source', 'target', 'conversion_function'))):
    """
    Conversion specification for *Colour* graph for automatic colour
    conversion describing two nodes and the edge in the graph.

    Parameters
    ----------
    source : unicode
        Source node in the graph.
    target : array_like
        Target node in the graph.
    conversion_function : callable
        Callable converting from the ``source`` node to the ``target`` node.
    """

    def __new__(cls, source=None, target=None, conversion_function=None):
        return super(ConversionSpecification, cls).__new__(
            cls, source.lower(), target.lower(), conversion_function)


def CIECAM02_to_JMh_CIECAM02(CIECAM02_specification):
    """
    Converts from *CIECAM02* specification to *CIECAM02* :math:`JMh`
    correlates.

    Parameters
    ----------
    CIECAM02_specification : CIECAM02_Specification
        *CIECAM02* colour appearance model specification.

    Returns
    -------
    ndarray
        *CIECAM02* :math:`JMh` correlates.

    Examples
    --------
    >>> specification = CIECAM02_Specification(J=41.731091132513917,
    ...                                        M=0.108842175669226,
    ...                                        h=219.048432658311780)
    >>> CIECAM02_to_JMh_CIECAM02(specification)  # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    return tstack([
        CIECAM02_specification.J,
        CIECAM02_specification.M,
        CIECAM02_specification.h,
    ])


def JMh_CIECAM02_to_CIECAM02(JMh):
    """
    Converts from *CIECAM02* :math:`JMh` correlates to *CIECAM02*
    specification.

    Parameters
    ----------
    JMh : array_like
         *CIECAM02* :math:`JMh` correlates.

    Returns
    -------
    CIECAM02_Specification
        *CIECAM02* colour appearance model specification.

    Examples
    --------
    >>> JMh = np.array([4.17310911e+01, 1.08842176e-01, 2.19048433e+02])
    >>> JMh_CIECAM02_to_CIECAM02(JMh)  # doctest: +ELLIPSIS
    CIECAM02_Specification(J=41.7310911..., C=None, h=219.0484329..., s=None, \
Q=None, M=0.1088421..., H=None, HC=None)
    """

    J, M, h = tsplit(JMh)

    return CIECAM02_Specification(J=J, M=M, h=h)


def CAM16_to_JMh_CAM16(CAM16_specification):
    """
    Converts from *CAM16* specification to *CAM16* :math:`JMh` correlates.

    Parameters
    ----------
    CAM16_specification : CAM16_Specification
        *CAM16* colour appearance model specification.

    Returns
    -------
    ndarray
        *CAM16* :math:`JMh` correlates.

    Examples
    --------
    >>> specification = CAM16_Specification(J=41.731207905126638,
    ...                                     M=0.107436772335905,
    ...                                     h=217.067959767393010)
    >>> CAM16_to_JMh_CAM16(specification)  # doctest: +ELLIPSIS
    array([  4.1731207...e+01,   1.0743677...e-01,   2.1706796...e+02])
    """

    return tstack([
        CAM16_specification.J,
        CAM16_specification.M,
        CAM16_specification.h,
    ])


def JMh_CAM16_to_CAM16(JMh):
    """
    Converts from *CAM6* :math:`JMh` correlates to *CAM6* specification.

    Parameters
    ----------
    JMh : array_like
         *CAM6* :math:`JMh` correlates.

    Returns
    -------
    CAM6_Specification
        *CAM6* colour appearance model specification.

    Examples
    --------
    >>> JMh = np.array([4.17312079e+01, 1.07436772e-01, 2.17067960e+02])
    >>> JMh_CAM16_to_CAM16(JMh)  # doctest: +ELLIPSIS
    CAM16_Specification(J=41.7312079..., C=None, h=217.06796..., s=None, \
Q=None, M=0.1074367..., H=None, HC=None)
    """

    J, M, h = tsplit(JMh)

    return CAM16_Specification(J=J, M=M, h=h)


def XYZ_to_luminance(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to *luminance* :math:`Y`.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    array_like
        *Luminance* :math:`Y`.

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_luminance(XYZ)  # doctest: +ELLIPSIS
    0.1219722...
    """

    _X, Y, _Z = tsplit(XYZ)

    return Y


def RGB_luminance_to_RGB(Y):
    """
    Converts from *luminance* :math:`Y` to *RGB*.

    Parameters
    ----------
    Y : array_like
        *Luminance* :math:`Y`.

    Returns
    -------
    array_like
        *RGB*.

    Examples
    --------
    >>> RGB_luminance_to_RGB(0.123014562384318)  # doctest: +ELLIPSIS
    array([ 0.1230145...,  0.1230145...,  0.1230145...])
    """

    return tstack([Y, Y, Y])


_DEFAULT_ILLUMINANT = 'D65'
"""
Default automatic colour conversion graph illuminant name.

_DEFAULT_ILLUMINANT : unicode
"""

_DEFAULT_ILLUMINANT_SD = ILLUMINANTS_SDS[_DEFAULT_ILLUMINANT]
"""
Default automatic colour conversion graph illuminant spectral distribution.

_DEFAULT_ILLUMINANT_SD : SpectralDistribution
"""

_DEFAULT_ILLUMINANT_XY = ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    _DEFAULT_ILLUMINANT]
"""
Default automatic colour conversion graph illuminant *CIE xy* chromaticity
coordinates.

_DEFAULT_ILLUMINANT_XY : ndarray
"""

_DEFAULT_ILLUMINANT_XYZ = xy_to_XYZ(_DEFAULT_ILLUMINANT_XY)
"""
Default automatic colour conversion graph illuminant *CIE XYZ* tristimulus
values.

_DEFAULT_ILLUMINANT_XYZ : ndarray
"""

_DEFAULT_RGB_COLOURSPACE = sRGB_COLOURSPACE
"""
Default automatic colour conversion graph *RGB* colourspace.

_DEFAULT_RGB_COLOURSPACE : RGB_COLOURSPACE
"""

CONVERSION_SPECIFICATIONS_DATA = [
    # Colorimetry
    ('Spectral Distribution', 'CIE XYZ',
     partial(sd_to_XYZ, illuminant=_DEFAULT_ILLUMINANT_SD)),
    ('CIE XYZ', 'Spectral Distribution', XYZ_to_sd),
    ('Spectral Distribution', 'Luminous Flux', luminous_flux),
    ('Spectral Distribution', 'Luminous Efficiency', luminous_efficiency),
    ('Spectral Distribution', 'Luminous Efficacy', luminous_efficacy),
    ('CIE XYZ', 'Luminance', XYZ_to_luminance),
    ('Luminance', 'Lightness', lightness),
    ('Lightness', 'Luminance', luminance),
    ('CIE XYZ', 'Whiteness', partial(whiteness,
                                     XYZ_0=_DEFAULT_ILLUMINANT_XYZ)),
    ('CIE XYZ', 'Yellowness', yellowness),
    ('CIE xy', 'Colorimetric Purity',
     partial(colorimetric_purity, xy_n=_DEFAULT_ILLUMINANT_XY)),
    ('CIE xy', 'Complementary Wavelength',
     partial(complementary_wavelength, xy_n=_DEFAULT_ILLUMINANT_XY)),
    ('CIE xy', 'Dominant Wavelength',
     partial(dominant_wavelength, xy_n=_DEFAULT_ILLUMINANT_XY)),
    ('CIE xy', 'Excitation Purity',
     partial(excitation_purity, xy_n=_DEFAULT_ILLUMINANT_XY)),
    ('Wavelength', 'CIE XYZ', wavelength_to_XYZ),
    # Colour Models
    ('CIE XYZ', 'CIE xyY', XYZ_to_xyY),
    ('CIE xyY', 'CIE XYZ', xyY_to_XYZ),
    ('CIE xyY', 'CIE xy', xyY_to_xy),
    ('CIE xy', 'CIE xyY', xy_to_xyY),
    ('CIE XYZ', 'CIE xy', XYZ_to_xy),
    ('CIE xy', 'CIE XYZ', xy_to_XYZ),
    ('CIE XYZ', 'CIE Lab', XYZ_to_Lab),
    ('CIE Lab', 'CIE XYZ', Lab_to_XYZ),
    ('CIE Lab', 'CIE LCHab', Lab_to_LCHab),
    ('CIE LCHab', 'CIE Lab', LCHab_to_Lab),
    ('CIE XYZ', 'CIE Luv', XYZ_to_Luv),
    ('CIE Luv', 'CIE XYZ', Luv_to_XYZ),
    ('CIE Luv', 'CIE Luv uv', Luv_to_uv),
    ('CIE Luv uv', 'CIE Luv', uv_to_Luv),
    ('CIE Luv uv', 'CIE xy', Luv_uv_to_xy),
    ('CIE xy', 'CIE Luv uv', xy_to_Luv_uv),
    ('CIE Luv', 'CIE LCHuv', Luv_to_LCHuv),
    ('CIE LCHuv', 'CIE Luv', LCHuv_to_Luv),
    ('CIE XYZ', 'CIE UCS', XYZ_to_UCS),
    ('CIE UCS', 'CIE XYZ', UCS_to_XYZ),
    ('CIE UCS', 'CIE UCS uv', UCS_to_uv),
    ('CIE UCS uv', 'CIE UCS', uv_to_UCS),
    ('CIE UCS uv', 'CIE xy', UCS_uv_to_xy),
    ('CIE xy', 'CIE UCS uv', xy_to_UCS_uv),
    ('CIE XYZ', 'CIE UVW', XYZ_to_UVW),
    ('CIE UVW', 'CIE XYZ', UVW_to_XYZ),
    ('CIE Lab', 'DIN99', Lab_to_DIN99),
    ('DIN99', 'CIE Lab', DIN99_to_Lab),
    ('CIE XYZ', 'hdr CIELab', XYZ_to_hdr_CIELab),
    ('hdr CIELab', 'CIE XYZ', hdr_CIELab_to_XYZ),
    ('CIE XYZ', 'Hunter Lab',
     partial(
         XYZ_to_Hunter_Lab,
         XYZ_n=HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
         ['D65'].XYZ_n / 100)),
    ('Hunter Lab', 'CIE XYZ',
     partial(
         Hunter_Lab_to_XYZ,
         XYZ_n=HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
         ['D65'].XYZ_n / 100)),
    ('CIE XYZ', 'Hunter Rdab',
     partial(
         XYZ_to_Hunter_Rdab,
         XYZ_n=HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
         ['D65'].XYZ_n / 100)),
    ('Hunter Rdab', 'CIE XYZ',
     partial(
         Hunter_Rdab_to_XYZ,
         XYZ_n=HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
         ['D65'].XYZ_n / 100)),
    ('CIE XYZ', 'IPT', XYZ_to_IPT),
    ('IPT', 'CIE XYZ', IPT_to_XYZ),
    ('CIE XYZ', 'JzAzBz', XYZ_to_JzAzBz),
    ('JzAzBz', 'CIE XYZ', JzAzBz_to_XYZ),
    ('CIE XYZ', 'hdr IPT', XYZ_to_hdr_IPT),
    ('hdr IPT', 'CIE XYZ', hdr_IPT_to_XYZ),
    ('CIE XYZ', 'OSA UCS', XYZ_to_OSA_UCS),
    ('OSA UCS', 'CIE XYZ', OSA_UCS_to_XYZ),
    # RGB Colour Models
    ('CIE XYZ', 'RGB',
     partial(
         XYZ_to_RGB,
         illuminant_XYZ=_DEFAULT_RGB_COLOURSPACE.whitepoint,
         illuminant_RGB=_DEFAULT_RGB_COLOURSPACE.whitepoint,
         XYZ_to_RGB_matrix=_DEFAULT_RGB_COLOURSPACE.XYZ_to_RGB_matrix)),
    ('RGB', 'CIE XYZ',
     partial(
         RGB_to_XYZ,
         illuminant_RGB=_DEFAULT_RGB_COLOURSPACE.whitepoint,
         illuminant_XYZ=_DEFAULT_RGB_COLOURSPACE.whitepoint,
         RGB_to_XYZ_matrix=_DEFAULT_RGB_COLOURSPACE.RGB_to_XYZ_matrix)),
    ('RGB', 'RGB',
     partial(
         RGB_to_RGB,
         input_colourspace=_DEFAULT_RGB_COLOURSPACE,
         output_colourspace=_DEFAULT_RGB_COLOURSPACE)),
    ('RGB', 'HSV', RGB_to_HSV),
    ('HSV', 'RGB', HSV_to_RGB),
    ('RGB', 'HSL', RGB_to_HSL),
    ('HSL', 'RGB', HSL_to_RGB),
    ('CMY', 'RGB', CMY_to_RGB),
    ('RGB', 'CMY', RGB_to_CMY),
    ('CMY', 'CMYK', CMY_to_CMYK),
    ('CMYK', 'CMY', CMYK_to_CMY),
    ('RGB', 'RGB Luminance',
     partial(
         RGB_luminance,
         primaries=_DEFAULT_RGB_COLOURSPACE.primaries,
         whitepoint=_DEFAULT_RGB_COLOURSPACE.whitepoint)),
    ('RGB Luminance', 'RGB', RGB_luminance_to_RGB),
    ('RGB', 'ICTCP', RGB_to_ICTCP),
    ('ICTCP', 'RGB', ICTCP_to_RGB),
    ('RGB', 'Prismatic', RGB_to_Prismatic),
    ('Prismatic', 'RGB', Prismatic_to_RGB),
    ('Output-Referred RGB', 'YCbCr', RGB_to_YCbCr),
    ('YCbCr', 'Output-Referred RGB', YCbCr_to_RGB),
    ('RGB', 'YcCbcCrc', RGB_to_YcCbcCrc),
    ('YcCbcCrc', 'RGB', YcCbcCrc_to_RGB),
    ('Output-Referred RGB', 'YCoCg', RGB_to_YCoCg),
    ('YCoCg', 'Output-Referred RGB', YCoCg_to_RGB),
    ('RGB', 'Output-Referred RGB', cctf_encoding),
    ('Output-Referred RGB', 'RGB', cctf_decoding),
    ('Scene-Referred RGB', 'Output-Referred RGB', cctf_encoding),
    ('Output-Referred RGB', 'Scene-Referred RGB', cctf_decoding),
    ('CIE XYZ', 'sRGB', XYZ_to_sRGB),
    ('sRGB', 'CIE XYZ', sRGB_to_XYZ),
    # Colour Notation Systems
    ('Output-Referred RGB', 'Hexadecimal', RGB_to_HEX),
    ('Hexadecimal', 'Output-Referred RGB', HEX_to_RGB),
    ('CIE xyY', 'Munsell Colour', xyY_to_munsell_colour),
    ('Munsell Colour', 'CIE xyY', munsell_colour_to_xyY),
    ('Luminance', 'Munsell Value', munsell_value),
    ('Munsell Value', 'Luminance', partial(luminance, method='ASTM D1535')),
    # Colour Quality
    ('Spectral Distribution', 'CRI', colour_rendering_index),
    ('Spectral Distribution', 'CQS', colour_quality_scale),
    # Colour Temperature
    ('CCT', 'CIE UCS uv', CCT_to_uv),
    ('CCT', 'CIE xy', CCT_to_xy),
    ('CIE UCS uv', 'CCT', uv_to_CCT),
    ('CIE xy', 'CCT', xy_to_CCT),
    # Advanced Colorimetry
    ('CIE XYZ', 'Hunt',
     partial(
         XYZ_to_Hunt,
         XYZ_w=_DEFAULT_ILLUMINANT_XYZ,
         XYZ_b=_DEFAULT_ILLUMINANT_XYZ,
         L_A=80 * 0.2)),
    ('CIE XYZ', 'ATD95',
     partial(
         XYZ_to_ATD95,
         XYZ_0=_DEFAULT_ILLUMINANT_XYZ,
         Y_0=80 * 0.2,
         k_1=0,
         k_2=(15 + 50) / 2)),
    ('CIE XYZ', 'CIECAM02',
     partial(
         XYZ_to_CIECAM02,
         XYZ_w=_DEFAULT_ILLUMINANT_XYZ,
         L_A=64 / np.pi * 0.2,
         Y_b=20)),
    ('CIECAM02', 'CIE XYZ',
     partial(
         CIECAM02_to_XYZ,
         XYZ_w=_DEFAULT_ILLUMINANT_XYZ,
         L_A=64 / np.pi * 0.2,
         Y_b=20)),
    ('CIECAM02', 'CIECAM02 JMh', CIECAM02_to_JMh_CIECAM02),
    ('CIECAM02 JMh', 'CIECAM02', JMh_CIECAM02_to_CIECAM02),
    ('CIE XYZ', 'CAM16',
     partial(
         XYZ_to_CAM16,
         XYZ_w=_DEFAULT_ILLUMINANT_XYZ,
         L_A=64 / np.pi * 0.2,
         Y_b=20)),
    ('CAM16', 'CIE XYZ',
     partial(
         CAM16_to_XYZ,
         XYZ_w=_DEFAULT_ILLUMINANT_XYZ,
         L_A=64 / np.pi * 0.2,
         Y_b=20)),
    ('CAM16', 'CAM16 JMh', CAM16_to_JMh_CAM16),
    ('CAM16 JMh', 'CAM16', JMh_CAM16_to_CAM16),
    ('CIE XYZ', 'LLAB',
     partial(XYZ_to_LLAB, XYZ_0=_DEFAULT_ILLUMINANT_XYZ, Y_b=80 * 0.2, L=80)),
    ('CIE XYZ', 'Nayatani95',
     partial(
         XYZ_to_Nayatani95,
         XYZ_n=_DEFAULT_ILLUMINANT_XYZ,
         Y_o=0.2,
         E_o=1000,
         E_or=1000)),
    ('CIE XYZ', 'RLAB', XYZ_to_RLAB),
    ('CIECAM02 JMh', 'CAM02LCD', JMh_CIECAM02_to_CAM02LCD),
    ('CAM02LCD', 'CIECAM02 JMh', CAM02LCD_to_JMh_CIECAM02),
    ('CIECAM02 JMh', 'CAM02SCD', JMh_CIECAM02_to_CAM02SCD),
    ('CAM02SCD', 'CIECAM02 JMh', CAM02SCD_to_JMh_CIECAM02),
    ('CIECAM02 JMh', 'CAM02UCS', JMh_CIECAM02_to_CAM02UCS),
    ('CAM02UCS', 'CIECAM02 JMh', CAM02UCS_to_JMh_CIECAM02),
    ('CAM16 JMh', 'CAM16LCD', JMh_CAM16_to_CAM16LCD),
    ('CAM16LCD', 'CAM16 JMh', CAM16LCD_to_JMh_CAM16),
    ('CAM16 JMh', 'CAM16SCD', JMh_CAM16_to_CAM16SCD),
    ('CAM16SCD', 'CAM16 JMh', CAM16SCD_to_JMh_CAM16),
    ('CAM16 JMh', 'CAM16UCS', JMh_CAM16_to_CAM16UCS),
    ('CAM16UCS', 'CAM16 JMh', CAM16UCS_to_JMh_CAM16),
]
"""
Automatic colour conversion graph specifications data describing two nodes and
the edge in the graph.

CONVERSION_SPECIFICATIONS_DATA : list
"""

CONVERSION_SPECIFICATIONS = [
    ConversionSpecification(*specification)
    for specification in CONVERSION_SPECIFICATIONS_DATA
]
"""
Automatic colour conversion graph specifications describing two nodes and
the edge in the graph.

CONVERSION_SPECIFICATIONS : list
"""

CONVERSION_GRAPH_NODE_LABELS = {
    specification[0].lower(): specification[0]
    for specification in CONVERSION_SPECIFICATIONS_DATA
}
"""
Automatic colour conversion graph node labels.

CONVERSION_GRAPH_NODE_LABELS : dict
"""

CONVERSION_GRAPH_NODE_LABELS.update({
    specification[1].lower(): specification[1]
    for specification in CONVERSION_SPECIFICATIONS_DATA
})


def _build_graph():
    """
    Builds the automatic colour conversion graph.

    Returns
    -------
    DiGraph
         Automatic colour conversion graph.
    """

    graph = nx.DiGraph()

    for specification in CONVERSION_SPECIFICATIONS:
        graph.add_edge(
            specification.source,
            specification.target,
            conversion_function=specification.conversion_function)

    return graph


CONVERSION_GRAPH = _build_graph() if is_networkx_installed() else None
"""
Automatic colour conversion graph.

CONVERSION_GRAPH : DiGraph
"""


def _conversion_path(source, target):
    """
    Returns the conversion path from the source node to the target node in the
    automatic colour conversion graph.

    Parameters
    ----------
    source : unicode
        Source node.
    target : unicode
        Target node.

    Returns
    -------
    list
        Conversion path from the source node to the target node, i.e. a list of
        conversion function callables.

    Examples
    --------
    >>> _conversion_path('cie lab', 'cct')
    ... # doctest: +ELLIPSIS
    [<function Lab_to_XYZ at 0x...>, <function XYZ_to_xy at 0x...>, \
<function xy_to_CCT at 0x...>]
    """
    if is_networkx_installed(raise_exception=True):  # pragma: no cover
        path = nx.shortest_path(CONVERSION_GRAPH, source, target)

        return [
            CONVERSION_GRAPH.get_edge_data(a, b)['conversion_function']
            for a, b in zip(path[:-1], path[1:])
        ]


def _lower_order_function(callable_):
    """
    Returns the lower order function associated with given callable, i.e.
    the function wrapped by a partial object.

    Parameters
    ----------
    callable_ : callable
        Callable to return the lower order function.

    Returns
    -------
    callable
        Lower order function or given callable if no lower order function
        exists.
    """

    return callable_.func if isinstance(callable_, partial) else callable_


def describe_conversion_path(source,
                             target,
                             mode='Short',
                             width=79,
                             padding=3,
                             print_callable=print,
                             **kwargs):
    """
    Describes the conversion path from source colour representation to target
    colour representation using the automatic colour conversion graph.

    Parameters
    ----------
    source : unicode
        Source colour representation, i.e. the source node in the automatic
        colour conversion graph.
    target : unicode
        Target colour representation, i.e. the target node in the automatic
        colour conversion graph.
    mode : unicode, optional
        **{'Short', 'Long', 'Extended'}**,
        Verbose mode: *Short* describes the conversion path, *Long* provides
        details about the arguments, definitions signatures and output values,
        *Extended* appends the definitions documentation.
    width : int, optional
        Message box width.
    padding : unicode, optional
        Padding on each sides of the message.
    print_callable : callable, optional
        Callable used to print the message box.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.convert`},
        Please refer to the documentation of the previously listed definition.

    Examples
    --------
    >>> describe_conversion_path('Spectral Distribution', 'sRGB', width=75)
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
        signature_inspection = inspect.getargspec

    source, target, mode = source.lower(), target.lower(), mode.lower()
    width = (79 + 2 + 2 * 3 - 4) if mode == 'extended' else width

    conversion_path = _conversion_path(source, target)

    message_box(
        '[ Conversion Path ]\n\n{0}'.format(' --> '.join([
            '"{0}"'.format(
                _lower_order_function(conversion_function).__name__)
            for conversion_function in conversion_path
        ])), width, padding, print_callable)

    for conversion_function in conversion_path:
        conversion_function_name = _lower_order_function(
            conversion_function).__name__

        # Filtering compatible keyword arguments passed directly and
        # irrespective of any conversion function name.
        filtered_kwargs = filter_kwargs(conversion_function, **kwargs)

        # Filtering keyword arguments passed as dictionary with the
        # conversion function name.
        filtered_kwargs.update(kwargs.get(conversion_function_name, {}))

        return_value = filtered_kwargs.pop('return', None)

        if mode in ('long', 'extended'):
            message = (
                '[ "{0}" ]'
                '\n\n[ Signature ]\n\n{1}').format(
                    _lower_order_function(conversion_function).__name__,
                    pformat(
                        signature_inspection(
                            _lower_order_function(conversion_function))))

            if filtered_kwargs:
                message += '\n\n[ Filtered Arguments ]\n\n{0}'.format(
                    pformat(filtered_kwargs))

            if mode in ('extended', ):
                message += '\n\n[ Documentation ]\n\n{0}'.format(
                    textwrap.dedent(
                        str(
                            _lower_order_function(conversion_function)
                            .__doc__)).strip())

            if return_value is not None:
                message += '\n\n[ Conversion Output ]\n\n{0}'.format(
                    return_value)

            message_box(message, width, padding, print_callable)


@domain_range_scale('1')
def convert(a, source, target, **kwargs):
    """
    Converts given object :math:`a` from source colour representation to target
    colour representation using the automatic colour conversion graph.

    The conversion is performed by finding the shortest path in a
    `NetworkX <https://networkx.github.io/>`_ :class:`DiGraph` class instance.

    The conversion path adopts the **'1'** domain-range scale and the object
    :math:`a` is expected to be *soft* normalised accordingly. For example,
    *CIE XYZ* tristimulus values arguments for use with the *CAM16* colour
    appearance model should be in domain `[0, 1]` instead of the domain
    `[0, 100]` used with the **'Reference'** domain-range scale. The arguments
    are typically converted as follows:

    -   *Scalars* in domain-range `[0, 10]`, e.g *Munsell Value* are
        scaled by *10*.
    -   *Percentages* in domain-range `[0, 100]` are scaled by *100*.
    -   *Degrees* in domain-range `[0, 360]` are scaled by *360*.
    -   *Integers* in domain-range `[0, 2**n -1]` where `n` is the bit
        depth are scaled by *2**n -1*.

    See the `Domain-Range Scales <../basics.html#domain-range-scales>`_ page
    for more information.

    Parameters
    ----------
    a : array_like or numeric or SpectralDistribution
        Object :math:`a` to convert. If :math:`a` represents a reflectance,
        transmittance or absorptance value, the expectation is that it is
        viewed under *CIE Standard Illuminant D Series* *D65*. The illuminant
        can be changed on a per definition basis along the conversion path.
    source : unicode
        Source colour representation, i.e. the source node in the automatic
        colour conversion graph.
    target : unicode
        Target colour representation, i.e. the target node in the automatic
        colour conversion graph.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {'\\*'},
        Please refer to the documentation of the supported conversion
        definitions.

        Arguments for the conversion definitions are passed as keyword
        arguments whose names is those of the conversion definitions and values
        set as dictionaries. For example, in the conversion from spectral
        distribution to *sRGB* colourspace, passing arguments to the
        :func:`colour.sd_to_XYZ` definition is done as follows::

            convert(sd, 'Spectral Distribution', 'sRGB', sd_to_XYZ={\
'illuminant': ILLUMINANTS_SDS['FL2']})

        It is also possible to pass keyword arguments directly to the various
        conversion definitions irrespective of their name. This is
        ``dangerous`` and could cause unexpected behaviour because of
        unavoidable discrepancies with the underlying
        :func:`colour.utilities.filter_kwargs` definition between Python 2.7
        and 3.x. Using this direct keyword arguments passing mechanism might
        also ends up passing incompatible arguments to a given conversion
        definition. Consider the following conversion::

             convert(sd, 'Spectral Distribution', 'sRGB', 'illuminant': \
ILLUMINANTS_SDS['FL2'])

        Because both the :func:`colour.sd_to_XYZ` and
        :func:`colour.XYZ_to_sRGB` definitions have an *illuminant* argument,
        `ILLUMINANTS_SDS['FL2']` will be passed to both of them and will raise
        an exception in the :func:`colour.XYZ_to_sRGB` definition. This will
        be addressed in the future by either catching the exception and trying
        a new time without the keyword argument or more elegantly via type
        checking.

        With that in mind, this mechanism offers some good benefits: For
        example, it allows defining a conversion from *CIE XYZ* colourspace to
        *n* different colour models while passing an illuminant argument but
        without having to explicitly define all the explicit conversion
        definition arguments::

            a = np.array([0.20654008, 0.12197225, 0.05136952])
            illuminant = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']\
['D65']
            for model in ('CIE xyY', 'CIE Lab'):
                convert(a, 'CIE XYZ', model, illuminant=illuminant)

        Instead of::

            for model in ('CIE xyY', 'CIE Lab'):
                convert(a, 'CIE XYZ', model, XYZ_to_xyY={'illuminant': \
illuminant}, XYZ_to_Lab={'illuminant': illuminant})

        Mixing both approaches is possible for the brevity benefits. It is made
        possible because the keyword arguments directly passed are filtered
        first and then the resulting dict is updated with the explicit
        conversion definition arguments::

            illuminant = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']\
['D65']
             convert(sd, 'Spectral Distribution', 'sRGB', 'illuminant': \
ILLUMINANTS_SDS['FL2'], XYZ_to_sRGB={'illuminant': illuminant})

        For inspection purposes, verbose is enabled by passing arguments to the
        :func:`colour.describe_conversion_path` definition via the ``verbose``
        keyword argument as follows::

            convert(sd, 'Spectral Distribution', 'sRGB', \
verbose={'mode': 'Long'})

    Returns
    -------
    ndarray or numeric or SpectralDistribution
        Converted object :math:`a`.

    Warnings
    --------
    The domain-range scale is **'1'** and cannot be changed.

    Notes
    -----
    -   Various defaults have been systematically adopted compared to the
        low-level *Colour* API:

        -   The default illuminant for the computation is
            *CIE Standard Illuminant D Series* *D65*. It can be changed on a
            per definition basis along the conversion path.
        -   The default *RGB* colourspace is the *sRGB* colourspace. It can
            also be changed on a per definition basis along the conversion
            path.
        -   Most of the colour appearance models have defaults set according to
            *IEC 61966-2-1:1999* viewing conditions, i.e. *sRGB* 64 Lux ambiant
            illumination, 80 :math:`cd/m^2`, adapting field luminance about
            20% of a white object in the scene.

    -   The **RGB** colour representation is assumed to be linear and
        representing *scene-referred* imagery. To convert such *RGB* values to
        *output-referred* (*display-referred*) imagery, i.e. encode the *RGB*
        values using an encoding colour component transfer function
        (Encoding CCTF) / opto-electronic transfer function (OETF / OECF), the
        **Output-Referred RGB** representation must be used.

    Examples
    --------
    >>> from colour import COLOURCHECKERS_SDS
    >>> sd = COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']
    >>> convert(sd, 'Spectral Distribution', 'sRGB',
    ...     verbose={'mode': 'Short', 'width': 75})
    ... # doctest: +ELLIPSIS
    ===========================================================================
    *                                                                         *
    *   [ Conversion Path ]                                                   *
    *                                                                         *
    *   "sd_to_XYZ" --> "XYZ_to_sRGB"                                         *
    *                                                                         *
    ===========================================================================
    array([ 0.4567579...,  0.3098698...,  0.2486192...])
    >>> illuminant = ILLUMINANTS_SDS['FL2']
    >>> convert(sd, 'Spectral Distribution', 'sRGB',
    ...     sd_to_XYZ={'illuminant': illuminant})
    ... # doctest: +ELLIPSIS
    array([ 0.4792457...,  0.3167696...,  0.1736272...])
    >>> a = np.array([0.45675795, 0.30986982, 0.24861924])
    >>> convert(a, 'Output-Referred RGB', 'CAM16UCS')
    ... # doctest: +ELLIPSIS
    array([ 0.3999481...,  0.0920655...,  0.0812752...])
    >>> a = np.array([0.39994811, 0.09206558, 0.08127526])
    >>> convert(a, 'CAM16UCS', 'sRGB', verbose={'mode': 'Short', 'width': 75})
    ... # doctest: +ELLIPSIS
    ===========================================================================
    *                                                                         *
    *   [ Conversion Path ]                                                   *
    *                                                                         *
    *   "UCS_Luo2006_to_JMh_CIECAM02" --> "JMh_CAM16_to_CAM16" -->            *
    *   "CAM16_to_XYZ" --> "XYZ_to_sRGB"                                      *
    *                                                                         *
    ===========================================================================
    array([ 0.4567576...,  0.3098826...,  0.2486222...])
    """

    # TODO: Remove the following warning whenever the automatic colour
    # conversion graph implementation is considered stable.
    usage_warning(
        'The "Automatic Colour Conversion Graph" is a beta feature, be '
        'mindful of this when using it. Please report any unexpected '
        'behaviour and do not hesitate to ask any questions should they arise.'
        '\nThis warning can be disabled with the '
        '"colour.utilities.suppress_warnings" context manager as follows:\n'
        'with colour.utilities.suppress_warnings(colour_usage_warnings=True): '
        '\n    convert(*args, **kwargs)')

    source, target = source.lower(), target.lower()

    conversion_path = _conversion_path(source, target)

    verbose_kwargs = copy(kwargs)
    for conversion_function in conversion_path:
        conversion_function_name = _lower_order_function(
            conversion_function).__name__

        # Filtering compatible keyword arguments passed directly and
        # irrespective of any conversion function name.
        filtered_kwargs = filter_kwargs(conversion_function, **kwargs)

        # Filtering keyword arguments passed as dictionary with the
        # conversion function name.
        filtered_kwargs.update(kwargs.get(conversion_function_name, {}))

        a = conversion_function(a, **filtered_kwargs)

        if conversion_function_name in verbose_kwargs:
            verbose_kwargs[conversion_function_name]['return'] = a
        else:
            verbose_kwargs[conversion_function_name] = {'return': a}

    if 'verbose' in verbose_kwargs:
        verbose_kwargs.update(verbose_kwargs.pop('verbose'))
        describe_conversion_path(source, target, **verbose_kwargs)

    return a
