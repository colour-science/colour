# -*- coding: utf-8 -*-
"""
Y'CbCr Colour Encoding
======================

Defines the *Y'CbCr* colour encoding related transformations:

-   :func:`colour.RGB_to_YCbCr`
-   :func:`colour.YCbCr_to_RGB`
-   :func:`colour.RGB_to_YcCbcCrc`
-   :func:`colour.YcCbcCrc_to_RGB`

Notes
-----
-   *Y'CbCr* is not an absolute colourspace.

See Also
--------
`YCbCr Colours Encoding Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/ycbcr.ipynb>`_

References
----------
-   :cite:`InternationalTelecommunicationUnion2011e` : International
    Telecommunication Union. (2011). Recommendation ITU-T T.871 - Information
    technology - Digital compression and coding of continuous-tone still
    images: JPEG File Interchange Format (JFIF). Retrieved from
    https://www.itu.int/rec/dologin_pub.asp?lang=e&\
id=T-REC-T.871-201105-I!!PDF-E&type=items
-   :cite:`InternationalTelecommunicationUnion2015h` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.2020 - Parameter
    values for ultra-high definition television systems for production and
    international programme exchange. Retrieved from
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2020-2-201510-I!!PDF-E.pdf
-   :cite:`InternationalTelecommunicationUnion2015i` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.709-6 - Parameter
    values for the HDTV standards for production and international programme
    exchange BT Series Broadcasting service. Retrieved from
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.709-6-201506-I!!PDF-E.pdf
-   :cite:`SocietyofMotionPictureandTelevisionEngineers1999b` : Society of
    Motion Picture and Television Engineers. (1999). ANSI/SMPTE 240M-1995 -
    Signal Parameters - 1125-Line High-Definition Production Systems. Retrieved
    from http://car.france3.mars.free.fr/HD/INA- 26 jan 06/\
SMPTE normes et confs/s240m.pdf
-   :cite:`Wikipedia2004d` : Wikipedia. (2004). YCbCr. Retrieved February 29,
    2016, from https://en.wikipedia.org/wiki/YCbCr
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from colour.models.rgb.transfer_functions import (CV_range, oetf_BT2020,
                                                  eotf_BT2020)
from colour.utilities import (CaseInsensitiveMapping, as_float_array,
                              domain_range_scale, from_range_1, to_domain_1,
                              tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Development'

__all__ = [
    'YCBCR_WEIGHTS', 'YCbCr_ranges', 'RGB_to_YCbCr', 'YCbCr_to_RGB',
    'RGB_to_YcCbcCrc', 'YcCbcCrc_to_RGB'
]

YCBCR_WEIGHTS = CaseInsensitiveMapping({
    'ITU-R BT.601': np.array([0.2990, 0.1140]),
    'ITU-R BT.709': np.array([0.2126, 0.0722]),
    'ITU-R BT.2020': np.array([0.2627, 0.0593]),
    'SMPTE-240M': np.array([0.2122, 0.0865])
})
"""
Luma weightings presets.

References
----------
:cite:`InternationalTelecommunicationUnion2011e`,
:cite:`InternationalTelecommunicationUnion2015i`,
:cite:`InternationalTelecommunicationUnion2015h`,
:cite:`SocietyofMotionPictureandTelevisionEngineers1999b`,
:cite:`Wikipedia2004d`

YCBCR_WEIGHTS : dict
    **{'ITU-R BT.601', 'ITU-R BT.709', 'ITU-R BT.2020', 'SMPTE-240M}**
"""


def YCbCr_ranges(bits, is_legal, is_int):
    """"
    Returns the *Y'CbCr* colour encoding ranges array for given bit depth,
    range legality and representation.

    Parameters
    ----------
    bits : int
        Bit depth of the *Y'CbCr* colour encoding ranges array.
    is_legal : bool
        Whether the *Y'CbCr* colour encoding ranges array is legal.
    is_int : bool
        Whether the *Y'CbCr* colour encoding ranges array represents integer
        code values.

    Returns
    -------
    ndarray
        *Y'CbCr* colour encoding ranges array.

    Examples
    --------
    >>> YCbCr_ranges(8, True, True)
    array([ 16, 235,  16, 240])
    >>> YCbCr_ranges(8, True, False)  # doctest: +ELLIPSIS
    array([ 0.0627451...,  0.9215686...,  0.0627451...,  0.9411764...])
    >>> YCbCr_ranges(10, False, False)
    array([ 0. ,  1. , -0.5,  0.5])
    """

    if is_legal:
        ranges = np.array([16, 235, 16, 240])
        ranges *= 2 ** (bits - 8)
    else:
        ranges = np.array([0, 2 ** bits - 1, 0, 2 ** bits - 1])

    if not is_int:
        ranges = ranges.astype(DEFAULT_FLOAT_DTYPE) / (2 ** bits - 1)

    if is_int and not is_legal:
        ranges[3] = 2 ** bits

    if not is_int and not is_legal:
        ranges[2] = -0.5
        ranges[3] = 0.5

    return ranges


def RGB_to_YCbCr(RGB,
                 K=YCBCR_WEIGHTS['ITU-R BT.709'],
                 in_bits=10,
                 in_legal=False,
                 in_int=False,
                 out_bits=8,
                 out_legal=True,
                 out_int=False,
                 **kwargs):
    """
    Converts an array of *R'G'B'* values to the corresponding *Y'CbCr* colour
    encoding values array.

    Parameters
    ----------
    RGB : array_like
        Input *R'G'B'* array of floats or integer values.
    K : array_like, optional
        Luma weighting coefficients of red and blue. See
        :attr:`colour.YCBCR_WEIGHTS` for presets. Default is
        *(0.2126, 0.0722)*, the weightings for *ITU-R BT.709*.
    in_bits : int, optional
        Bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Default is *10*.
    in_legal : bool, optional
        Whether to treat the input values as legal range. Default is *False*.
    in_int : bool, optional
        Whether to treat the input values as ``in_bits`` integer code values.
        Default is *False*.
    out_bits : int, optional
        Bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Ignored if ``out_legal`` and
        ``out_int`` are both *False*. Default is *8*.
    out_legal : bool, optional
        Whether to return legal range values. Default is *True*.
    out_int : bool, optional
        Whether to return values as ``out_bits`` integer code values. Default
        is *False*.

    Other Parameters
    ----------------
    in_range : array_like, optional
        Array overriding the computed range such as
        *in_range = (RGB_min, RGB_max)*. If ``in_range`` is undefined,
        *RGB_min* and *RGB_max* will be computed using :func:`colour.CV_range`
        definition.
    out_range : array_like, optional
        Array overriding the computed range such as
        *out_range = (Y_min, Y_max, C_min, C_max)`. If ``out_range`` is
        undefined, *Y_min*, *Y_max*, *C_min* and *C_max* will be computed
        using :func:`colour.models.rgb.ycbcr.YCbCr_ranges` definition.

    Returns
    -------
    ndarray
        *Y'CbCr* colour encoding array of integer or float values.

    Warning
    -------
    For *Recommendation ITU-R BT.2020*, :func:`colour.RGB_to_YCbCr` definition
    is only applicable to the non-constant luminance implementation.
    :func:`colour.RGB_to_YcCbcCrc` definition should be used for the constant
    luminance case as per :cite:`InternationalTelecommunicationUnion2015h`.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``RGB``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``YCbCr``      | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    \\* This definition has input and output integer switches, thus the
    domain-range scale information is only given for the floating point mode.

    -   The default arguments, ``**{'in_bits': 10, 'in_legal': False,
        'in_int': False, 'out_bits': 8, 'out_legal': True, 'out_int': False}``
        transform a float *R'G'B'* input array normalised to domain [0, 1]
        (``in_bits`` is ignored) to a float *Y'CbCr* output array where *Y'* is
        normalised to range [16 / 255, 235 / 255] and *Cb* and *Cr* are
        normalised to range [16 / 255, 240./255]. The float values are
        calculated based on an [0, 255] integer range, but no 8-bit
        quantisation or clamping are performed.

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2011e`,
    :cite:`InternationalTelecommunicationUnion2015i`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers1999b`,
    :cite:`Wikipedia2004d`

    Examples
    --------
    >>> RGB = np.array([1.0, 1.0, 1.0])
    >>> RGB_to_YCbCr(RGB)  # doctest: +ELLIPSIS
    array([ 0.9215686...,  0.5019607...,  0.5019607...])

    Matching float output of The Foundry Nuke's Colorspace node set to YCbCr:

    >>> RGB_to_YCbCr(RGB,
    ...              out_range=(16 / 255, 235 / 255, 15.5 / 255, 239.5 / 255))
    ... # doctest: +ELLIPSIS
    array([ 0.9215686...,  0.5       ,  0.5       ])

    Matching float output of The Foundry Nuke's Colorspace node set to YPbPr:

    >>> RGB_to_YCbCr(RGB, out_legal=False, out_int=False)
    ... # doctest: +ELLIPSIS
    array([ 1.,  0.,  0.])

    Creating integer code values as per standard 10-bit SDI:

    >>> RGB_to_YCbCr(RGB, out_legal=True, out_bits=10, out_int=True)
    array([940, 512, 512])

    For JFIF JPEG conversion as per ITU-T T.871
    :cite:`InternationalTelecommunicationUnion2011e`:

    >>> RGB = np.array([102, 0, 51])
    >>> RGB_to_YCbCr(RGB, K=YCBCR_WEIGHTS['ITU-R BT.601'], in_range=(0, 255),
    ...              out_range=(0, 255, 0, 256), out_int=True)
    array([ 36, 136, 175])

    Note the use of 256 for the max *Cb / Cr* value, which is required so that
    the *Cb* and *Cr* output is centered about 128. Using 255 centres it
    about 127.5, meaning that there is no integer code value to represent
    achromatic colours. This does however create the possibility of output
    integer codes with value of 256, which cannot be stored in 8-bit integer
    representation. Recommendation ITU-T T.871 specifies these should be
    clamped to 255.

    These JFIF JPEG ranges are also obtained as follows:

    >>> RGB_to_YCbCr(RGB, K=YCBCR_WEIGHTS['ITU-R BT.601'], in_bits=8,
    ...              in_int=True, out_legal=False, out_int=True)
    array([ 36, 136, 175])
    """

    if in_int:
        RGB = as_float_array(RGB)
    else:
        RGB = to_domain_1(RGB)

    Kr, Kb = K
    RGB_min, RGB_max = kwargs.get('in_range',
                                  CV_range(in_bits, in_legal, in_int))
    Y_min, Y_max, C_min, C_max = kwargs.get(
        'out_range', YCbCr_ranges(out_bits, out_legal, out_int))

    RGB_float = RGB.astype(DEFAULT_FLOAT_DTYPE) - RGB_min
    RGB_float *= 1 / (RGB_max - RGB_min)
    R, G, B = tsplit(RGB_float)

    Y = Kr * R + (1 - Kr - Kb) * G + Kb * B
    Cb = 0.5 * (B - Y) / (1 - Kb)
    Cr = 0.5 * (R - Y) / (1 - Kr)
    Y *= Y_max - Y_min
    Y += Y_min
    Cb *= C_max - C_min
    Cr *= C_max - C_min
    Cb += (C_max + C_min) / 2
    Cr += (C_max + C_min) / 2

    YCbCr = tstack([Y, Cb, Cr])
    YCbCr = np.round(YCbCr).astype(
        DEFAULT_INT_DTYPE) if out_int else from_range_1(YCbCr)

    return YCbCr


def YCbCr_to_RGB(YCbCr,
                 K=YCBCR_WEIGHTS['ITU-R BT.709'],
                 in_bits=8,
                 in_legal=True,
                 in_int=False,
                 out_bits=10,
                 out_legal=False,
                 out_int=False,
                 **kwargs):
    """
    Converts an array of *Y'CbCr* colour encoding values to the corresponding
    *R'G'B'* values array.

    Parameters
    ----------
    YCbCr : array_like
        Input *Y'CbCr* colour encoding array of integer or float values.
    K : array_like, optional
        Luma weighting coefficients of red and blue. See
        :attr:`colour.YCBCR_WEIGHTS` for presets. Default is
        *(0.2126, 0.0722)*, the weightings for *ITU-R BT.709*.
    in_bits : int, optional
        Bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Default is *8*.
    in_legal : bool, optional
        Whether to treat the input values as legal range. Default is *True*.
    in_int : bool, optional
        Whether to treat the input values as ``in_bits`` integer code values.
        Default is *False*.
    out_bits : int, optional
        Bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Ignored if ``out_legal`` and
        ``out_int`` are both *False*. Default is *10*.
    out_legal : bool, optional
        Whether to return legal range values. Default is *False*.
    out_int : bool, optional
        Whether to return values as ``out_bits`` integer code values. Default
        is *False*.

    Other Parameters
    ----------------
    in_range : array_like, optional
        Array overriding the computed range such as
        *in_range = (Y_min, Y_max, C_min, C_max)*. If ``in_range`` is
        undefined, *Y_min*, *Y_max*, *C_min* and *C_max* will be computed using
        :func:`colour.models.rgb.ycbcr.YCbCr_ranges` definition.
    out_range : array_like, optional
        Array overriding the computed range such as
        *out_range = (RGB_min, RGB_max)*. If ``out_range`` is undefined,
        *RGB_min* and *RGB_max* will be computed using :func:`colour.CV_range`
        definition.

    Returns
    -------
    ndarray
        *R'G'B'* array of integer or float values.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``YCbCr``      | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``RGB``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    \\* This definition has input and output integer switches, thus the
    domain-range scale information is only given for the floating point mode.

    Warning
    -------
    For *Recommendation ITU-R BT.2020*, :func:`colour.YCbCr_to_RGB`
    definition is only applicable to the non-constant luminance implementation.
    :func:`colour.YcCbcCrc_to_RGB` definition should be used for the constant
    luminance case as per :cite:`InternationalTelecommunicationUnion2015h`.

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2011e`,
    :cite:`InternationalTelecommunicationUnion2015i`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers1999b`,
    :cite:`Wikipedia2004d`

    Examples
    --------
    >>> YCbCr = np.array([502, 512, 512])
    >>> YCbCr_to_RGB(YCbCr, in_bits=10, in_legal=True, in_int=True)
    array([ 0.5,  0.5,  0.5])
    """

    if in_int:
        YCbCr = as_float_array(YCbCr)
    else:
        YCbCr = to_domain_1(YCbCr)

    Y, Cb, Cr = tsplit(YCbCr.astype(DEFAULT_FLOAT_DTYPE))
    Kr, Kb = K
    Y_min, Y_max, C_min, C_max = kwargs.get(
        'in_range', YCbCr_ranges(in_bits, in_legal, in_int))
    RGB_min, RGB_max = kwargs.get('out_range',
                                  CV_range(out_bits, out_legal, out_int))

    Y -= Y_min
    Cb -= (C_max + C_min) / 2
    Cr -= (C_max + C_min) / 2
    Y *= 1 / (Y_max - Y_min)
    Cb *= 1 / (C_max - C_min)
    Cr *= 1 / (C_max - C_min)
    R = Y + (2 - 2 * Kr) * Cr
    B = Y + (2 - 2 * Kb) * Cb
    G = (Y - Kr * R - Kb * B) / (1 - Kr - Kb)

    RGB = tstack([R, G, B])
    RGB *= RGB_max - RGB_min
    RGB += RGB_min
    RGB = np.round(RGB).astype(DEFAULT_INT_DTYPE) if out_int else from_range_1(
        RGB)

    return RGB


def RGB_to_YcCbcCrc(RGB,
                    out_bits=10,
                    out_legal=True,
                    out_int=False,
                    is_12_bits_system=False,
                    **kwargs):
    """
    Converts an array of *RGB* linear values to the corresponding *Yc'Cbc'Crc'*
    colour encoding values array.

    Parameters
    ----------
    RGB : array_like
        Input *RGB* array of linear float values.
    out_bits : int, optional
        Bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Ignored if ``out_legal`` and
        ``out_int`` are both *False*. Default is *10*.
    out_legal : bool, optional
        Whether to return legal range values. Default is *True*.
    out_int : bool, optional
        Whether to return values as ``out_bits`` integer code values. Default
        is *False*.
    is_12_bits_system : bool, optional
        *Recommendation ITU-R BT.2020* OETF (OECF) adopts different parameters
        for 10 and 12 bit systems. Default is *False*.

    Other Parameters
    ----------------
    out_range : array_like, optional
        Array overriding the computed range such as
        *out_range = (Y_min, Y_max, C_min, C_max)*. If ``out_range`` is
        undefined, *Y_min*, *Y_max*, *C_min* and *C_max* will be computed
        using :func:`colour.models.rgb.ycbcr.YCbCr_ranges` definition.

    Returns
    -------
    ndarray
        *Yc'Cbc'Crc'* colour encoding array of integer or float values.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``RGB``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``YcCbcCrc``   | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    \\* This definition has input and output integer switches, thus the
    domain-range scale information is only given for the floating point mode.

    Warning
    -------
    This definition is specifically for usage with
    *Recommendation ITU-R BT.2020* when adopting the constant luminance
    implementation.

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2015h`, :cite:`Wikipedia2004d`

    Examples
    --------
    >>> RGB = np.array([0.18, 0.18, 0.18])
    >>> RGB_to_YcCbcCrc(RGB, out_legal=True, out_bits=10, out_int=True,
    ...                 is_12_bits_system=False)
    array([422, 512, 512])
    """

    R, G, B = tsplit(to_domain_1(RGB))
    Y_min, Y_max, C_min, C_max = kwargs.get(
        'out_range', YCbCr_ranges(out_bits, out_legal, out_int))

    Yc = 0.2627 * R + 0.6780 * G + 0.0593 * B

    with domain_range_scale('ignore'):
        Yc = oetf_BT2020(Yc, is_12_bits_system=is_12_bits_system)
        R = oetf_BT2020(R, is_12_bits_system=is_12_bits_system)
        B = oetf_BT2020(B, is_12_bits_system=is_12_bits_system)

    Cbc = np.where((B - Yc) <= 0, (B - Yc) / 1.9404, (B - Yc) / 1.5816)
    Crc = np.where((R - Yc) <= 0, (R - Yc) / 1.7184, (R - Yc) / 0.9936)
    Yc *= Y_max - Y_min
    Yc += Y_min
    Cbc *= C_max - C_min
    Crc *= C_max - C_min
    Cbc += (C_max + C_min) / 2
    Crc += (C_max + C_min) / 2

    YcCbcCrc = tstack([Yc, Cbc, Crc])
    YcCbcCrc = (np.round(YcCbcCrc).astype(DEFAULT_INT_DTYPE)
                if out_int else from_range_1(YcCbcCrc))

    return YcCbcCrc


def YcCbcCrc_to_RGB(YcCbcCrc,
                    in_bits=10,
                    in_legal=True,
                    in_int=False,
                    is_12_bits_system=False,
                    **kwargs):
    """
    Converts an array of *Yc'Cbc'Crc'* colour encoding values to the
    corresponding *RGB* array of linear values.

    Parameters
    ----------
    YcCbcCrc : array_like
        Input *Yc'Cbc'Crc'* colour encoding array of linear float values.
    in_bits : int, optional
        Bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Default is *10*.
    in_legal : bool, optional
        Whether to treat the input values as legal range. Default is *False*.
    in_int : bool, optional
        Whether to treat the input values as ``in_bits`` integer code values.
        Default is *False*.
    is_12_bits_system : bool, optional
        *Recommendation ITU-R BT.2020* EOTF (EOCF) adopts different parameters
        for 10 and 12 bit systems. Default is *False*.

    Other Parameters
    ----------------
    in_range : array_like, optional
        Array overriding the computed range such as
        *in_range = (Y_min, Y_max, C_min, C_max)*. If ``in_range`` is
        undefined, *Y_min*, *Y_max*, *C_min* and *C_max* will be computed using
        :func:`colour.models.rgb.ycbcr.YCbCr_ranges` definition.

    Returns
    -------
    ndarray
        *RGB* array of linear float values.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``YcCbcCrc``   | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``RGB``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    \\* This definition has input and output integer switches, thus the
    domain-range scale information is only given for the floating point mode.

    Warning
    -------
    This definition is specifically for usage with
    *Recommendation ITU-R BT.2020* when adopting the constant luminance
    implementation.

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2015h`,
    :cite:`Wikipedia2004d`

    Examples
    --------
    >>> YcCbcCrc = np.array([1689, 2048, 2048])
    >>> YcCbcCrc_to_RGB(YcCbcCrc, in_legal=True, in_bits=12, in_int=True,
    ...                 is_12_bits_system=True)
    ... # doctest: +ELLIPSIS
    array([ 0.1800903...,  0.1800903...,  0.1800903...])
    """

    if in_int:
        YcCbcCrc = as_float_array(YcCbcCrc)
    else:
        YcCbcCrc = to_domain_1(YcCbcCrc)

    Yc, Cbc, Crc = tsplit(YcCbcCrc.astype(DEFAULT_FLOAT_DTYPE))
    Y_min, Y_max, C_min, C_max = kwargs.get(
        'in_range', YCbCr_ranges(in_bits, in_legal, in_int))

    Yc -= Y_min
    Cbc -= (C_max + C_min) / 2
    Crc -= (C_max + C_min) / 2
    Yc *= 1 / (Y_max - Y_min)
    Cbc *= 1 / (C_max - C_min)
    Crc *= 1 / (C_max - C_min)
    B = np.where(Cbc <= 0, Cbc * 1.9404 + Yc, Cbc * 1.5816 + Yc)
    R = np.where(Crc <= 0, Crc * 1.7184 + Yc, Crc * 0.9936 + Yc)

    with domain_range_scale('ignore'):
        Yc = eotf_BT2020(Yc, is_12_bits_system=is_12_bits_system)
        B = eotf_BT2020(B, is_12_bits_system=is_12_bits_system)
        R = eotf_BT2020(R, is_12_bits_system=is_12_bits_system)

    G = (Yc - 0.0593 * B - 0.2627 * R) / 0.6780

    RGB = tstack([R, G, B])

    return from_range_1(RGB)
