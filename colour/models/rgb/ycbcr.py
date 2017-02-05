#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Y'CbCr Colour Encoding
======================

Defines the *Y'CbCr* colour encoding related transformations:

-   :func:`RGB_to_YCbCr`
-   :func:`YCbCr_to_RGB`
-   :func:`RGB_to_YcCbcCrc`
-   :func:`YcCbcCrc_to_RGB`

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
.. [1]  Wikipedia. (n.d.). YCbCr. Retrieved February 29, 2016, from
        https://en.wikipedia.org/wiki/YCbCr
.. [2]  International Telecommunication Union. (2015). Recommendation
        ITU-R BT.709-6 - Parameter values for the HDTV standards for
        production and international programme exchange BT Series Broadcasting
        service (Vol. 5). Retrieved from https://www.itu.int/dms_pubrec/\
itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf
.. [3]  International Telecommunication Union. (2015). Recommendation
        ITU-R BT.2020 - Parameter values for ultra-high definition television
        systems for production and international programme exchange (Vol. 1).
        Retrieved from https://www.itu.int/dms_pubrec/\
itu-r/rec/bt/R-REC-BT.2020-2-201510-I!!PDF-E.pdf
.. [4]  Society of Motion Picture and Television Engineers. (1999).
        ANSI/SMPTE 240M-1995 - Signal Parameters - 1125-Line High-Definition
        Production Systems, 1–7. Retrieved from
        http://car.france3.mars.free.fr/\
HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/s240m.pdf
.. [5]  International Telecommunication Union. (2011). Recommendation ITU-T
        T.871 - Information technology – Digital compression and coding of
        continuous-tone still images: JPEG File Interchange Format (JFIF).
        Retrieved from https://www.itu.int/rec/dologin_pub.asp?lang=e&\
id=T-REC-T.871-201105-I!!PDF-E&type=items
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping, tsplit, tstack
from colour.models.rgb.transfer_functions import oetf_BT2020, eotf_BT2020

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['YCBCR_WEIGHTS',
           'RGB_range',
           'YCbCr_ranges',
           'RGB_to_YCbCr',
           'YCbCr_to_RGB',
           'RGB_to_YcCbcCrc',
           'YcCbcCrc_to_RGB']

YCBCR_WEIGHTS = CaseInsensitiveMapping(
    {'Rec. 601': np.array([0.2990, 0.1140]),
     'Rec. 709': np.array([0.2126, 0.0722]),
     'Rec. 2020': np.array([0.2627, 0.0593]),
     'SMPTE-240M': np.array([0.2122, 0.0865])})
"""
Luma weightings presets.

YCBCR_WEIGHTS : dict
    **{'Rec. 601', 'Rec. 709', 'Rec. 2020', 'SMPTE-240M}**
"""


def RGB_range(bits, is_legal, is_int):
    """"
    Returns the *RGB* range array for given bit depth, range legality and
    representation.

    Parameters
    ----------
    bits : int
        Bit depth of the *RGB* output ranges array.
    is_legal : bool
        Whether the *RGB* output ranges array is legal.
    is_int : bool
        Whether the *RGB* output ranges array represents integer code values.

    Returns
    -------
    ndarray
        *RGB* ranges array.

    Examples
    --------
    >>> RGB_range(8, True, True)
    array([ 16, 235])
    >>> RGB_range(8, True, False)  # doctest: +ELLIPSIS
    array([ 0.0627451...,  0.9215686...])
    >>> RGB_range(10, False, False)
    array([ 0.,  1.])
    """

    if is_legal:
        ranges = np.array([16, 235])
        ranges *= 2 ** (bits - 8)
    else:
        ranges = np.array([0, 2 ** bits - 1])

    if not is_int:
        ranges = ranges.astype(np.float_) / (2 ** bits - 1)

    return ranges


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
        ranges = ranges.astype(np.float_) / (2 ** bits - 1)

    if is_int and not is_legal:
        ranges[3] = 2 ** bits

    if not is_int and not is_legal:
        ranges[2] = -0.5
        ranges[3] = 0.5

    return ranges


def RGB_to_YCbCr(RGB,
                 K=YCBCR_WEIGHTS['Rec. 709'],
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
        Luma weighting coefficients of red and blue. See :attr:
        `YCBCR_WEIGHTS` for presets. Default is `(0.2126, 0.0722)`, the
        weightings for Rec. 709.
    in_bits : int, optional
        Bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is `235 / 255`. Default is `10`.
    in_legal : bool, optional
        Whether to treat the input values as legal range. Default is `False`.
    in_int : bool, optional
        Whether to treat the input values as `in_bits` integer code values.
        Default is `False`.
    out_bits : int, optional
        Bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is `235 / 255`. Ignored if `out_legal` and
        `out_int` are both False. Default is `8`.
    out_legal : bool, optional
        Whether to return legal range values. Default is `True`.
    out_int : bool, optional
        Whether to return values as `out_bits` integer code values. Default is
        `False`.

    Other Parameters
    ----------------
    in_range : array_like, optional
        Array overriding the computed range such as
        `in_range = (RGB_min, RGB_max)`. If `in_range` is undefined, `RGB_min`
        and `RGB_max` will be computed using :func:`RGB_range` definition.
    out_range : array_like, optional
        Array overriding the computed range such as
        `out_range = (Y_min, Y_max, C_min, C_max)`. If `out_range` is
        undefined, `Y_min`, `Y_max`, `C_min` and `C_max` will be computed
        using :func:`YCbCr_ranges` definition.

    Returns
    -------
    ndarray
        *Y'CbCr* colour encoding array of integer or float values.

    Warning
    -------
    For *Recommendation ITU-R BT.2020*, :func:`RGB_to_YCbCr` definition is only
    applicable to the non-constant luminance implementation.
    :func:`RGB_to_YcCbcCrc` definition should be used for the constant
    luminance case as per [3]_.

    Notes
    -----
    -   The default arguments, ``**{'in_bits': 10, 'in_legal': False,
        'in_int': False, 'out_bits': 8, 'out_legal': True, 'out_int': False}``
        transform a float *R'G'B'* input array in range [0, 1] (`in_bits` is
        ignored) to a float *Y'CbCr* output array where *Y'* is in range
        [16 / 255, 235 / 255] and *Cb* and *Cr* are in range
        [16 / 255, 240./255]. The float values are calculated based on an
        [0, 255] integer range, but no 8-bit quantisation or clamping are
        performed.

    Examples
    --------
    >>> RGB = np.array([1.0, 1.0, 1.0])
    >>> RGB_to_YCbCr(RGB)  # doctest: +ELLIPSIS
    array([ 0.9215686...,  0.5019607...,  0.5019607...])

    Matching float output of The Foundry Nuke's Colorspace node set to YCbCr:

    >>> RGB_to_YCbCr(  # doctest: +ELLIPSIS
    ...     RGB,
    ...     out_range=(16 / 255, 235 / 255, 15.5 / 255, 239.5 / 255))
    array([ 0.9215686...,  0.5       ,  0.5       ])

    Matching float output of The Foundry Nuke's Colorspace node set to YPbPr:

    >>> RGB_to_YCbCr(  # doctest: +ELLIPSIS
    ...     RGB,
    ...     out_legal=False,
    ...     out_int=False)
    array([ 1.,  0.,  0.])

    Creating integer code values as per standard 10-bit SDI:

    >>> RGB_to_YCbCr(RGB, out_legal=True, out_bits=10, out_int=True)
    array([940, 512, 512])

    For JFIF JPEG conversion as per ITU-T T.871 [5]_:

    >>> RGB = np.array([102, 0, 51])
    >>> RGB_to_YCbCr(
    ...     RGB,
    ...     K=YCBCR_WEIGHTS['Rec. 601'],
    ...     in_range=(0, 255),
    ...     out_range=(0, 255, 0, 256),
    ...     out_int=True)
    array([ 36, 136, 175])

    Note the use of 256 for the max *Cb / Cr* value, which is required so that
    the *Cb* and *Cr* output is centered about 128. Using 255 centres it
    about 127.5, meaning that there is no integer code value to represent
    achromatic colours. This does however create the possibility of output
    integer codes with value of 256, which cannot be stored in 8-bit integer
    representation. Recommendation ITU-T T.871 specifies these should be
    clamped to 255.

    These JFIF JPEG ranges are also obtained as follows:

    >>> RGB_to_YCbCr(
    ...     RGB,
    ...     K=YCBCR_WEIGHTS['Rec. 601'],
    ...     in_bits=8,
    ...     in_int=True,
    ...     out_legal=False,
    ...     out_int=True)
    array([ 36, 136, 175])
    """

    RGB = np.asarray(RGB)
    Kr, Kb = K
    RGB_min, RGB_max = kwargs.get(
        'in_range', RGB_range(in_bits, in_legal, in_int))
    Y_min, Y_max, C_min, C_max = kwargs.get(
        'out_range', YCbCr_ranges(out_bits, out_legal, out_int))

    RGB_float = RGB.astype(np.float_) - RGB_min
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

    YCbCr = tstack((Y, Cb, Cr))
    YCbCr = np.round(YCbCr).astype(np.int_) if out_int else YCbCr

    return YCbCr


def YCbCr_to_RGB(YCbCr,
                 K=YCBCR_WEIGHTS['Rec. 709'],
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
        Luma weighting coefficients of red and blue. See :attr:
        `YCBCR_WEIGHTS` for presets. Default is `(0.2126, 0.0722)`, the
        weightings for Rec. 709.
    in_bits : int, optional
        Bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is `235 / 255`. Default is `10`.
    in_legal : bool, optional
        Whether to treat the input values as legal range. Default is `False`.
    in_int : bool, optional
        Whether to treat the input values as `in_bits` integer code values.
        Default is `False`.
    out_bits : int, optional
        Bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is `235 / 255`. Ignored if `out_legal` and
        `out_int` are both False. Default is `8`.
    out_legal : bool, optional
        Whether to return legal range values. Default is `True`.
    out_int : bool, optional
        Whether to return values as `out_bits` integer code values. Default is
        `False`.

    Other Parameters
    ----------------
    in_range : array_like, optional
        Array overriding the computed range such as
        `in_range = (Y_min, Y_max, C_min, C_max)`. If `in_range` is undefined,
        `Y_min`, `Y_max`, `C_min` and `C_max` will be computed using
        :func:`YCbCr_ranges` definition.
    out_range : array_like, optional
        Array overriding the computed range such as
        `out_range = (RGB_min, RGB_max)`. If `out_range` is undefined,
        `RGB_min` and `RGB_max` will be computed using :func:`RGB_range`
        definition.

    Returns
    -------
    ndarray
        *R'G'B'* array of integer or float values.

    Warning
    -------
    For *Recommendation ITU-R BT.2020*, :func:`YCbCr_to_RGB`
    definition is only applicable to the non-constant luminance implementation.
    :func:`YcCbcCrc_to_RGB` definition should be used for the constant
    luminance case as per [3]_.

    Examples
    --------
    >>> YCbCr = np.array([502, 512, 512])
    >>> YCbCr_to_RGB(
    ...     YCbCr,
    ...     in_bits=10,
    ...     in_legal=True,
    ...     in_int=True)
    array([ 0.5,  0.5,  0.5])
    """

    YCbCr = np.asarray(YCbCr)
    Y, Cb, Cr = tsplit(YCbCr.astype(np.float_))
    Kr, Kb = K
    Y_min, Y_max, C_min, C_max = kwargs.get(
        'in_range', YCbCr_ranges(in_bits, in_legal, in_int))
    RGB_min, RGB_max = kwargs.get(
        'out_range', RGB_range(out_bits, out_legal, out_int))

    Y -= Y_min
    Cb -= (C_max + C_min) / 2
    Cr -= (C_max + C_min) / 2
    Y *= 1 / (Y_max - Y_min)
    Cb *= 1 / (C_max - C_min)
    Cr *= 1 / (C_max - C_min)
    R = Y + (2 - 2 * Kr) * Cr
    B = Y + (2 - 2 * Kb) * Cb
    G = (Y - Kr * R - Kb * B) / (1 - Kr - Kb)

    RGB = tstack((R, G, B))
    RGB *= RGB_max - RGB_min
    RGB += RGB_min
    RGB = np.round(RGB).astype(np.int_) if out_int else RGB

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
        value for legal white is `235 / 255`. Ignored if `out_legal` and
        `out_int` are both False. Default is `10`.
    out_legal : bool, optional
        Whether to return legal range values. Default is `True`.
    out_int : bool, optional
        Whether to return values as `out_bits` integer code values. Default is
        `False`.
    is_12_bits_system : bool, optional
        *Recommendation ITU-R BT.2020* OETF (OECF) adopts different parameters
        for 10 and 12 bit systems. Default is `False`.

    Other Parameters
    ----------------
    out_range : array_like, optional
        Array overriding the computed range such as
        `out_range = (Y_min, Y_max, C_min, C_max)`. If `out_range` is
        undefined, `Y_min`, `Y_max`, `C_min` and `C_max` will be computed
        using :func:`YCbCr_ranges` definition.

    Returns
    -------
    ndarray
        *Yc'Cbc'Crc'* colour encoding array of integer or float values.

    Warning
    -------
    This definition is specifically for usage with
    *Recommendation ITU-R BT.2020* [3]_ when adopting the constant luminance
    implementation.

    Examples
    --------
    >>> RGB = np.array([0.18, 0.18, 0.18])
    >>> RGB_to_YcCbcCrc(
    ...     RGB,
    ...     out_legal=True,
    ...     out_bits=10,
    ...     out_int=True,
    ...     is_12_bits_system=False)
    array([422, 512, 512])
    """

    RGB = np.asarray(RGB)
    R, G, B = tsplit(RGB)
    Y_min, Y_max, C_min, C_max = kwargs.get(
        'out_range', YCbCr_ranges(out_bits, out_legal, out_int))

    Yc = 0.2627 * R + 0.6780 * G + 0.0593 * B
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

    YcCbcCrc = tstack((Yc, Cbc, Crc))
    YcCbcCrc = np.round(YcCbcCrc).astype(np.int_) if out_int else YcCbcCrc

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
        value for legal white is `235 / 255`. Default is `10`.
    in_legal : bool, optional
        Whether to treat the input values as legal range. Default is `False`.
    in_int : bool, optional
        Whether to treat the input values as `in_bits` integer code values.
        Default is `False`.
    is_12_bits_system : bool, optional
        *Recommendation ITU-R BT.2020* EOTF (EOCF) adopts different parameters
        for 10 and 12 bit systems. Default is `False`.

    Other Parameters
    ----------------
    in_range : array_like, optional
        Array overriding the computed range such as
        `in_range = (Y_min, Y_max, C_min, C_max)`. If `in_range` is undefined,
        `Y_min`, `Y_max`, `C_min` and `C_max` will be computed using
        :func:`YCbCr_ranges` definition.

    Returns
    -------
    ndarray
        *RGB* array of linear float values.

    Warning
    -------
    This definition is specifically for usage with
    *Recommendation ITU-R BT.2020* [3]_ when adopting the constant luminance
    implementation.

    Examples
    --------
    >>> YcCbcCrc = np.array([1689, 2048, 2048])
    >>> YcCbcCrc_to_RGB(  # doctest: +ELLIPSIS
    ...     YcCbcCrc,
    ...     in_legal=True,
    ...     in_bits=12,
    ...     in_int=True,
    ...     is_12_bits_system=True)
    array([ 0.1800903...,  0.1800903...,  0.1800903...])
    """

    YcCbcCrc = np.asarray(YcCbcCrc)
    Yc, Cbc, Crc = tsplit(YcCbcCrc.astype(np.float_))
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
    Yc = eotf_BT2020(Yc, is_12_bits_system=is_12_bits_system)
    B = eotf_BT2020(B, is_12_bits_system=is_12_bits_system)
    R = eotf_BT2020(R, is_12_bits_system=is_12_bits_system)
    G = (Yc - 0.0593 * B - 0.2627 * R) / 0.6780

    RGB = tstack((R, G, B))

    return RGB
