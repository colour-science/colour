#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Y'CbCr Colour Encoding
======================

Defines the Y'CbCr encoding (Y'CbCr is not an absolute colourspace)
transformations:

-   :func:`RGB_to_YCbCr`
-   :func:`YCbCr_to_RGB`
-   :func:`RGB_to_YcCbcCrc`
-   :func:`YcCbcCrc_to_RGB`

References
----------
.. [1]  Wikipedia. YCbCr. Retrieved February 29, 2016, from
        https://en.wikipedia.org/wiki/YCbCr
.. [2]  Recommendation ITU-R BT.709-6(06/2015). Retrieved February 29, 2016,
        from https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
        R-REC-BT.709-6-201506-I!!PDF-E.pdf
.. [3]  Recommendation ITU-R BT.2020(08/2012). Retrieved February 29, 2016,
        from https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
        R-REC-BT.2020-0-201208-S!!PDF-E.pdf
.. [4]  Ford, A., & Roberts, A. (1998). Colour Space Converions. Retrieved
        April 29, 2016, from http://www.poynton.com/PDFs/coloureq.pdf
"""

import numpy as np

from colour.utilities import CaseInsensitiveMapping, tsplit, tstack
from colour.models.rgb import REC_2020_COLOURSPACE

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['YCBCR_WEIGHTS',
           'RGB_RANGE',
           'YCBCR_RANGES',
           'RGB_to_YCbCr',
           'YCbCr_to_RGB',
           'RGB_to_YcCbcCrc',
           'YcCbcCrc_to_RGB']

YCBCR_WEIGHTS = CaseInsensitiveMapping(
        {'Rec. 601': (0.299, 0.114),
         'Rec. 709': (0.2126, 0.0722),
         'Rec. 2020': (0.2627, 0.0593),
         'SMPTE-240M': (0.2122, 0.0865)})
"""
List of preset luma weightings:
    'Rec. 601' : (0.299, 0.114)
    'Rec. 709' : (0.2126, 0.0722)
    'Rec. 2020' : (0.2627, 0.0593)
    'SMPTE-240M': (0.2122, 0.0865)
"""


def RGB_RANGE(bits, is_legal, is_int):
    if is_legal:
        range = np.array([16, 235])
        range *= 2**(bits - 8)
    else:
        range = np.array([0, 2**bits - 1])
    if not(is_int):
        range = range.astype(float)/(2**bits - 1)
    return range


def YCBCR_RANGES(bits, is_legal, is_int):
    if is_legal:
        ranges = np.array([16, 235, 16, 240])
        ranges *= 2**(bits - 8)
    else:
        ranges = np.array([0, 2**bits - 1, 0, 2**bits - 1])
    if not(is_int):
        ranges = ranges.astype(float)/(2**bits - 1)
    if (is_int and not(is_legal)):
        ranges[3] = 2**bits
    if (not(is_int) and not(is_legal)):
        ranges[2] = -0.5
        ranges[3] = 0.5
    return ranges


def RGB_to_YCbCr(rgb,
                 K=YCBCR_WEIGHTS['Rec. 709'],
                 in_range=None,
                 out_range=None,
                 in_bits=10,
                 in_legal=False,
                 in_int=False,
                 out_bits=8,
                 out_legal=True,
                 out_int=False):
    """
    Converts an array of R'G'B' values to the corresponding Y'CbCr values.

    Parameters
    ----------
    rgb : array_like
        Input R'G'B' values. These may be floats or integers.
    K : tuple (optional)
        The luma weighting coefficients of red and blue. Use the function
        YCBCR_WEIGHTS(colourspace) for presets
        **{'Rec. 601', 'Rec. 709', 'Rec. 2020','SMPTE-240M'}**
        Default: YCBCR_WEIGHTS['Rec. 709']
    in_range : array_like (optional)
        The values of R', G' and B' corresponding to 0 IRE and 100 IRE.
        These may be floats or integer code values. If values are given here
        they take precedence over values calculated from in_legal, in_float
        and in_bits.
    out_range : array_like (optional)
        The values of Y' corresponding to 0 IRE and 100 IRE
        and the minimum and maximum values of Cb and Cr.
        These may be floats or integer code values. If values are given here
        they take precedence over values calculated from out_legal, out_float
        and out_bits.
    in_bits : int (optional)
        The bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8 bit means the float
        value for legal white is 235/255.
        Default : 10
    in_legal : bool (optional)
        Whether to treat the input values as legal range.
        Default : False
    in_int : bool (optional)
        Whether to treat the input values as in_bits integer code values.
        Default : False
    out_bits : int (optional)
        The bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8 bit means the float
        value for legal white is 235/255. Ignored if out_legal and out_int are
        both False.
        Default : 8
    out_legal : bool (optional)
        Whether to return legal range data.
        Default : True
    out_int : bool (optional)
        Whether to return values as out_bits integer code values.
        Default : False

    Returns
    -------
    ndarray
        Y'CbCr array.
        These may be floats or integer code values, depending on parameters.

    Notes
    -----
    -   For ITU-R BT.2020 (Rec.2020) the RGB_to_YCbCr function is only
        applicable tothe non-constant luminance implementation. The
        RGB_to_YcCbcCrc function should be used for the constant luminance case
        See https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
        R-REC-BT.2020-0-201208-S!!PDF-E.pdf

    -   The default settings of (in_bits=10, in_legal=False, in_int=False,
        out_bits=8, out_legal=True, out_int=False) will transform float RGB
        input ranged 0.0-1.0 (in_bits is ignored) to float Y'CbCr output where
        Y' is ranged 16./255 to 235./255 and Cb and Cr are ranged 16./255 to
        240./255. So the float values are calculated based on an integer range
        from 0-255. But no 8-bit quantisation is applied, and neither is any
        clamping.

    -   To match the float output of Nuke's Colorspace node set to YCbCr,
        set out_range=(16./255, 235./255, 15.5/255, 239.5/255)

    -   To match the float output of Nuke's Colorspace node set to YPbPr,
        set (out_legal=False, out_int=False)

    -   To create integer code values as per standard 10-bit SDI, set
        (out_bits=8, out_legal=True, out_int=True)

    Examples
    --------
    >>> rgb = np.array([1.0, 1.0, 1.0])
    >>> RGB_to_YCbCr(  # doctest: +ELLIPSIS
    ...              rgb)
    array([ 0.9215686...,  0.5019607...,  0.5019607...])
    >>> RGB_to_YCbCr(rgb, out_legal=True, out_bits=10, out_int=True)
    array([940, 512, 512])

    For JFIF JPEG conversion as per http://www.w3.org/Graphics/JPEG/jfif3.pdf:

    >>> rgb = np.array([102, 0, 51])
    >>> RGB_to_YCbCr(rgb,
    ...              K=YCBCR_WEIGHTS['Rec. 601'],
    ...              in_range=(0, 255),
    ...              out_range=(0, 255, 0, 256),
    ...              out_int=True)
    array([ 36, 136, 175])

    Note the use of 256 for the max Cb/Cr value, which is required so that the
    Cb and Cr output is centred about 128. Using 255 would centre them about
    127.5 meaning that there was no integer code value to represent neutral
    colours. This does however create the possibility of output values of 256,
    which cannot be stored in an 8-bit integer. The JPEG document specifies
    these should be clamped to 255.
    """
    Kr = K[0]
    Kb = K[1]
    if in_range is None:
        in_range = RGB_RANGE(in_bits, in_legal, in_int)
    if out_range is None:
        out_range = YCBCR_RANGES(out_bits, out_legal, out_int)
    rgb_float = (rgb.astype(float) - in_range[0])
    rgb_float *= 1.0/(in_range[1] - in_range[0])
    r, g, b = tsplit(rgb_float)
    Y = Kr*r + (1.0 - Kr - Kb)*g + Kb*b
    Cb = 0.5*(b - Y)/(1.0 - Kb)
    Cr = 0.5*(r - Y)/(1.0 - Kr)
    Y *= out_range[1] - out_range[0]
    Y += out_range[0]
    Cb *= out_range[3] - out_range[2]
    Cr *= out_range[3] - out_range[2]
    Cb += (out_range[3] + out_range[2])/2
    Cr += (out_range[3] + out_range[2])/2
    YCbCr = tstack((Y, Cb, Cr))
    if out_int:
        YCbCr = np.round(YCbCr).astype(int)
    return YCbCr


def YCbCr_to_RGB(YCbCr,
                 K=YCBCR_WEIGHTS['Rec. 709'],
                 in_range=None,
                 out_range=None,
                 in_bits=8,
                 in_legal=True,
                 in_int=False,
                 out_bits=10,
                 out_legal=False,
                 out_int=False):
    """
    Converts an array of Y'CbCr values to the corresponding R'G'B' values.

    Parameters
    ----------
    YCbCr : array_like
        Input Y'CbCr values. These may be floats or integers.
    K : tuple (optional)
        The luma weighting coefficients of red and blue. Use the function
        YCBCR_WEIGHTS(colourspace) for presets
        **{'Rec. 601', 'Rec. 709', 'Rec. 2020','SMPTE-240M'}**
        Default: YCBCR_WEIGHTS['Rec. 709']
    in_range : array_like (optional)
        The values of Y' corresponding to 0 IRE and 100 IRE
        and the minimum and maximum values of Cb and Cr.
        These may be floats or integer code values. If values are given here
        they take precedence over values calculated from in_legal, in_float
        and in_bits.
    out_range : array_like (optional)
        The values of R', G' and B' corresponding to 0 IRE and 100 IRE.
        These may be floats or integer code values. If values are given here
        they take precedence over values calculated from out_legal, out_float
        and out_bits.
    in_bits : int (optional)
        The bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8 bit means the float
        value for legal white is 235/255.
        Default : 8
    in_legal : bool (optional)
        Whether to treat the input values as legal range.
        Default : True
    in_int : bool (optional)
        Whether to treat the input values as in_bits integer code values.
        Default : False
    out_bits : int (optional)
        The bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8 bit means the float
        value for legal white is 235/255. Ignored if out_legal and out_int are
        both False.
        Default : 10
    out_legal : bool (optional)
        Whether to return legal range data.
        Default : False
    out_int : bool (optional)
        Whether to return values as out_bits integer code values.
        Default : False

    Returns
    -------
    ndarray
        R'G'B' array.
        These may be floats or integer code values, depending on parameters.

    Notes
    -----
    -   For ITU-R BT.2020 (Rec.2020) the YCbCr_to_RGB function is only
        applicable tothe non-constant luminance implementation. The
        YcCbcCrc_to_RGB function should be used for the constant luminance case
        See https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
        R-REC-BT.2020-0-201208-S!!PDF-E.pdf

    Examples
    --------
    >>> YCbCr = np.array([502, 512, 512])
    >>> YCbCr_to_RGB(YCbCr,
    ...              in_bits=10,
    ...              in_legal=True,
    ...              in_int=True)
    array([ 0.5,  0.5,  0.5])
    """
    Kr = K[0]
    Kb = K[1]
    if out_range is None:
        out_range = RGB_RANGE(out_bits, out_legal, out_int)
    if in_range is None:
        in_range = YCBCR_RANGES(in_bits, in_legal, in_int)
    Y, Cb, Cr = tsplit(YCbCr.astype(float))
    Y -= in_range[0]
    Cb -= (in_range[3] + in_range[2])/2.0
    Cr -= (in_range[3] + in_range[2])/2.0
    Y *= 1.0/(in_range[1] - in_range[0])
    Cb *= 1.0/(in_range[3] - in_range[2])
    Cr *= 1.0/(in_range[3] - in_range[2])
    r = Y + (2.0 - 2.0*Kr)*Cr
    b = Y + (2.0 - 2.0*Kb)*Cb
    g = (Y - Kr*r - Kb*b)/(1.0 - Kr - Kb)
    rgb = tstack((r, g, b))
    rgb *= out_range[1] - out_range[0]
    rgb += out_range[0]
    if out_int:
        rgb = np.round(rgb).astype(int)
    return rgb


def RGB_to_YcCbcCrc(rgb,
                    out_range=None,
                    out_bits=8,
                    out_legal=True,
                    out_int=False,
                    is_10_bits_system=True):
    """
    Converts an array of RGB (linear) values to the corresponding Yc'Cbc'Crc'
    values.

    Parameters
    ----------
    rgb : array_like
        Input RGB floating point linear values.
    out_range : array_like (optional)
        The values of Y' corresponding to 0 IRE and 100 IRE
        and the minimum and maximum values of Cb and Cr.
        These may be floats or integer code values. If values are given here
        they take precedence over values calculated from out_legal, out_float
        and out_bits.
    out_bits : int (optional)
        The bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8 bit means the float
        value for legal white is 235/255. Ignored if out_legal and out_int are
        both False.
        Default : 8
    out_legal : bool (optional)
        Whether to return legal range data.
        Default : True
    out_int : bool (optional)
        Whether to return values as out_bits integer code values.
        Default : False
    is_10_bits_system : bool (optional)
        The Rec.2020 OECF varies slightly for 10 and 12 bit systems.
        Default: True

    Returns
    -------
    ndarray
        Yc'Cbc'Crc' array.
        These may be floats or integer code values, depending on parameters.

    Notes
    -----
    -   This fuction is specifically for use with ITU-R BT.2020 (Rec.2020)
        when using the constant luminance implementation.
        See https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
        R-REC-BT.2020-0-201208-S!!PDF-E.pdf

    Examples
    --------
    >>> rgb = np.array([0.18, 0.18, 0.18])
    >>> RGB_to_YcCbcCrc(rgb,
    ...                 out_legal=True,
    ...                 out_bits=10,
    ...                 out_int=True,
    ...                 is_10_bits_system = True)
    array([422, 512, 512])
    """
    if out_range is None:
        out_range = YCBCR_RANGES(out_bits, out_legal, out_int)
    r, g, b = tsplit(rgb)
    Yc = 0.2627*r + 0.6780*g + 0.0593*b
    Yc = REC_2020_COLOURSPACE.OECF(Yc, is_10_bits_system=is_10_bits_system)
    r = REC_2020_COLOURSPACE.OECF(r, is_10_bits_system=is_10_bits_system)
    b = REC_2020_COLOURSPACE.OECF(b, is_10_bits_system=is_10_bits_system)
    Cbc = np.where((b - Yc) <= 0.0, (b - Yc)/1.9404, (b - Yc)/1.5816)
    Crc = np.where((r - Yc) <= 0.0, (r - Yc)/1.7184, (r - Yc)/0.9936)
    Yc *= out_range[1] - out_range[0]
    Yc += out_range[0]
    Cbc *= out_range[3] - out_range[2]
    Crc *= out_range[3] - out_range[2]
    Cbc += (out_range[3] + out_range[2])/2
    Crc += (out_range[3] + out_range[2])/2
    YcCbcCrc = tstack((Yc, Cbc, Crc))
    if out_int:
        YcCbcCrc = np.round(YcCbcCrc).astype(int)
    return YcCbcCrc


def YcCbcCrc_to_RGB(YcCbcCrc,
                    in_range=None,
                    in_bits=8,
                    in_legal=True,
                    in_int=False,
                    is_10_bits_system=True):
    """
    Converts an array of Yc'Cbc'Crc' values to the corresponding RGB (linear)
    values.

    Parameters
    ----------
    YcCbcCrc : array_like
        Input Yc'Cbc'Crc' values. These may be floats or integers.
    in_range : array_like (optional)
        The values of Y' corresponding to 0 IRE and 100 IRE
        and the minimum and maximum values of Cb and Cr.
        These may be floats or integer code values. If values are given here
        they take precedence over values calculated from in_legal, in_float
        and in_bits.
    out_bits : int (optional)
        The bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8 bit means the float
        value for legal white is 235/255. Ignored if out_legal and out_int are
        both False.
        Default : 10
    out_legal : bool (optional)
        Whether to return legal range data.
        Default : False
    out_int : bool (optional)
        Whether to return values as out_bits integer code values.
        Default : False
    is_10_bits_system : bool (optional)
        The Rec.2020 EOCF varies slightly for 10 and 12 bit systems.
        Default: True


    Returns
    -------
    ndarray
        RGB values corresponding to the input colour.
        These will be floating point linear values.

    Notes
    -----
    -   This fuction is specifically for use with ITU-R BT.2020 (Rec.2020)
        when using the constant luminance implementation.
        See https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
        R-REC-BT.2020-0-201208-S!!PDF-E.pdf

    Examples
    --------
    >>> YcCbcCrc = np.array([1689, 2048, 2048])
    >>> YcCbcCrc_to_RGB(  # doctest: +ELLIPSIS
    ...                 YcCbcCrc,
    ...                 in_legal=True,
    ...                 in_bits=12,
    ...                 in_int=True,
    ...                 is_10_bits_system=False)
    array([ 0.1800903..., 0.1800903..., 0.1800903...])
    """
    if in_range is None:
        in_range = YCBCR_RANGES(in_bits, in_legal, in_int)
    Yc, Cbc, Crc = tsplit(YcCbcCrc.astype(float))
    Yc -= in_range[0]
    Cbc -= (in_range[3] + in_range[2])/2.0
    Crc -= (in_range[3] + in_range[2])/2.0
    Yc *= 1.0/(in_range[1] - in_range[0])
    Cbc *= 1.0/(in_range[3] - in_range[2])
    Crc *= 1.0/(in_range[3] - in_range[2])
    b = np.where(Cbc <= 0.0, Cbc*1.9404 + Yc, Cbc*1.5816 + Yc)
    r = np.where(Crc <= 0.0, Crc*1.7184 + Yc, Crc*0.9936 + Yc)
    Yc = REC_2020_COLOURSPACE.EOCF(Yc, is_10_bits_system=is_10_bits_system)
    b = REC_2020_COLOURSPACE.EOCF(b, is_10_bits_system=is_10_bits_system)
    r = REC_2020_COLOURSPACE.EOCF(r, is_10_bits_system=is_10_bits_system)
    g = (Yc - 0.0593*b - 0.2627*r)/0.6780
    rgb = tstack((r, g, b))
    return rgb
