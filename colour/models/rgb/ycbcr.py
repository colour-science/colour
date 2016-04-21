#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Y'CbCr Colour Encoding
======================

Defines the Y'CbCr colour encoding (Y'CbCr is not a colour space) transformations:

-   :func:`rgb_to_YCbCr`
-   :func:`YCbCr_to_rgb`
-   :func:`rgb_to_YcCbcCrc`
-   :func:`YcCbcCrc_to_rgb`

References
----------
.. [1]  Wikipedia. YCbCr. Retrieved February 29, 2016, from
        https://en.wikipedia.org/wiki/YCbCr
.. [2]  Recommendation ITU-R BT.709-6(06/2015). Retrieved February 29, 2016, from
        https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf
.. [3]  Recommendation ITU-R BT.2020(08/2012). Retrieved February 29, 2016, from
        https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-S!!PDF-E.pdf
"""

import numpy as np

from colour.utilities import tsplit, tstack, CaseInsensitiveMapping
from colour.models.rgb import *

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['WEIGHT',
           'RANGE',
           'rgb_to_YCbCr',
           'YCbCr_to_rgb',
           'rgb_to_YcCbcCrc',
           'YcCbcCrc_to_rgb']

WEIGHT = (CaseInsensitiveMapping(
        {'Rec.601' : [0.299, 0.114],
         'Rec.709' : [0.2126, 0.0722],
         'Rec.2020' : [0.2627, 0.0593]}))
"""
List of preset luma weightings:
    'Rec.601' : [0.299, 0.114]
    'Rec.709' : [0.2126, 0.0722]
    'Rec.2020' : [0.2627, 0.0593]
"""

RANGE = (CaseInsensitiveMapping(
        {'legal_12' : [256./4095, 3760./4095],
         'legal_12_YC' : [256./4095, 3760./4095, 256./4095, 3840./4095],
         'legal_12_int' : [256, 3760],
         'legal_12_YC_int' : [256, 3760, 256, 3840],
         'legal_10' : [64./1023, 940./1023],
         'legal_10_YC' : [64./1023, 940./1023, 64./1023, 960./1023],
         'legal_10_int' : [64, 940],
         'legal_10_YC_int' : [64, 940, 64, 960],
         'legal_8' : [16./255, 235./255],
         'legal_8_YC' : [16./255, 235./255, 16./255, 240./255],
         'legal_8_int' : [16, 235],
         'legal_8_YC_int' : [16, 235, 16, 240],
         'full' : [0.0, 1.0],
         'YPbPr' : [0.0, 1.0, -0.5, 0.5]}))
"""
List of preset ranges:
    'legal_12' : [256./4095, 3760./4095]
    'legal_12_YC' : [256./4095, 3760./4095, 256./4095, 3840./4095]
    'legal_12_int' : [256, 3760]
    'legal_12_YC_int' : [256, 3760, 256, 3840]
    'legal_10' : [64./1023, 940./1023]
    'legal_10_YC' : [64./1023, 940./1023, 64./1023, 960./1023]
    'legal_10_int' : [64, 940]
    'legal_10_YC_int' : [64, 940, 64, 960]
    'legal_8' : [16./255, 235./255]
    'legal_8_YC' : [16./255, 235./255, 16./255, 240./255]
    'legal_8_int' : [16, 235]
    'legal_8_YC_int' : [16, 235, 16, 240]
    'full' : [0.0, 1.0]
    'YPbPr' : [0.0, 1.0, -0.5, 0.5]
"""

def rgb_to_YCbCr(rgb, K = WEIGHT['Rec.709'], inRange = RANGE['full'], outRange = RANGE['legal_10_YC']):
    """
    Converts an array of R'G'B' values to the corresponding Y'CbCr values.
    
    Note: For ITU-R BT.2020 (Rec.2020) the rgb_to_YCbCr function is only applicable to
    the non-constant luminance implementation.
    The rgb_to_YcCbcCrc function should be used for the constant luminance case.
    See https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-S!!PDF-E.pdf
    
    Parameters
    ----------
    rgb : array_like
        Input R'G'B' values. These may be floats or integers.
    K : array of two values
        The luma weighting coefficients of red and blue.
        Default: WEIGHT['Rec.709']
    inRange : array of two values (optional)
        The values of R', G' and B' corresponding to 0 IRE and 100 IRE.
        These may be floats or integer code values.
        Default: RANGE['full']
    outRange : array of four values (optional)
        The values of Y' corresponding to 0 IRE and 100 IRE
        and the minimum and maximum values of Cb and Cr.
        These may be floats or integer code values.
        Default: RANGE['legal_10_YC']
    
    Returns
    -------
    ndarray
        Y'CbCr array.
        These may be floats or integer code values, depending on the parameter values
        given for outRange.
    
    Examples
    --------
    >>> RGB = np.array([1.0, 1.0, 1.0])
    >>> rgb_to_YCbCr(RGB)
    array([ 0.91886608,  0.50048876,  0.50048876])
    >>> rgb_to_YCbCr(RGB, outRange=RANGE['legal_10_YC_int'])
    array([940, 512, 512])
    
    For the JFIF JPEG conversion shown on https://en.wikipedia.org/wiki/YCbCr:
    
    >>> RGB = np.array([102, 0, 51])
    >>> rgb_to_YCbCr(RGB, K=WEIGHT['Rec.601'], inRange=[0, 255], outRange=[0, 255, 0, 256])
    array([ 36, 136, 175])
    
    Note the use of 256 for the max Cb/Cr value, which is required so that the Cb and Cr
    output is centred about 128. Using 255 would centre them about 127.5 meaning that
    there was no integer code value to represent neutral colours. This does however create
    the possibility of output values of 256, which cannot be stored in an 8-bit integer.
    """
    Kr = K[0]
    Kb = K[1]
    intOut = isinstance(outRange[0], int) and isinstance(outRange[1], int) and isinstance(outRange[2], int) and isinstance(outRange[3], int)
    rgb_float = (rgb.astype(float) - inRange[0])
    rgb_float *= 1.0/(inRange[1] - inRange[0])
    r, g, b = tsplit(rgb_float)
    Y = Kr*r + (1.0 - Kr - Kb)*g + Kb*b
    Cb = 0.5*(b - Y)/(1.0 - Kb)
    Cr = 0.5*(r - Y)/(1.0 - Kr)
    Y *= outRange[1] - outRange[0]
    Y += outRange[0]
    Cb *= outRange[3] - outRange[2]
    Cr *= outRange[3] - outRange[2]
    Cb += (outRange[3] + outRange[2])/2
    Cr += (outRange[3] + outRange[2])/2
    if intOut:
        Y = np.round(Y).astype(int)
        Cb = np.round(Cb).astype(int)
        Cr = np.round(Cr).astype(int)
    YCbCr = tstack((Y, Cb, Cr))
    return YCbCr

def YCbCr_to_rgb(YCbCr, K = WEIGHT['Rec.709'], inRange = RANGE['legal_10_YC'], outRange = RANGE['full']):
    """
    Converts an array of Y'CbCr values to the corresponding R'G'B' values.
    
    Note: For ITU-R BT.2020 (Rec.2020) the YCbCr_to_rgb function is only applicable to
    the non-constant luminance implementation.
    The YcCbcCrc_to_rgb function should be used for the constant luminance case.
    See https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-S!!PDF-E.pdf
    
    Parameters
    ----------
    YCbCr : array_like
        Input Y'CbCr values. These may be floats or integers.
    K : array of two values (optional)
        The luma weighting coefficients of red and blue.
        Default: WEIGHT['Rec.709']
    inRange : array of four values (optional)
        The values of Y' corresponding to 0 IRE and 100 IRE,
        and the minimum and maximum values of Cb and Cr.
        These may be floats or integer code values.
        Default: RANGE['legal_10_YC']
    outRange : array of two values (optional)
        The values of R', G' and B' corresponding to 0 IRE and 100 IRE.
        These may be floats or integer code values.
        Default: RANGE['full']
    
    Returns
    -------
    ndarray
        R'G'B' array.
        These may be floats or integer code values, depending on the parameter values
        given for outRange.
    
    Examples
    --------
    >>> YCbCr = np.array([502, 512, 512])
    >>> YCbCr_to_rgb(YCbCr, inRange=RANGE['legal_10_YC_int'])
    array([ 0.5,  0.5,  0.5])
    """
    Kr = K[0]
    Kb = K[1]
    intOut = isinstance(outRange[0], int) and isinstance(outRange[1], int)
    Y, Cb, Cr = tsplit(YCbCr.astype(float))
    Y -= inRange[0]
    Cb -= (inRange[3] + inRange[2])/2.0
    Cr -= (inRange[3] + inRange[2])/2.0
    Y *= 1.0/(inRange[1] - inRange[0])
    Cb *= 1.0/(inRange[3] - inRange[2])
    Cr *= 1.0/(inRange[3] - inRange[2])
    r = Y + (2.0 - 2.0*Kr)*Cr
    b = Y + (2.0 - 2.0*Kb)*Cb
    g = (Y - Kr*r - Kb*b)/(1.0 - Kr -Kb)
    r *= outRange[1] - outRange[0]
    g *= outRange[1] - outRange[0]
    b *= outRange[1] - outRange[0]
    r += outRange[0]
    g += outRange[0]
    b += outRange[0]
    if intOut:
        r = np.round(r).astype(int)
        g = np.round(g).astype(int)
        b = np.round(b).astype(int)
    rgb = tstack((r, g, b))
    return rgb

def rgb_to_YcCbcCrc(rgb, outRange = RANGE['legal_10_YC'], is_10_bits_system = True):
    """
    Converts an array of RGB (linear) values to the corresponding Yc'Cbc'Crc' values.
    
    Note: This fuction is specifically for use with ITU-R BT.2020 (Rec.2020) when using
    the constant luminance implementation.
    See https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-S!!PDF-E.pdf
    
    Parameters
    ----------
    rgb : array_like
        Input RGB floating point linear values.
    outRange : array of four values (optional)
        The values of Y' corresponding to 0 IRE and 100 IRE,
        and the minimum and maximum values of Cb and Cr.
        These may be floats or integer code values.
        Default: RANGE['legal_10_YC']
    is_10_bits_system : bool (optional)
        The Rec.2020 OECF varies slightly for 10 and 12 bit systems.
        Default: True
    
    Returns
    -------
    ndarray
        Yc'Cbc'Crc' array.
        These may be floats or integer code values, depending on the parameter values
        given for outRange.
    
    Examples
    --------
    >>> rgb = np.array([0.18, 0.18, 0.18])
    >>> rgb_to_YcCbcCrc(rgb, outRange=RANGE['legal_10_YC_int'], is_10_bits_system = True)
    array([422, 512, 512])
    """
    intOut = isinstance(outRange[0], int) and isinstance(outRange[1], int) and isinstance(outRange[2], int) and isinstance(outRange[3], int)
    r, g, b = tsplit(rgb)
    Yc = 0.2627*r + 0.6780*g + 0.0593*b
    Yc = REC_2020_COLOURSPACE.OECF(Yc, is_10_bits_system=is_10_bits_system)
    r = REC_2020_COLOURSPACE.OECF(r, is_10_bits_system=is_10_bits_system)
    b = REC_2020_COLOURSPACE.OECF(b, is_10_bits_system=is_10_bits_system)
    Cbc = np.where((b - Yc) <= 0.0, (b - Yc)/1.9404, (b - Yc)/1.5816)
    Crc = np.where((r - Yc) <= 0.0, (r - Yc)/1.7184, (r - Yc)/0.9936)
    Yc *= outRange[1] - outRange[0]
    Yc += outRange[0]
    Cbc *= outRange[3] - outRange[2]
    Crc *= outRange[3] - outRange[2]
    Cbc += (outRange[3] + outRange[2])/2
    Crc += (outRange[3] + outRange[2])/2
    if intOut:
        Yc = np.round(Yc).astype(int)
        Cbc = np.round(Cbc).astype(int)
        Crc = np.round(Crc).astype(int)
    YcCbcCrc = tstack((Yc, Cbc, Crc))
    return YcCbcCrc

def YcCbcCrc_to_rgb(YcCbcCrc, inRange = RANGE['legal_10_YC'], is_10_bits_system = True):
    """
    Converts an array of Yc'Cbc'Crc' values to the corresponding RGB (linear) values.
    
    Note: This fuction is specifically for use with ITU-R BT.2020 (Rec.2020) when using
    the constant luminance implementation.
    See https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-S!!PDF-E.pdf
    
    Parameters
    ----------
    YcCbcCrc : array_like
        Input Yc'Cbc'Crc' values. These may be floats or integers.
    inRange : array of four values (optional)
        The values of Yc' corresponding to 0 IRE and 100 IRE,
        and the minimum and maximum values of Cbc' and Crc'.
        These may be floats or integer code values.
        Default = RANGE['legal_10_YC']
    is_10_bits_system : bool (optional)
        The Rec.2020 EOCF varies slightly for 10 and 12 bit systems.
        Default: True

    
    Returns
    -------
    ndarray
        RGB values corresponding to the input colour.
        These will be floating point linear values.
    
    Examples
    --------
    >>> YcCbcCrc = np.array([1689, 2048, 2048])
    >>> YcCbcCrc_to_rgb(YcCbcCrc, inRange=RANGE['legal_12_YC_int'], is_10_bits_system = False)
    array([ 0.18009037,  0.18009037,  0.18009037])
    """
    Yc, Cbc, Crc = tsplit(YcCbcCrc.astype(float))
    Yc -= inRange[0]
    Cbc -= (inRange[3] + inRange[2])/2.0
    Crc -= (inRange[3] + inRange[2])/2.0
    Yc *= 1.0/(inRange[1] - inRange[0])
    Cbc *= 1.0/(inRange[3] - inRange[2])
    Crc *= 1.0/(inRange[3] - inRange[2])
    b = np.where(Cbc <= 0.0, Cbc*1.9404 + Yc, Cbc*1.5816 + Yc)
    r = np.where(Crc <= 0.0, Crc*1.7184 + Yc, Crc*0.9936 + Yc)
    Yc = REC_2020_COLOURSPACE.EOCF(Yc, is_10_bits_system=is_10_bits_system)
    b = REC_2020_COLOURSPACE.EOCF(b, is_10_bits_system=is_10_bits_system)
    r = REC_2020_COLOURSPACE.EOCF(r, is_10_bits_system=is_10_bits_system)
    g = (Yc - 0.0593*b - 0.2627*r)/0.6780
    rgb = tstack((r, g, b))
    return rgb
    