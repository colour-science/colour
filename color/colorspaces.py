#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**colorspaces.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *colorspaces* data and manipulation objects.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***    External imports.
#**********************************************************************************************************************
import numpy
from collections import namedtuple

#**********************************************************************************************************************
#***	Internal Imports.
#**********************************************************************************************************************
import color.illuminants
import color.verbose

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
		   "COLORSPACE",
		   "CIE_RGB_PRIMARIES",
		   "CIE_RGB_WHITEPOINT",
		   "CIE_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_CIE_RGB_MATRIX",
		   "CIE_RGB_TRANSFER_FUNCTION",
		   "CIE_RGB_INVERSE_TRANSFER_FUNCTION",
		   "CIE_RGB_COLORSPACE",
		   "ACES_RGB_PRIMARIES",
		   "ACES_RGB_WHITEPOINT",
		   "ACES_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_ACES_RGB_MATRIX",
		   "ACES_RGB_TRANSFER_FUNCTION",
		   "ACES_RGB_INVERSE_TRANSFER_FUNCTION",
		   "ACES_RGB_COLORSPACE",
		   "sRGB_PRIMARIES",
		   "REC_709_PRIMARIES",
		   "sRGB_WHITEPOINT",
		   "REC_709_WHITEPOINT",
		   "sRGB_TO_XYZ_MATRIX",
		   "REC_709_TO_XYZ_MATRIX",
		   "XYZ_TO_sRGB_MATRIX",
		   "XYZ_TO_REC_709_MATRIX",
		   "sRGB_TRANSFER_FUNCTION",
		   "sRGB_INVERSE_TRANSFER_FUNCTION",
		   "REC_709_TRANSFER_FUNCTION",
		   "REC_709_INVERSE_TRANSFER_FUNCTION",
		   "sRGB_COLORSPACE",
		   "REC_709_COLORSPACE",
		   "ADOBE_RGB_1998_PRIMARIES",
		   "ADOBE_RGB_1998_WHITEPOINT",
		   "ADOBE_RGB_1998_TO_XYZ_MATRIX",
		   "XYZ_TO_ADOBE_RGB_1998_MATRIX",
		   "ADOBE_RGB_1998_TRANSFER_FUNCTION",
		   "ADOBE_RGB_1998_INVERSE_TRANSFER_FUNCTION",
		   "ADOBE_RGB_1998_COLORSPACE",
		   "PROPHOTO_RGB_PRIMARIES",
		   "PROPHOTO_RGB_WHITEPOINT",
		   "PROPHOTO_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_PROPHOTO_RGB_MATRIX",
		   "PROPHOTO_RGB_TRANSFER_FUNCTION",
		   "PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION",
		   "PROPHOTO_RGB_COLORSPACE",
		   "DCI_P3_PRIMARIES",
		   "DCI_P3_WHITEPOINT",
		   "DCI_P3_TO_XYZ_MATRIX",
		   "XYZ_TO_DCI_P3_MATRIX",
		   "DCI_P3_TRANSFER_FUNCTION",
		   "DCI_P3_INVERSE_TRANSFER_FUNCTION",
		   "DCI_P3_COLORSPACE",
		   "COLORSPACES",
		   "POINTER_GAMUT_DATA",
		   "colorspace"]

LOGGER = color.verbose.installLogger()

COLORSPACE = namedtuple("Colorspace", ("name",
									   "primaries",
									   "whitepoint",
									   "toXYZ",
									   "fromXYZ",
									   "transferFunction",
									   "inverseTransferFunction"))

#**********************************************************************************************************************
#*** *CIE RGB*
#**********************************************************************************************************************
# http://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright.E2.80.93Guild_data
CIE_RGB_PRIMARIES = numpy.matrix([1., 0.,
								  0., 1.,
								  0., 0.]).reshape((3, 2))

CIE_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("E")

CIE_RGB_TO_XYZ_MATRIX = 1 / 0.17697 * numpy.matrix([0.49, 0.31, 0.20,
													0.17697, 0.81240, 0.01063,
													0.00, 0.01, 0.99]).reshape((3, 3))

XYZ_TO_CIE_RGB_MATRIX = CIE_RGB_TO_XYZ_MATRIX.getI()

CIE_RGB_TRANSFER_FUNCTION = lambda x: x

CIE_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x

CIE_RGB_COLORSPACE = COLORSPACE("CIE RGB",
								CIE_RGB_PRIMARIES,
								CIE_RGB_WHITEPOINT,
								CIE_RGB_TO_XYZ_MATRIX,
								XYZ_TO_CIE_RGB_MATRIX,
								CIE_RGB_TRANSFER_FUNCTION,
								CIE_RGB_INVERSE_TRANSFER_FUNCTION)

#**********************************************************************************************************************
#*** *ACES RGB*
#**********************************************************************************************************************
# http://www.oscars.org/science-technology/council/projects/aces.html
# https://www.dropbox.com/sh/iwd09buudm3lfod/gyjDF-k7oC/ACES_v1.0.1.pdf: 4.1.2 Color space chromaticities
ACES_RGB_PRIMARIES = numpy.matrix([0.73470, 0.26530,
								   0.00000, 1.00000,
								   0.00010, -0.07700]).reshape((3, 2))

ACES_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D60")

# https://www.dropbox.com/sh/iwd09buudm3lfod/gyjDF-k7oC/ACES_v1.0.1.pdf: 4.1.4 Converting ACES RGB values to CIE XYZ values
ACES_RGB_TO_XYZ_MATRIX = numpy.matrix([9.52552396e-01, 0.00000000e+00, 9.36786317e-05,
									   3.43966450e-01, 7.28166097e-01, -7.21325464e-02,
									   0.00000000e+00, 0.00000000e+00, 1.00882518e+00]).reshape((3, 3))

XYZ_TO_ACES_RGB_MATRIX = ACES_RGB_TO_XYZ_MATRIX.getI()

ACES_RGB_TRANSFER_FUNCTION = lambda x: x

ACES_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x

ACES_RGB_COLORSPACE = COLORSPACE("ACES RGB",
								 ACES_RGB_PRIMARIES,
								 ACES_RGB_WHITEPOINT,
								 ACES_RGB_TO_XYZ_MATRIX,
								 XYZ_TO_ACES_RGB_MATRIX,
								 ACES_RGB_TRANSFER_FUNCTION,
								 ACES_RGB_INVERSE_TRANSFER_FUNCTION)

#**********************************************************************************************************************
#*** *sRGB / Rec. 709*
#**********************************************************************************************************************
# http://www.color.org/srgb.pdf
# http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf: 1 Opto-electronic conversion
sRGB_PRIMARIES = REC_709_PRIMARIES = numpy.matrix([0.6400, 0.3300,
												   0.3000, 0.6000,
												   0.1500, 0.0600]).reshape((3, 2))

sRGB_WHITEPOINT = REC_709_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get(
	"D65")

sRGB_TO_XYZ_MATRIX = REC_709_TO_XYZ_MATRIX = numpy.matrix([0.41238656, 0.35759149, 0.18045049,
														   0.21263682, 0.71518298, 0.0721802,
														   0.01933062, 0.11919716, 0.95037259]).reshape((3, 3))

XYZ_TO_sRGB_MATRIX = XYZ_TO_REC_709_MATRIX = sRGB_TO_XYZ_MATRIX.getI()

def __sRGBTransferFunction(RGB):
	"""
	Defines the *sRGB* colorspace transfer function.

	Reference: http://www.color.org/srgb.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x * 12.92 if x <= 0.0031308 else 1.055 * (x ** (1 / 2.4)) - 0.055, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __sRGBInverseTransferFunction(RGB):
	"""
	Defines the *sRGB* colorspace inverse transfer function.

	Reference: http://www.color.org/srgb.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x / 12.92 if x <= 0.0031308 else ((x + 0.055) / 1.055) ** 2.4, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

sRGB_TRANSFER_FUNCTION = __sRGBTransferFunction

sRGB_INVERSE_TRANSFER_FUNCTION = __sRGBInverseTransferFunction

def __rec709TransferFunction(RGB):
	"""
	Defines the *Rec. 709* colorspace transfer function.

	Reference: http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf: 1 Opto-electronic conversion

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x * 4.5 if x <= 0.018 else 1.099 * (x ** 0.45) - 0.099, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __rec709InverseTransferFunction(RGB):
	"""
	Defines the *Rec. 709* colorspace inverse transfer function.

	Reference: http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf: 1 Opto-electronic conversion

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x / 4.5 if x <= 0.018 else ((x + 0.099) / 1.099) ** (1 / 0.45), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

REC_709_TRANSFER_FUNCTION = __rec709TransferFunction

REC_709_INVERSE_TRANSFER_FUNCTION = __rec709InverseTransferFunction

sRGB_COLORSPACE = COLORSPACE("sRGB",
							 sRGB_PRIMARIES,
							 sRGB_WHITEPOINT,
							 sRGB_TO_XYZ_MATRIX,
							 XYZ_TO_sRGB_MATRIX,
							 sRGB_TRANSFER_FUNCTION,
							 sRGB_INVERSE_TRANSFER_FUNCTION)

REC_709_COLORSPACE = COLORSPACE("Rec. 709",
								sRGB_PRIMARIES,
								sRGB_WHITEPOINT,
								sRGB_TO_XYZ_MATRIX,
								XYZ_TO_sRGB_MATRIX,
								REC_709_TRANSFER_FUNCTION,
								REC_709_INVERSE_TRANSFER_FUNCTION)

#**********************************************************************************************************************
#*** *Adobe RGB 1998*
#**********************************************************************************************************************
# http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
ADOBE_RGB_1998_PRIMARIES = numpy.matrix([0.6400, 0.3300,
										 0.2100, 0.7100,
										 0.1500, 0.0600]).reshape((3, 2))

ADOBE_RGB_1998_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D65")

# http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf: 4.3.5.3 Converting RGB to normalized XYZ values
ADOBE_RGB_1998_TO_XYZ_MATRIX = numpy.matrix([0.57666809, 0.18556195, 0.1881985,
											 0.29734449, 0.62737611, 0.0752794,
											 0.02703132, 0.07069027, 0.99117879]).reshape((3, 3))

XYZ_TO_ADOBE_RGB_1998_MATRIX = ADOBE_RGB_1998_TO_XYZ_MATRIX.getI()

def __adobe1998TransferFunction(RGB):
	"""
	Defines the *Adobe RGB 1998* colorspace transfer function.

	Reference: http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.19921875), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __adobe1998InverseTransferFunction(RGB):
	"""
	Defines the *Adobe RGB 1998* colorspace inverse transfer function.

	Reference: http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.19921875, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

ADOBE_RGB_1998_TRANSFER_FUNCTION = __adobe1998TransferFunction

ADOBE_RGB_1998_INVERSE_TRANSFER_FUNCTION = __adobe1998InverseTransferFunction

ADOBE_RGB_1998_COLORSPACE = COLORSPACE("Adobe RGB 1998",
									   ADOBE_RGB_1998_PRIMARIES,
									   ADOBE_RGB_1998_WHITEPOINT,
									   ADOBE_RGB_1998_TO_XYZ_MATRIX,
									   XYZ_TO_ADOBE_RGB_1998_MATRIX,
									   ADOBE_RGB_1998_TRANSFER_FUNCTION,
									   ADOBE_RGB_1998_INVERSE_TRANSFER_FUNCTION)

#**********************************************************************************************************************
#*** *ProPhoto RGB*
#**********************************************************************************************************************
# http://www.color.org/ROMMRGB.pdf
PROPHOTO_RGB_PRIMARIES = numpy.matrix([0.7347, 0.2653,
									   0.1596, 0.8404,
									   0.0366, 0.0001]).reshape((3, 2))

PROPHOTO_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

PROPHOTO_RGB_TO_XYZ_MATRIX = numpy.matrix([7.97667235e-01, 1.35192231e-01, 3.13525290e-02,
										   2.88037454e-01, 7.11876883e-01, 8.56626476e-05,
										   0.00000000e+00, 0.00000000e+00, 8.25188285e-01]).reshape((3, 3))

XYZ_TO_PROPHOTO_RGB_MATRIX = PROPHOTO_RGB_TO_XYZ_MATRIX.getI()

def __prophotoRgbTransferFunction(RGB):
	"""
	Defines the *Prophoto RGB* colorspace transfer function.

	Reference: http://www.color.org/ROMMRGB.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: 0.003473 + 0.0622829 * x if x < 0.03125 else 0.003473 + 0.996527 * x ** ( 1 / 1.8 ),
			  numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __prophotoRgbTransferFunction(RGB):
	"""
	Defines the *Prophoto RGB* colorspace transfer function.

	Reference: http://www.color.org/ROMMRGB.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x * 16 if x < 0.001953 else x ** ( 1 / 1.8), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __prophotoRgbInverseTransferFunction(RGB):
	"""
	Defines the *Prophoto RGB* colorspace inverse transfer function.

	Reference: http://www.color.org/ROMMRGB.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x / 16 if x < 0.001953 else x ** 1.8, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

PROPHOTO_RGB_TRANSFER_FUNCTION = __prophotoRgbTransferFunction

PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION = __prophotoRgbInverseTransferFunction

PROPHOTO_RGB_COLORSPACE = COLORSPACE("ProPhoto RGB",
									 PROPHOTO_RGB_PRIMARIES,
									 PROPHOTO_RGB_WHITEPOINT,
									 PROPHOTO_RGB_TO_XYZ_MATRIX,
									 XYZ_TO_PROPHOTO_RGB_MATRIX,
									 PROPHOTO_RGB_TRANSFER_FUNCTION,
									 PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION)

#**********************************************************************************************************************
#*** *DCI-P3*
#**********************************************************************************************************************
# http://www.hp.com/united-states/campaigns/workstations/pdfs/lp2480zx-dci--p3-emulation.pdf
DCI_P3_PRIMARIES = numpy.matrix([0.680, 0.320,
								 0.265, 0.690,
								 0.150, 0.060]).reshape((3, 2))

DCI_P3_WHITEPOINT = (0.314, 0.351)

DCI_P3_TO_XYZ_MATRIX = numpy.matrix([0.44516982, 0.27713441, 0.17228267,
									 0.20949168, 0.72159525, 0.06891307,
									 0., 0.04706056, 0.90735539]).reshape((3, 3))

XYZ_TO_DCI_P3_MATRIX = DCI_P3_TO_XYZ_MATRIX.getI()

DCI_P3_TRANSFER_FUNCTION = lambda x: x

DCI_P3_INVERSE_TRANSFER_FUNCTION = lambda x: x

DCI_P3_COLORSPACE = COLORSPACE("DCI-P3",
							   DCI_P3_PRIMARIES,
							   DCI_P3_WHITEPOINT,
							   DCI_P3_TO_XYZ_MATRIX,
							   XYZ_TO_DCI_P3_MATRIX,
							   DCI_P3_TRANSFER_FUNCTION,
							   DCI_P3_INVERSE_TRANSFER_FUNCTION)

COLORSPACES = {"CIE RGB": CIE_RGB_COLORSPACE,
			   "ACES RGB": ACES_RGB_COLORSPACE,
			   "sRGB": sRGB_COLORSPACE,
			   "Rec. 709": REC_709_COLORSPACE,
			   "Adobe RGB 1998": ADOBE_RGB_1998_COLORSPACE,
			   "ProPhoto RGB": PROPHOTO_RGB_COLORSPACE,
			   "DCI-P3": DCI_P3_COLORSPACE}

#**********************************************************************************************************************
#*** *Pointer Gamut*
#**********************************************************************************************************************
# http://www.cis.rit.edu/research/mcsl2/online/PointerData.xls
POINTER_GAMUT_DATA = ((0.659, 0.316),
					  (0.634, 0.351),
					  (0.594, 0.391),
					  (0.557, 0.427),
					  (0.523, 0.462),
					  (0.482, 0.491),
					  (0.444, 0.515),
					  (0.409, 0.546),
					  (0.371, 0.558),
					  (0.332, 0.573),
					  (0.288, 0.584),
					  (0.242, 0.576),
					  (0.202, 0.530),
					  (0.177, 0.454),
					  (0.151, 0.389),
					  (0.151, 0.330),
					  (0.162, 0.295),
					  (0.157, 0.266),
					  (0.159, 0.245),
					  (0.142, 0.214),
					  (0.141, 0.195),
					  (0.129, 0.168),
					  (0.138, 0.141),
					  (0.145, 0.129),
					  (0.145, 0.106),
					  (0.161, 0.094),
					  (0.188, 0.084),
					  (0.252, 0.104),
					  (0.324, 0.127),
					  (0.393, 0.165),
					  (0.451, 0.199),
					  (0.508, 0.226))

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
def colorspace(name=None,
			   primaries=None,
			   whitepoint=None,
			   toXYZ=None,
			   fromXYZ=None,
			   transferFunction=None,
			   inverseTransferFunction=None):
	"""
	Defines a factory for :data:`color.colorspaces.COLORSPACE` attribute.

	:param name: Colorspace name.
	:type name: unicode
	:param primaries: Primaries matrix.
	:type primaries: Matrix (3x3)
	:param whitepoint: Reference whitepoint / illuminant.
	:type whitepoint: tuple
	:param toXYZ: Matrix to convert from colorspace to *CIE XYZ* colorspace.
	:type toXYZ: Matrix (3x3)
	:param fromXYZ: Matrix to convert from *CIE XYZ* colorspace to colorspace.
	:type fromXYZ: Matrix (3x3)
	:param transferFunction: Transfer function to convert values from linear to screen.
	:type transferFunction: object
	:param inverseTransferFunction: Inverse transfer function to convert values from screen to linear.
	:type inverseTransferFunction: object
	:return: Colorspace.
	:rtype: Colorspace
	"""

	return COLORSPACE(name, primaries, whitepoint, toXYZ, fromXYZ, transferFunction, inverseTransferFunction)