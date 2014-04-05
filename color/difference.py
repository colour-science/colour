#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**difference.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package color *difference* manipulation objects.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***    External imports.
#**********************************************************************************************************************
import math
import numpy

#**********************************************************************************************************************
#***    Internal imports.
#**********************************************************************************************************************
import color.verbose

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal - Michael Parsons - The Moving picture Company"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
		   "delta_E_CIE_1976",
		   "delta_E_CIE_1994",
		   "delta_E_CIE_2000",
		   "delta_E_CMC"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#***    Module classes and definitions.
#**********************************************************************************************************************
def delta_E_CIE_1976(lab1, lab2):
	"""
	Returns the difference between two given *CIE Lab* colors using *CIE 1976* recommendation.

	Reference: http://brucelindbloom.com/Eqn_DeltaE_CIE76.html

	Usage::

		>>> lab1 = numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1))
		>>> lab2 = numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))
		>>> delta_E_CIE_1976(lab1, lab2)
		451.713301974

	:param lab1: *CIE Lab* color 1.
	:type lab1: Matrix (3x1)
	:param lab2: *CIE Lab* color 2.
	:type lab2: Matrix (3x1)
	:return: Colors difference.
	:rtype: float
	"""

	L1, a1, b1 = numpy.ravel(lab1)
	L2, a2, b2 = numpy.ravel(lab2)

	return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def delta_E_CIE_1994(lab1, lab2, textiles=True):
	"""
	Returns the difference between two given *CIE Lab* colors using *CIE 1994* recommendation.

	Reference: http://brucelindbloom.com/Eqn_DeltaE_CIE94.html

	Usage::

		>>> lab1 = numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1))
		>>> lab2 = numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))
		>>> delta_E_CIE_1994(lab1, lab2)
		88.3355530575

	:param lab1: *CIE Lab* color 1.
	:type lab1: Matrix (3x1)
	:param lab2: *CIE Lab* color 2.
	:type lab2: Matrix (3x1)
	:param textiles: Application specific weights.
	:type textiles: bool
	:return: Colors difference.
	:rtype: float
	"""

	k1 = 0.048 if textiles else 0.045
	k2 = 0.014 if textiles else 0.015
	kL = 2. if textiles else 1.
	kC = 1.
	kH = 1.

	L1, a1, b1 = numpy.ravel(lab1)
	L2, a2, b2 = numpy.ravel(lab2)

	C1 = math.sqrt(a1 ** 2 + b1 ** 2)
	C2 = math.sqrt(a2 ** 2 + b2 ** 2)

	sL = 1
	sC = 1 + k1 * C1
	sH = 1 + k2 * C1

	deltaL = L1 - L2
	deltaC = C1 - C2
	deltaA = a1 - a2
	deltaB = b1 - b2

	try:
		deltaH = math.sqrt(deltaA ** 2 + deltaB ** 2 - deltaC ** 2)
	except ValueError:
		deltaH = 0.0

	L = (deltaL / (kL * sL)) ** 2
	C = (deltaC / (kC * sC)) ** 2
	H = (deltaH / (kH * sH)) ** 2

	return math.sqrt(L + C + H)

def delta_E_CIE_2000(lab1, lab2):
	"""
	Returns the difference between two given *CIE Lab* colors using *CIE 2000* recommendation.

	Reference: http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html

	Usage::

		>>> lab1 = numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1))
		>>> lab2 = numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))
		>>> delta_E_CIE_2000(lab1, lab2)
		94.0356490267

	:param lab1: *CIE Lab* color 1.
	:type lab1: Matrix (3x1)
	:param lab2: *CIE Lab* color 2.
	:type lab2: Matrix (3x1)
	:return: Colors difference.
	:rtype: float
	"""

	L1, a1, b1 = numpy.ravel(lab1)
	L2, a2, b2 = numpy.ravel(lab2)

	kL = 1.
	kC = 1.
	kH = 1.

	lBarPrime = 0.5 * (L1 + L2)

	c1 = math.sqrt(a1 * a1 + b1 * b1)
	c2 = math.sqrt(a2 * a2 + b2 * b2)

	cBar = 0.5 * (c1 + c2)
	cBar7 = cBar * cBar * cBar * cBar * cBar * cBar * cBar

	g = 0.5 * (1. - math.sqrt(cBar7 / (cBar7 + 25. ** 7)))

	a1Prime = a1 * (1. + g)
	a2Prime = a2 * (1. + g)
	c1Prime = math.sqrt(a1Prime * a1Prime + b1 * b1)
	c2Prime = math.sqrt(a2Prime * a2Prime + b2 * b2)
	cBarPrime = 0.5 * (c1Prime + c2Prime)

	h1Prime = (math.atan2(b1, a1Prime) * 180.) / math.pi
	if h1Prime < 0.:
		h1Prime += 360.

	h2Prime = (math.atan2(b2, a2Prime) * 180.) / math.pi
	if h2Prime < 0.0:
		h2Prime += 360.

	hBarPrime = 0.5 * (h1Prime + h2Prime + 360.) if math.fabs(h1Prime - h2Prime) > 180. else 0.5 * (h1Prime + h2Prime)
	t = 1. - 0.17 * math.cos(math.pi * (hBarPrime - 30.) / 180.) + 0.24 * math.cos(math.pi * (2. * hBarPrime) / 180.) + \
		0.32 * math.cos(math.pi * (3. * hBarPrime + 6.) / 180.) - 0.20 * math.cos(
		math.pi * (4. * hBarPrime - 63.) / 180.)

	if math.fabs(h2Prime - h1Prime) <= 180.:
		deltahPrime = h2Prime - h1Prime
	else:
		deltahPrime = h2Prime - h1Prime + 360. if h2Prime <= h1Prime else h2Prime - h1Prime - 360.

	deltaLPrime = L2 - L1
	deltaCPrime = c2Prime - c1Prime
	deltaHPrime = 2. * math.sqrt(c1Prime * c2Prime) * math.sin(math.pi * (0.5 * deltahPrime) / 180.)

	sL = 1. + ((0.015 * (lBarPrime - 50.) * (lBarPrime - 50.)) / math.sqrt(20. + (lBarPrime - 50.) * (lBarPrime - 50.)))
	sC = 1. + 0.045 * cBarPrime
	sH = 1. + 0.015 * cBarPrime * t

	deltaTheta = 30. * math.exp(-((hBarPrime - 275.) / 25.) * ((hBarPrime - 275.) / 25.))

	cBarPrime7 = cBarPrime * cBarPrime * cBarPrime * cBarPrime * cBarPrime * cBarPrime * cBarPrime

	rC = math.sqrt(cBarPrime7 / (cBarPrime7 + 25. ** 7))
	rT = -2. * rC * math.sin(math.pi * (2. * deltaTheta) / 180.)

	return math.sqrt((deltaLPrime / (kL * sL)) * (deltaLPrime / (kL * sL)) + \
					 (deltaCPrime / (kC * sC)) * (deltaCPrime / (kC * sC)) + \
					 (deltaHPrime / (kH * sH)) * (deltaHPrime / (kH * sH)) + \
					 (deltaCPrime / (kC * sC)) * (deltaHPrime / (kH * sH)) * rT)

def delta_E_CMC(lab1, lab2, l=2., c=1.):
	"""
	Returns the difference between two given *CIE Lab* colors using *Color Measurement Committee* recommendation.
	The quasimetric has two parameters: *Lightness* (l) and *chroma* (c), allowing the users to weight the difference based on the ratio of l:c.
	Commonly used values are 2:1 for acceptability and 1:1 for the threshold of imperceptibility.

	Reference: http://brucelindbloom.com/Eqn_DeltaE_CMC.html

	Usage::

		>>> lab1 = numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1))
		>>> lab2 = numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))
		>>> delta_E_CMC(lab1, lab2)
		172.704771287

	:param lab1: *CIE Lab* color 1.
	:type lab1: Matrix (3x1)
	:param lab2: *CIE Lab* color 2.
	:type lab2: Matrix (3x1)
	:param l: Lightness weighting factor.
	:type l: float
	:param c: Chroma weighting factor.
	:type c: float
	:return: Colors difference.
	:rtype: float
	"""

	L1, a1, b1 = numpy.ravel(lab1)
	L2, a2, b2 = numpy.ravel(lab2)

	c1 = math.sqrt(a1 * a1 + b1 * b1)
	c2 = math.sqrt(a2 * a2 + b2 * b2)
	sl = 0.511 if L1 < 16. else (0.040975 * L1) / (1. + 0.01765 * L1)
	sc = 0.0638 * c1 / (1. + 0.0131 * c1) + 0.638
	h1 = 0. if c1 < 0.000001 else (math.atan2(b1, a1) * 180.) / math.pi

	while h1 < 0.:
		h1 += 360.

	while h1 >= 360.:
		h1 -= 360.

	t = 0.56 + math.fabs(0.2 * math.cos((math.pi * (h1 + 168.)) / 180.)) if h1 >= 164. and h1 <= 345. else \
		0.36 + math.fabs(0.4 * math.cos((math.pi * (h1 + 35.)) / 180.))
	c4 = c1 * c1 * c1 * c1
	f = math.sqrt(c4 / (c4 + 1900.))
	sh = sc * (f * t + 1. - f)

	deltaL = L1 - L2
	deltaC = c1 - c2
	deltaA = a1 - a2
	deltaB = b1 - b2
	deltaH2 = deltaA * deltaA + deltaB * deltaB - deltaC * deltaC

	v1 = deltaL / (l * sl)
	v2 = deltaC / (c * sc)
	v3 = sh

	return math.sqrt(v1 * v1 + v2 * v2 + (deltaH2 / (v3 * v3)))