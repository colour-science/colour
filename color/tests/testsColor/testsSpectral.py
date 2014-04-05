#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**testsSpectral.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines units tests for :mod:`color.spectral` module.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	External imports.
#**********************************************************************************************************************
import numpy
import sys

if sys.version_info[:2] <= (2, 6):
	import unittest2 as unittest
else:
	import unittest

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
from color.spectral import AbstractColorMatchingFunctions
from color.spectral import RGB_ColorMatchingFunctions
from color.spectral import SpectralPowerDistribution
from color.spectral import XYZ_ColorMatchingFunctions

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["SAMPLE_SPD_DATA",
		   "RESAMPLE_SAMPLE_SPD_DATA",
		   "SPARSE_SAMPLE_SPD_DATA",
		   "STANDARD_CIE_1931_2_DEGREE_OBSERVER",
		   "CMFS_DATA",
		   "SpectralDistributionTestCase",
		   "AbstractColorMatchingFunctionsTestCase",
		   "RGB_ColorMatchingFunctionsTestCase",
		   "XYZ_ColorMatchingFunctionsTestCase"]

SAMPLE_SPD_DATA = {340: 0.0000,
				   360: 0.0000,
				   380: 0.0000,
				   400: 0.0641,
				   420: 0.0645,
				   440: 0.0562,
				   460: 0.0537,
				   480: 0.0559,
				   500: 0.0651,
				   520: 0.0705,
				   540: 0.0772,
				   560: 0.0870,
				   580: 0.1128,
				   600: 0.1360,
				   620: 0.1511,
				   640: 0.1688,
				   660: 0.1996,
				   680: 0.2397,
				   700: 0.2852,
				   720: 0.0000,
				   740: 0.0000,
				   760: 0.0000,
				   780: 0.0000,
				   800: 0.0000,
				   820: 0.0000}

SPARSE_SAMPLE_SPD_DATA = numpy.array([0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.0641,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.0645,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.0562,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.0537,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.0559,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.0651,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.0705,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.0772,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.087,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.1128,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.136,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.1511,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.1688,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.1996,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.2397,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.2852,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.,
									  0.])

RESAMPLE_SAMPLE_SPD_DATA = numpy.array([0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.003205,
										0.00641,
										0.009615,
										0.01282,
										0.016025,
										0.01923,
										0.022435,
										0.02564,
										0.028845,
										0.03205,
										0.035255,
										0.03846,
										0.041665,
										0.04487,
										0.048075,
										0.05128,
										0.054485,
										0.05769,
										0.060895,
										0.0641,
										0.06412,
										0.06414,
										0.06416,
										0.06418,
										0.0642,
										0.06422,
										0.06424,
										0.06426,
										0.06428,
										0.0643,
										0.06432,
										0.06434,
										0.06436,
										0.06438,
										0.0644,
										0.06442,
										0.06444,
										0.06446,
										0.06448,
										0.0645,
										0.064085,
										0.06367,
										0.063255,
										0.06284,
										0.062425,
										0.06201,
										0.061595,
										0.06118,
										0.060765,
										0.06035,
										0.059935,
										0.05952,
										0.059105,
										0.05869,
										0.058275,
										0.05786,
										0.057445,
										0.05703,
										0.056615,
										0.0562,
										0.056075,
										0.05595,
										0.055825,
										0.0557,
										0.055575,
										0.05545,
										0.055325,
										0.0552,
										0.055075,
										0.05495,
										0.054825,
										0.0547,
										0.054575,
										0.05445,
										0.054325,
										0.0542,
										0.054075,
										0.05395,
										0.053825,
										0.0537,
										0.05381,
										0.05392,
										0.05403,
										0.05414,
										0.05425,
										0.05436,
										0.05447,
										0.05458,
										0.05469,
										0.0548,
										0.05491,
										0.05502,
										0.05513,
										0.05524,
										0.05535,
										0.05546,
										0.05557,
										0.05568,
										0.05579,
										0.0559,
										0.05636,
										0.05682,
										0.05728,
										0.05774,
										0.0582,
										0.05866,
										0.05912,
										0.05958,
										0.06004,
										0.0605,
										0.06096,
										0.06142,
										0.06188,
										0.06234,
										0.0628,
										0.06326,
										0.06372,
										0.06418,
										0.06464,
										0.0651,
										0.06537,
										0.06564,
										0.06591,
										0.06618,
										0.06645,
										0.06672,
										0.06699,
										0.06726,
										0.06753,
										0.0678,
										0.06807,
										0.06834,
										0.06861,
										0.06888,
										0.06915,
										0.06942,
										0.06969,
										0.06996,
										0.07023,
										0.0705,
										0.070835,
										0.07117,
										0.071505,
										0.07184,
										0.072175,
										0.07251,
										0.072845,
										0.07318,
										0.073515,
										0.07385,
										0.074185,
										0.07452,
										0.074855,
										0.07519,
										0.075525,
										0.07586,
										0.076195,
										0.07653,
										0.076865,
										0.0772,
										0.07769,
										0.07818,
										0.07867,
										0.07916,
										0.07965,
										0.08014,
										0.08063,
										0.08112,
										0.08161,
										0.0821,
										0.08259,
										0.08308,
										0.08357,
										0.08406,
										0.08455,
										0.08504,
										0.08553,
										0.08602,
										0.08651,
										0.087,
										0.08829,
										0.08958,
										0.09087,
										0.09216,
										0.09345,
										0.09474,
										0.09603,
										0.09732,
										0.09861,
										0.0999,
										0.10119,
										0.10248,
										0.10377,
										0.10506,
										0.10635,
										0.10764,
										0.10893,
										0.11022,
										0.11151,
										0.1128,
										0.11396,
										0.11512,
										0.11628,
										0.11744,
										0.1186,
										0.11976,
										0.12092,
										0.12208,
										0.12324,
										0.1244,
										0.12556,
										0.12672,
										0.12788,
										0.12904,
										0.1302,
										0.13136,
										0.13252,
										0.13368,
										0.13484,
										0.136,
										0.136755,
										0.13751,
										0.138265,
										0.13902,
										0.139775,
										0.14053,
										0.141285,
										0.14204,
										0.142795,
										0.14355,
										0.144305,
										0.14506,
										0.145815,
										0.14657,
										0.147325,
										0.14808,
										0.148835,
										0.14959,
										0.150345,
										0.1511,
										0.151985,
										0.15287,
										0.153755,
										0.15464,
										0.155525,
										0.15641,
										0.157295,
										0.15818,
										0.159065,
										0.15995,
										0.160835,
										0.16172,
										0.162605,
										0.16349,
										0.164375,
										0.16526,
										0.166145,
										0.16703,
										0.167915,
										0.1688,
										0.17034,
										0.17188,
										0.17342,
										0.17496,
										0.1765,
										0.17804,
										0.17958,
										0.18112,
										0.18266,
										0.1842,
										0.18574,
										0.18728,
										0.18882,
										0.19036,
										0.1919,
										0.19344,
										0.19498,
										0.19652,
										0.19806,
										0.1996,
										0.201605,
										0.20361,
										0.205615,
										0.20762,
										0.209625,
										0.21163,
										0.213635,
										0.21564,
										0.217645,
										0.21965,
										0.221655,
										0.22366,
										0.225665,
										0.22767,
										0.229675,
										0.23168,
										0.233685,
										0.23569,
										0.237695,
										0.2397,
										0.241975,
										0.24425,
										0.246525,
										0.2488,
										0.251075,
										0.25335,
										0.255625,
										0.2579,
										0.260175,
										0.26245,
										0.264725,
										0.267,
										0.269275,
										0.27155,
										0.273825,
										0.2761,
										0.278375,
										0.28065,
										0.282925,
										0.2852,
										0.27094,
										0.25668,
										0.24242,
										0.22816,
										0.2139,
										0.19964,
										0.18538,
										0.17112,
										0.15686,
										0.1426,
										0.12834,
										0.11408,
										0.09982,
										0.08556,
										0.0713,
										0.05704,
										0.04278,
										0.02852,
										0.01426,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.,
										0.])
STANDARD_CIE_1931_2_DEGREE_OBSERVER = {
"barX": {380: 0.001368,
		 385: 0.002236,
		 390: 0.004243,
		 395: 0.007650,
		 400: 0.014310,
		 405: 0.023190,
		 410: 0.043510,
		 415: 0.077630,
		 420: 0.134380,
		 425: 0.214770,
		 430: 0.283900,
		 435: 0.328500,
		 440: 0.348280,
		 445: 0.348060,
		 450: 0.336200,
		 455: 0.318700,
		 460: 0.290800,
		 465: 0.251100,
		 470: 0.195360,
		 475: 0.142100,
		 480: 0.095640,
		 485: 0.057950,
		 490: 0.032010,
		 495: 0.014700,
		 500: 0.004900,
		 505: 0.002400,
		 510: 0.009300,
		 515: 0.029100,
		 520: 0.063270,
		 525: 0.109600,
		 530: 0.165500,
		 535: 0.225750,
		 540: 0.290400,
		 545: 0.359700,
		 550: 0.433450,
		 555: 0.512050,
		 560: 0.594500,
		 565: 0.678400,
		 570: 0.762100,
		 575: 0.842500,
		 580: 0.916300,
		 585: 0.978600,
		 590: 1.026300,
		 595: 1.056700,
		 600: 1.062200,
		 605: 1.045600,
		 610: 1.002600,
		 615: 0.938400,
		 620: 0.854450,
		 625: 0.751400,
		 630: 0.642400,
		 635: 0.541900,
		 640: 0.447900,
		 645: 0.360800,
		 650: 0.283500,
		 655: 0.218700,
		 660: 0.164900,
		 665: 0.121200,
		 670: 0.087400,
		 675: 0.063600,
		 680: 0.046770,
		 685: 0.032900,
		 690: 0.022700,
		 695: 0.015840,
		 700: 0.011359,
		 705: 0.008111,
		 710: 0.005790,
		 715: 0.004109,
		 720: 0.002899,
		 725: 0.002049,
		 730: 0.001440,
		 735: 0.001000,
		 740: 0.000690,
		 745: 0.000476,
		 750: 0.000332,
		 755: 0.000235,
		 760: 0.000166,
		 765: 0.000117,
		 770: 0.000083,
		 775: 0.000059,
		 780: 0.000042},
"barY": {380: 0.000039,
		 385: 0.000064,
		 390: 0.000120,
		 395: 0.000217,
		 400: 0.000396,
		 405: 0.000640,
		 410: 0.001210,
		 415: 0.002180,
		 420: 0.004000,
		 425: 0.007300,
		 430: 0.011600,
		 435: 0.016840,
		 440: 0.023000,
		 445: 0.029800,
		 450: 0.038000,
		 455: 0.048000,
		 460: 0.060000,
		 465: 0.073900,
		 470: 0.090980,
		 475: 0.112600,
		 480: 0.139020,
		 485: 0.169300,
		 490: 0.208020,
		 495: 0.258600,
		 500: 0.323000,
		 505: 0.407300,
		 510: 0.503000,
		 515: 0.608200,
		 520: 0.710000,
		 525: 0.793200,
		 530: 0.862000,
		 535: 0.914850,
		 540: 0.954000,
		 545: 0.980300,
		 550: 0.994950,
		 555: 1.000000,
		 560: 0.995000,
		 565: 0.978600,
		 570: 0.952000,
		 575: 0.915400,
		 580: 0.870000,
		 585: 0.816300,
		 590: 0.757000,
		 595: 0.694900,
		 600: 0.631000,
		 605: 0.566800,
		 610: 0.503000,
		 615: 0.441200,
		 620: 0.381000,
		 625: 0.321000,
		 630: 0.265000,
		 635: 0.217000,
		 640: 0.175000,
		 645: 0.138200,
		 650: 0.107000,
		 655: 0.081600,
		 660: 0.061000,
		 665: 0.044580,
		 670: 0.032000,
		 675: 0.023200,
		 680: 0.017000,
		 685: 0.011920,
		 690: 0.008210,
		 695: 0.005723,
		 700: 0.004102,
		 705: 0.002929,
		 710: 0.002091,
		 715: 0.001484,
		 720: 0.001047,
		 725: 0.000740,
		 730: 0.000520,
		 735: 0.000361,
		 740: 0.000249,
		 745: 0.000172,
		 750: 0.000120,
		 755: 0.000085,
		 760: 0.000060,
		 765: 0.000042,
		 770: 0.000030,
		 775: 0.000021,
		 780: 0.000015},
"barZ": {380: 0.006450,
		 385: 0.010550,
		 390: 0.020050,
		 395: 0.036210,
		 400: 0.067850,
		 405: 0.110200,
		 410: 0.207400,
		 415: 0.371300,
		 420: 0.645600,
		 425: 1.039050,
		 430: 1.385600,
		 435: 1.622960,
		 440: 1.747060,
		 445: 1.782600,
		 450: 1.772110,
		 455: 1.744100,
		 460: 1.669200,
		 465: 1.528100,
		 470: 1.287640,
		 475: 1.041900,
		 480: 0.812950,
		 485: 0.616200,
		 490: 0.465180,
		 495: 0.353300,
		 500: 0.272000,
		 505: 0.212300,
		 510: 0.158200,
		 515: 0.111700,
		 520: 0.078250,
		 525: 0.057250,
		 530: 0.042160,
		 535: 0.029840,
		 540: 0.020300,
		 545: 0.013400,
		 550: 0.008750,
		 555: 0.005750,
		 560: 0.003900,
		 565: 0.002750,
		 570: 0.002100,
		 575: 0.001800,
		 580: 0.001650,
		 585: 0.001400,
		 590: 0.001100,
		 595: 0.001000,
		 600: 0.000800,
		 605: 0.000600,
		 610: 0.000340,
		 615: 0.000240,
		 620: 0.000190,
		 625: 0.000100,
		 630: 0.000050,
		 635: 0.000030,
		 640: 0.000020,
		 645: 0.000010,
		 650: 0.000000,
		 655: 0.000000,
		 660: 0.000000,
		 665: 0.000000,
		 670: 0.000000,
		 675: 0.000000,
		 680: 0.000000,
		 685: 0.000000,
		 690: 0.000000,
		 695: 0.000000,
		 700: 0.000000,
		 705: 0.000000,
		 710: 0.000000,
		 715: 0.000000,
		 720: 0.000000,
		 725: 0.000000,
		 730: 0.000000,
		 735: 0.000000,
		 740: 0.000000,
		 745: 0.000000,
		 750: 0.000000,
		 755: 0.000000,
		 760: 0.000000,
		 765: 0.000000,
		 770: 0.000000,
		 775: 0.000000,
		 780: 0.000000}
}

CMFS_DATA = {380: (0.001368, 3.9e-05, 0.00645),
			 385: (0.002236, 6.4e-05, 0.01055),
			 390: (0.004243, 0.00012, 0.02005),
			 395: (0.00765, 0.000217, 0.03621),
			 400: (0.01431, 0.000396, 0.06785),
			 405: (0.02319, 0.00064, 0.1102),
			 410: (0.04351, 0.00121, 0.2074),
			 415: (0.07763, 0.00218, 0.3713),
			 420: (0.13438, 0.004, 0.6456),
			 425: (0.21477, 0.0073, 1.03905),
			 430: (0.2839, 0.0116, 1.3856),
			 435: (0.3285, 0.01684, 1.62296),
			 440: (0.34828, 0.023, 1.74706),
			 445: (0.34806, 0.0298, 1.7826),
			 450: (0.3362, 0.038, 1.77211),
			 455: (0.3187, 0.048, 1.7441),
			 460: (0.2908, 0.06, 1.6692),
			 465: (0.2511, 0.0739, 1.5281),
			 470: (0.19536, 0.09098, 1.28764),
			 475: (0.1421, 0.1126, 1.0419),
			 480: (0.09564, 0.13902, 0.81295),
			 485: (0.05795, 0.1693, 0.6162),
			 490: (0.03201, 0.20802, 0.46518),
			 495: (0.0147, 0.2586, 0.3533),
			 500: (0.0049, 0.323, 0.272),
			 505: (0.0024, 0.4073, 0.2123),
			 510: (0.0093, 0.503, 0.1582),
			 515: (0.0291, 0.6082, 0.1117),
			 520: (0.06327, 0.71, 0.07825),
			 525: (0.1096, 0.7932, 0.05725),
			 530: (0.1655, 0.862, 0.04216),
			 535: (0.22575, 0.91485, 0.02984),
			 540: (0.2904, 0.954, 0.0203),
			 545: (0.3597, 0.9803, 0.0134),
			 550: (0.43345, 0.99495, 0.00875),
			 555: (0.51205, 1.0, 0.00575),
			 560: (0.5945, 0.995, 0.0039),
			 565: (0.6784, 0.9786, 0.00275),
			 570: (0.7621, 0.952, 0.0021),
			 575: (0.8425, 0.9154, 0.0018),
			 580: (0.9163, 0.87, 0.00165),
			 585: (0.9786, 0.8163, 0.0014),
			 590: (1.0263, 0.757, 0.0011),
			 595: (1.0567, 0.6949, 0.001),
			 600: (1.0622, 0.631, 0.0008),
			 605: (1.0456, 0.5668, 0.0006),
			 610: (1.0026, 0.503, 0.00034),
			 615: (0.9384, 0.4412, 0.00024),
			 620: (0.85445, 0.381, 0.00019),
			 625: (0.7514, 0.321, 0.0001),
			 630: (0.6424, 0.265, 5e-05),
			 635: (0.5419, 0.217, 3e-05),
			 640: (0.4479, 0.175, 2e-05),
			 645: (0.3608, 0.1382, 1e-05),
			 650: (0.2835, 0.107, 0.0),
			 655: (0.2187, 0.0816, 0.0),
			 660: (0.1649, 0.061, 0.0),
			 665: (0.1212, 0.04458, 0.0),
			 670: (0.0874, 0.032, 0.0),
			 675: (0.0636, 0.0232, 0.0),
			 680: (0.04677, 0.017, 0.0),
			 685: (0.0329, 0.01192, 0.0),
			 690: (0.0227, 0.00821, 0.0),
			 695: (0.01584, 0.005723, 0.0),
			 700: (0.011359, 0.004102, 0.0),
			 705: (0.008111, 0.002929, 0.0),
			 710: (0.00579, 0.002091, 0.0),
			 715: (0.004109, 0.001484, 0.0),
			 720: (0.002899, 0.001047, 0.0),
			 725: (0.002049, 0.00074, 0.0),
			 730: (0.00144, 0.00052, 0.0),
			 735: (0.001, 0.000361, 0.0),
			 740: (0.00069, 0.000249, 0.0),
			 745: (0.000476, 0.000172, 0.0),
			 750: (0.000332, 0.00012, 0.0),
			 755: (0.000235, 8.5e-05, 0.0),
			 760: (0.000166, 6e-05, 0.0),
			 765: (0.000117, 4.2e-05, 0.0),
			 770: (8.3e-05, 3e-05, 0.0),
			 775: (5.9e-05, 2.1e-05, 0.0),
			 780: (4.2e-05, 1.5e-05, 0.0)}

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
class SpectralDistributionTestCase(unittest.TestCase):
	"""
	Defines :class:`color.spectral.SpectralDistribution` class units tests methods.
	"""

	def testRequiredAttributes(self):
		"""
		Tests presence of required attributes.
		"""

		requiredAttributes = ("name",
							  "spd",
							  "wavelengths",
							  "values",
							  "shape")

		for attribute in requiredAttributes:
			self.assertIn(attribute, dir(SpectralPowerDistribution))

	def testRequiredMethods(self):
		"""
		Tests presence of required methods.
		"""

		requiredMethods = ("get",
						   "resparse",
						   "resample")

		for method in requiredMethods:
			self.assertIn(method, dir(SpectralPowerDistribution))

	def test__getitem__(self):
		"""
		Tests :func:`color.spectral.SpectralDistribution.__getitem__` method.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		self.assertEqual(spd[340], 0.)
		self.assertEqual(spd[620], 0.1511)
		self.assertEqual(spd[820], 0.)

	def test__iter__(self):
		"""
		Tests :func:`color.spectral.SpectralDistribution.__iter__` method.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		self.assertEqual(
			dict([(key, value) for key, value in spd]),
			SAMPLE_SPD_DATA)

	def test__contains__(self):
		"""
		Tests :func:`color.spectral.SpectralDistribution.__contains__` method.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		self.assertIn(340, spd)
		self.assertIn(460, spd)
		self.assertNotIn(461, spd)

	def test__len__(self):
		"""
		Tests :func:`color.spectral.SpectralDistribution.__len__` method.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		self.assertEqual(len(spd), 25)

	def testWavelengths(self):
		"""
		Tests :attr:`color.spectral.SpectralDistribution.wavelengths` attribute.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		numpy.testing.assert_almost_equal(spd.wavelengths,
										  sorted(SAMPLE_SPD_DATA))

	def testValues(self):
		"""
		Tests :attr:`color.spectral.SpectralDistribution.values` attribute.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		numpy.testing.assert_almost_equal(spd.values, [v for k, v in sorted(SAMPLE_SPD_DATA.items())])

	def testShape(self):
		"""
		Tests :attr:`color.spectral.SpectralDistribution.shape` attribute.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		self.assertTupleEqual(spd.shape, (340, 820, 20))

	def testGet(self):
		"""
		Tests :func:`color.spectral.SpectralDistribution.get` method.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		self.assertEqual(spd.get(340), 0.)
		self.assertEqual(spd.get(620), 0.1511)
		self.assertEqual(spd.get(820), 0.)
		self.assertEqual(spd.get(900, -1), -1)

	def testResample(self):
		"""
		Tests :func:`color.spectral.SpectralDistribution.resample` method.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		numpy.testing.assert_almost_equal(spd.resample(steps=1).values,
										  RESAMPLE_SAMPLE_SPD_DATA)

	def testResparse(self):
		"""
		Tests :func:`color.spectral.SpectralDistribution.resparse` method.
		"""

		spd = SpectralPowerDistribution(name="", spd=SAMPLE_SPD_DATA)

		numpy.testing.assert_almost_equal(spd.resparse(steps=1).values,
										  SPARSE_SAMPLE_SPD_DATA)

class AbstractColorMatchingFunctionsTestCase(unittest.TestCase):
	"""
	Defines :class:`color.spectral.AbstractColorMatchingFunctions` class units tests methods.
	"""

	def testRequiredAttributes(self):
		"""
		Tests presence of required attributes.
		"""

		requiredAttributes = ("name",
							  "mapping",
							  "labels",
							  "cmfs",
							  "x",
							  "y",
							  "z",
							  "wavelengths",
							  "values",
							  "shape")

		for attribute in requiredAttributes:
			self.assertIn(attribute, dir(AbstractColorMatchingFunctions))

	def testRequiredMethods(self):
		"""
		Tests presence of required methods.
		"""

		requiredMethods = ("get",
						   "resample",
						   "resparse")

		for method in requiredMethods:
			self.assertIn(method, dir(AbstractColorMatchingFunctions))

	def test__getitem__(self):
		"""
		Tests :func:`color.spectral.AbstractColorMatchingFunctions.__getitem__` method.
		"""

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping={"x": "barX",
													   "y": "barY",
													   "z": "barZ"},
											  cmfs=STANDARD_CIE_1931_2_DEGREE_OBSERVER,
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		self.assertTupleEqual(cmfs[380], (0.001368, 3.9e-05, 0.00645))
		self.assertTupleEqual(cmfs[600], (1.0622, 0.631, 0.0008))
		self.assertTupleEqual(cmfs[700], (0.011359, 0.004102, 0.))

	def test__iter__(self):
		"""
		Tests :func:`color.spectral.AbstractColorMatchingFunctions.__iter__` method.
		"""
		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping={"x": "barX",
													   "y": "barY",
													   "z": "barZ"},
											  cmfs=STANDARD_CIE_1931_2_DEGREE_OBSERVER,
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		self.assertEqual(dict([(key, value) for key, value in cmfs]), CMFS_DATA)

	def test__contains__(self):
		"""
		Tests :func:`color.spectral.AbstractColorMatchingFunctions.__contains__` method.
		"""

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping={"x": "barX",
													   "y": "barY",
													   "z": "barZ"},
											  cmfs=STANDARD_CIE_1931_2_DEGREE_OBSERVER,
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		self.assertIn(380, cmfs)
		self.assertIn(460, cmfs)
		self.assertNotIn(461, cmfs)

	def test__len__(self):
		"""
		Tests :func:`color.spectral.AbstractColorMatchingFunctions.__len__` method.
		"""

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping={"x": "barX",
													   "y": "barY",
													   "z": "barZ"},
											  cmfs=STANDARD_CIE_1931_2_DEGREE_OBSERVER,
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		self.assertEqual(len(cmfs), 81)

	def testWavelengths(self):
		"""
		Tests :attr:`color.spectral.AbstractColorMatchingFunctions.wavelengths` attribute.
		"""

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping={"x": "barX",
													   "y": "barY",
													   "z": "barZ"},
											  cmfs=STANDARD_CIE_1931_2_DEGREE_OBSERVER,
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		numpy.testing.assert_almost_equal(cmfs.wavelengths, sorted(STANDARD_CIE_1931_2_DEGREE_OBSERVER.get("barX")))

	def testValues(self):
		"""
		Tests :attr:`color.spectral.AbstractColorMatchingFunctions.values` attribute.
		"""

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping={"x": "barX",
													   "y": "barY",
													   "z": "barZ"},
											  cmfs=STANDARD_CIE_1931_2_DEGREE_OBSERVER,
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		numpy.testing.assert_almost_equal(cmfs.values,
										  numpy.array(zip(*([v for k, v in sorted(
											  STANDARD_CIE_1931_2_DEGREE_OBSERVER.get("barX").items())],
															[v for k, v in sorted(
																STANDARD_CIE_1931_2_DEGREE_OBSERVER.get(
																	"barY").items())],
															[v for k, v in sorted(
																STANDARD_CIE_1931_2_DEGREE_OBSERVER.get(
																	"barZ").items())]))))

	def testShape(self):
		"""
		Tests :attr:`color.spectral.AbstractColorMatchingFunctions.shape` attribute.
		"""

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping={"x": "barX",
													   "y": "barY",
													   "z": "barZ"},
											  cmfs=STANDARD_CIE_1931_2_DEGREE_OBSERVER,
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		self.assertTupleEqual(cmfs.shape, (380, 780, 5))

	def testGet(self):
		"""
		Tests :func:`color.spectral.AbstractColorMatchingFunctions.get` method.
		"""

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping={"x": "barX",
													   "y": "barY",
													   "z": "barZ"},
											  cmfs=STANDARD_CIE_1931_2_DEGREE_OBSERVER,
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		self.assertTupleEqual(cmfs.get(380), (0.001368, 3.9e-05, 0.00645))
		self.assertTupleEqual(cmfs.get(600), (1.0622, 0.631, 0.0008))
		self.assertTupleEqual(cmfs.get(700), (0.011359, 0.004102, 0.))
		self.assertTupleEqual(cmfs.get(900, (0, 0, 0)), (0, 0, 0))

	def testResample(self):
		"""
		Tests :func:`color.spectral.AbstractColorMatchingFunctions.resample` method.
		"""

		mapping = {"x": "barX",
				   "y": "barY",
				   "z": "barZ"}

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping=mapping,
											  cmfs={"barX": SAMPLE_SPD_DATA,
													"barY": SAMPLE_SPD_DATA,
													"barZ": SAMPLE_SPD_DATA},
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		cmfs.resample(steps=1)
		for i in mapping.iterkeys():
			numpy.testing.assert_almost_equal(getattr(cmfs, i).values, RESAMPLE_SAMPLE_SPD_DATA)

	def testResparse(self):
		"""
		Tests :func:`color.spectral.AbstractColorMatchingFunctions.resparse` method.
		"""

		mapping = {"x": "barX",
				   "y": "barY",
				   "z": "barZ"}

		cmfs = AbstractColorMatchingFunctions(name="",
											  mapping=mapping,
											  cmfs={"barX": SAMPLE_SPD_DATA,
													"barY": SAMPLE_SPD_DATA,
													"barZ": SAMPLE_SPD_DATA},
											  labels={"x": "barX",
													  "y": "barY",
													  "z": "barZ"})

		cmfs.resparse(steps=1)
		for i in mapping.iterkeys():
			numpy.testing.assert_almost_equal(getattr(cmfs, i).values, SPARSE_SAMPLE_SPD_DATA)

class RGB_ColorMatchingFunctionsTestCase(unittest.TestCase):
	"""
	Defines :class:`color.spectral.RGB_ColorMatchingFunctions` class units tests methods.
	"""

	def testRequiredAttributes(self):
		"""
		Tests presence of required attributes.
		"""

		requiredAttributes = ("name",
							  "mapping",
							  "labels",
							  "cmfs",
							  "x",
							  "y",
							  "z",
							  "wavelengths",
							  "values",
							  "shape",
							  "barR",
							  "barG",
							  "barB")

		for attribute in requiredAttributes:
			self.assertIn(attribute, dir(RGB_ColorMatchingFunctions))

	def testRequiredMethods(self):
		"""
		Tests presence of required methods.
		"""

		requiredMethods = ("get",
						   "resample",
						   "resparse")

		for method in requiredMethods:
			self.assertIn(method, dir(RGB_ColorMatchingFunctions))

class XYZ_ColorMatchingFunctionsTestCase(unittest.TestCase):
	"""
	Defines :class:`color.spectral.XYZ_ColorMatchingFunctions` class units tests methods.
	"""

	def testRequiredAttributes(self):
		"""
		Tests presence of required attributes.
		"""

		requiredAttributes = ("name",
							  "mapping",
							  "labels",
							  "cmfs",
							  "x",
							  "y",
							  "z",
							  "wavelengths",
							  "values",
							  "shape",
							  "barX",
							  "barY",
							  "barZ")

		for attribute in requiredAttributes:
			self.assertIn(attribute, dir(XYZ_ColorMatchingFunctions))

	def testRequiredMethods(self):
		"""
		Tests presence of required methods.
		"""

		requiredMethods = ("get",
						   "resample",
						   "resparse")

		for method in requiredMethods:
			self.assertIn(method, dir(XYZ_ColorMatchingFunctions))

if __name__ == "__main__":
	unittest.main()
