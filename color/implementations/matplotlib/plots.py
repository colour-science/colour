#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**plots.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package plotting objects.

**Others:**

"""

#**********************************************************************************************************************
#***    External imports.
#**********************************************************************************************************************
import bisect
import functools
import matplotlib
import matplotlib.image
import matplotlib.path
import matplotlib.pyplot
import matplotlib.ticker
import numpy
import os
import pylab
import random
from collections import namedtuple

#**********************************************************************************************************************
#***	Internal Imports.
#**********************************************************************************************************************
import color.blackbody
import color.colorCheckers
import color.colorspaces
import color.illuminants
import color.exceptions
import color.matrix
import color.spectral
import color.temperature
import color.transformations
import color.dataStructures
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
		   "RESOURCES_DIRECTORY",
		   "DEFAULT_FIGURE_SIZE",
		   "COLOR_PARAMETER",
		   "XYZ_to_sRGB",
		   "figureSize",
		   "boundingBox",
		   "display",
		   "colorParameter",
		   "colorParametersPlot",
		   "singleColorPlot",
		   "multiColorPlot",
		   "colorCheckerPlot",
		   "singleSpectralPowerDistributionPlot",
		   "multiSpectralPowerDistributionPlot",
		   "singleColorMatchingFunctionsPlot",
		   "multiColorMatchingFunctionsPlot",
		   "singleIlluminantRelativeSpdPlot",
		   "multiIlluminantsRelativeSpdPlot",
		   "visibleSpectrumPlot",
		   "CIE_1931_chromaticityDiagramColorsPlot",
		   "CIE_1931_chromaticityDiagramPlot",
		   "colorspaces_CIE_1931_chromaticityDiagramPlot",
		   "planckianLocus_CIE_1931_chromaticityDiagramPlot",
		   "CIE_1960_UCS_chromaticityDiagramColorsPlot",
		   "CIE_1960_UCS_chromaticityDiagramPlot",
		   "planckianLocus_CIE_1960_UCS_chromaticityDiagramPlot",
		   "CIE_1976_UCS_chromaticityDiagramColorsPlot",
		   "CIE_1976_UCS_chromaticityDiagramPlot",
		   "multiTransferFunctionPlot",
		   "singleTransferFunctionPlot",
		   "blackbodySpectralRadiancePlot",
		   "blackbodyColorsPlot"]

LOGGER = color.verbose.installLogger()

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")

DEFAULT_FIGURE_SIZE = 14, 7

COLOR_PARAMETER = namedtuple("ColorParameter", ("name", "RGB", "x", "y0", "y1"))

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
# Defining default figure size.
pylab.rcParams["figure.figsize"] = DEFAULT_FIGURE_SIZE

# Defining an alternative font that can display scientific notations.
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

def XYZ_to_sRGB(XYZ, illuminant=color.colorspaces.sRGB_COLORSPACE.whitepoint):
	"""
	Converts from *CIE XYZ* colorspace to *sRGB* colorspace.

	:param XYZ: *CIE XYZ* matrix.
	:type XYZ: Matrix (3x1)
	:param illuminant: Source illuminant chromaticity coordinates.
	:type illuminant: tuple
	:return: *sRGB* color matrix.
	:rtype: Matrix (3x1)
	"""

	return color.transformations.XYZ_to_RGB(XYZ,
											illuminant,
											color.colorspaces.sRGB_COLORSPACE.whitepoint,
											"CAT02",
											color.colorspaces.sRGB_COLORSPACE.fromXYZ,
											color.colorspaces.sRGB_COLORSPACE.transferFunction)

def figureSize(size=DEFAULT_FIGURE_SIZE):
	"""
	Sets figures sizes.

	:param size: Figure size.
	:type size: tuple
	:return: Object.
	:rtype: object
	"""

	def figureSizeDecorator(object):
		"""
		Sets figures sizes.

		:param object: Object to decorate.
		:type object: object
		:return: Object.
		:rtype: object
		"""

		@functools.wraps(object)
		def figureSizeWrapper(*args, **kwargs):
			"""
			Sets figures sizes.

			:param \*args: Arguments.
			:type \*args: \*
			:param \*\*kwargs: Keywords arguments.
			:type \*\*kwargs: \*\*
			:return: Object.
			:rtype: object
			"""

			pylab.rcParams["figure.figsize"] = kwargs.get("figureSize") if kwargs.get(
				"figureSize") is not None else size

			try:
				return object(*args, **kwargs)
			finally:
				pylab.rcParams["figure.figsize"] = DEFAULT_FIGURE_SIZE

		return figureSizeWrapper

	return figureSizeDecorator

def aspect(**kwargs):
	"""
	Sets the figure aspect.

	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	settings = color.dataStructures.Structure(**{"title": None,
												 "xLabel": None,
												 "yLabel": None,
												 "legend": False,
												 "legendLocation": "upper right",
												 "tickerX": False,
												 "tickerY": False,
												 "tickerXlocator": matplotlib.ticker.AutoMinorLocator(2),
												 "tickerYlocator": matplotlib.ticker.AutoMinorLocator(2),
												 "noTicks": False,
												 "grid": False,
												 "gridAxis": "both",
												 "axisXLine": False,
												 "axisYLine": False,
												 "aspect": None})
	settings.update(kwargs)

	settings.title and pylab.title(settings.title)
	settings.xLabel and pylab.xlabel(settings.xLabel)
	settings.yLabel and pylab.ylabel(settings.yLabel)
	settings.legend and pylab.legend(loc=settings.legendLocation)
	settings.tickerX and matplotlib.pyplot.gca().xaxis.set_minor_locator(settings.tickerXlocator)
	settings.tickerY and matplotlib.pyplot.gca().yaxis.set_minor_locator(settings.tickerYlocator)
	if settings.noTicks:
		matplotlib.pyplot.gca().set_xticks([])
		matplotlib.pyplot.gca().set_yticks([])
	settings.grid and pylab.grid(which=settings.gridAxis)
	settings.axisXLine and pylab.axvline(0, color="black", linestyle="--")
	settings.axisYLine and pylab.axhline(0, color="black", linestyle="--")
	settings.aspect and matplotlib.pyplot.axes().set_aspect(settings.aspect)

	return True

def boundingBox(**kwargs):
	"""
	Sets the plot bounding box.

	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	settings = color.dataStructures.Structure(**{"boundingBox": None,
													   "tightenX": False,
													   "tightenY": False,
													   "limits": [0., 1., 0., 1.],
													   "margins": [0., 0., 0., 0.]})
	settings.update(kwargs)

	if settings.boundingBox is None:
		xLimitMin, xLimitMax, yLimitMin, yLimitMax = settings.limits
		xMarginMin, xMarginMax, yMarginMin, yMarginMax = settings.margins
		settings.tightenX and pylab.xlim(xLimitMin + xMarginMin, xLimitMax + xMarginMax)
		settings.tightenY and pylab.ylim(yLimitMin + yMarginMin, yLimitMax + yMarginMax)
	else:
		pylab.xlim(settings.boundingBox[0], settings.boundingBox[1])
		pylab.ylim(settings.boundingBox[2], settings.boundingBox[3])

	return True

def display(**kwargs):
	"""
	Sets the figure display.

	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	settings = color.dataStructures.Structure(**{"standalone": True,
													   "filename": None})
	settings.update(kwargs)

	if settings.standalone:
		if settings.filename is not None:
			pylab.savefig(**kwargs)
		else:
			pylab.show()
		pylab.close()

	return True

def colorParameter(name=None, RGB=None, x=None, y0=None, y1=None):
	"""
	Defines a factory for :attr:`color.implementations.matplotlib.plots.COLOR_PARAMETER` attribute.

	:param name: Color name.
	:type name: unicode
	:param RGB: RGB Color.
	:type RGB: Matrix
	:param x: X data.
	:type x: float
	:param y0: Y0 data.
	:type y0: float
	:param y1: Y1 data.
	:type y1: float
	:return: ColorParameter.
	:rtype: ColorParameter
	"""

	return COLOR_PARAMETER(name, RGB, x, y0, y1)

def colorParametersPlot(colorParameters,
						plotY0=True,
						plotY1=True,
						**kwargs):
	"""
	Plots given color colorParameters.

	Usage::

		>>> cp1 = colorParameter(x=390, RGB=[0.03009021, 0., 0.12300545])
		>>> cp2 = colorParameter(x=391, RGB=[0.03434063, 0., 0.13328537], y0=0, y1=0.25)
		>>> cp3 = colorParameter(x=392, RGB=[0.03826312, 0., 0.14276247], y0=0, y1=0.35)
		>>> cp4 = colorParameter(x=393, RGB=[0.04191844, 0., 0.15158707], y0=0, y1=0.05)
		>>> cp5 = colorParameter(x=394, RGB=[0.04535085, 0., 0.15986838], y0=0, y1=-.25)
		>>> colorParametersPlot([cp1, cp2, cp3, cp3, cp4, cp5])
		True

	:param colorParameters: ColorParameter sequence.
	:type colorParameters: list
	:param plotY0: Plot y0 line.
	:type plotY0: bool
	:param plotY1: Plot y1 line.
	:type plotY1: bool
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	for i in xrange(len(colorParameters) - 1):
		x0 = colorParameters[i].x
		x01 = colorParameters[i + 1].x
		y0 = 0. if colorParameters[i].y0 is None else colorParameters[i].y0
		y1 = 1. if colorParameters[i].y1 is None else colorParameters[i].y1
		y01 = 0. if colorParameters[i].y0 is None else colorParameters[i + 1].y0
		y11 = 1. if colorParameters[i].y1 is None else colorParameters[i + 1].y1
		polygonX = [x0, x01, x01, x0]
		polygonY = [y0, y01, y11, y1]
		pylab.fill(polygonX, polygonY, color=colorParameters[i].RGB, edgecolor=colorParameters[i].RGB)

	if all(map(lambda x: x.y0 is not None, colorParameters)):
		plotY0 and pylab.plot(map(lambda x: x.x, colorParameters), map(lambda x: x.y0, colorParameters), color="black",
							  linewidth=2.)

	if all(map(lambda x: x.y1 is not None, colorParameters)):
		plotY1 and pylab.plot(map(lambda x: x.x, colorParameters), map(lambda x: x.y1, colorParameters), color="black",
							  linewidth=2.)

	yLimitMin0, yLimitMax0 = min(map(lambda x: 0. if x.y0 is None else x.y0, colorParameters)), \
							 max(map(lambda x: 1. if x.y0 is None else x.y0, colorParameters))
	yLimitMin1, yLimitMax1 = min(map(lambda x: 0. if x.y1 is None else x.y1, colorParameters)), \
							 max(map(lambda x: 1. if x.y1 is None else x.y1, colorParameters))

	settings = {"xLabel": "Parameter",
				"yLabel": "Color",
				"limits": [min(map(lambda x: 0. if x.x is None else x.x, colorParameters)),
						   max(map(lambda x: 1. if x.x is None else x.x, colorParameters)),
						   yLimitMin0,
						   yLimitMax1]}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

def singleColorPlot(colorParameter,
					**kwargs):
	"""
	Plots given color.

	Usage::

		>>> singleColorPlot(colorParameter(RGB=(0.32315746, 0.32983556, 0.33640183)))
		True

	:param colorParameter: ColorParameter.
	:type colorParameter: ColorParameter
	:param displayText: Display color text.
	:type displayText: bool
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	return multiColorPlot([colorParameter], **kwargs)

def multiColorPlot(colorParameters,
				   width=1.,
				   height=1.,
				   spacing=0.,
				   across=3,
				   displayText=True,
				   textSize="large",
				   textOffset=0.075,
				   **kwargs):
	"""
	Plots given colors.

	Usage::

		>>> cp1 = colorParameter(RGB=(0.45293517, 0.31732158, 0.26414773))
		>>> cp2 = colorParameter(RGB=(0.77875824, 0.5772645,  0.50453169)
		>>> multiColorPlot([cp1, cp2])
		True

	:param colorParameters: ColorParameter sequence.
	:type colorParameters: list
	:param width: Color polygon width.
	:type width: float
	:param height: Color polygon height.
	:type height: float
	:param spacing: Color polygons spacing.
	:type spacing: float
	:param across: Color polygons count per row.
	:type across: int
	:param displayText: Display color text.
	:type displayText: bool
	:param textSize: Color text size.
	:type textSize: float
	:param textOffset: Color text offset.
	:type textOffset: float
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	offsetX = offsetY = 0
	xLimitMin, xLimitMax, yLimitMin, yLimitMax = 0, width, 0, height
	for i, colorParameter in enumerate(colorParameters):
		if i % across == 0 and i != 0:
			offsetX = 0
			offsetY -= height + spacing

		x0 = offsetX
		x1 = offsetX + width
		y0 = offsetY
		y1 = offsetY + height

		polygonX = [x0, x1, x1, x0]
		polygonY = [y0, y0, y1, y1]
		pylab.fill(polygonX, polygonY, color=colorParameters[i].RGB)
		if colorParameter.name is not None and displayText:
			pylab.text(x0 + textOffset, y0 + textOffset, colorParameter.name, size=textSize)

		offsetX += width + spacing

	xLimitMax = min(len(colorParameters), across)
	xLimitMax = xLimitMax * width + xLimitMax * spacing - spacing
	yLimitMin = offsetY

	settings = {"tightenX": True,
				"tightenY": True,
				"noTicks": True,
				"limits": [xLimitMin, xLimitMax, yLimitMin, yLimitMax],
				"aspect": "equal"}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

def colorCheckerPlot(colorChecker="ColorChecker 2005",
					 **kwargs):
	"""
	Plots given color checker.

	Usage::

		>>> colorCheckerPlot()
		True

	:param colorChecker: Color checker name.
	:type colorChecker: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	colorChecker, name = color.colorCheckers.COLORCHECKERS.get(colorChecker), colorChecker
	if colorChecker is None:
		raise color.exceptions.ProgrammingError(
			"Color checker '{0}' not found in color checkers: '{1}'.".format(name,
																			 sorted(
																				 color.colorCheckers.COLORCHECKERS.keys())))

	_, data, illuminant = colorChecker
	colorParameters = []
	for index, label, x, y, Y in data:
		XYZ = color.transformations.xyY_to_XYZ((x, y, Y))
		RGB = XYZ_to_sRGB(XYZ, illuminant)

		colorParameters.append(colorParameter(label.title(), numpy.clip(numpy.ravel(RGB), 0, 1)))

	backgroundColor = "0.1"
	matplotlib.pyplot.gca().patch.set_facecolor(backgroundColor)

	width = height = 1.0
	spacing = 0.25
	across = 6

	settings = {"standalone": False,
				"width": width,
				"height": height,
				"spacing": spacing,
				"across": across,
				"margins": [-0.125, 0.125, -0.5, 0.125]}
	settings.update(kwargs)

	multiColorPlot(colorParameters, **settings)

	textX = width * (across / 2) + (across * (spacing / 2)) - spacing / 2
	textY = -(len(colorParameters) / across + spacing / 2)

	pylab.text(textX,
			   textY,
			   "{0} - {1} - Color Rendition Chart".format(name, color.colorspaces.sRGB_COLORSPACE.name),
			   color="0.95",
			   ha="center")

	settings.update({"title": name,
					 "facecolor": backgroundColor,
					 "edgecolor": None,
					 "standalone": True})

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

def singleSpectralPowerDistributionPlot(spd,
										cmfs=color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(
											"Standard CIE 1931 2 Degree Observer"),
										**kwargs):
	"""
	Plots given spectral power distribution.

	Usage::

		>>> spd = color.SpectralPowerDistribution(name="Custom", spd={400: 0.0641, 420: 0.0645, 440: 0.0562})
		>>> singleSpectralPowerDistributionPlot(spd)
		True

	:param spd: Spectral power distribution to plot.
	:type spd: SpectralPowerDistribution
	:param cmfs: Standard observer color matching functions used for spectrum creation.
	:type cmfs: XYZ_ColorMatchingFunctions
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	start, end, steps = cmfs.shape
	spd.resample(start, end, steps)
	wavelengths = numpy.arange(start, end + steps, steps)

	colors = []
	y1 = []

	for wavelength, value in spd:
		XYZ = color.transformations.wavelength_to_XYZ(wavelength, cmfs)
		colors.append(XYZ_to_sRGB(XYZ))
		y1.append(value)

	colors = numpy.array(map(numpy.ravel, colors))
	colors *= 1. / numpy.max(colors)
	colors = numpy.clip(colors, 0, 1)

	settings = {"title": "'{0}' - {1}".format(spd.name, cmfs.name),
				"xLabel": u"Wavelength λ (nm)",
				"yLabel": "Spectral Power Distribution",
				"tightenX": True,
				"tickerX": True,
				"tickerY": True}

	settings.update(kwargs)
	return colorParametersPlot(map(lambda x: colorParameter(x=x[0], y1=x[1], RGB=x[2]), zip(wavelengths, y1, colors)),
							   **settings)

def multiSpectralPowerDistributionPlot(spds,
									   **kwargs):
	"""
	Plots given spectral power distributions.

	Usage::

		>>> spd1 = color.SpectralPowerDistribution(name="Custom1", spd={400: 0.0641, 420: 0.0645, 440: 0.0562})
		>>> spd2 = color.SpectralPowerDistribution(name="Custom2", spd={400: 0.134, 420: 0.789, 440: 1.289})
		>>> multiSpectralPowerDistributionPlot([spd1, spd2]))
		True

	:param spds: Spectral power distributions to plot.
	:type spds: list
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	xLimitMin, xLimitMax, yLimitMin, yLimitMax = [], [], [], []
	for spd in spds:
		wavelengths, values = zip(*[(key, value) for key, value in spd])

		start, end, steps = spd.shape
		xLimitMin.append(start)
		xLimitMax.append(end)
		yLimitMin.append(min(values))
		yLimitMax.append(max(values))

		pylab.plot(wavelengths, values, label=spd.name, linewidth=2.)

	settings = {"xLabel": u"Wavelength λ (nm)",
				"yLabel": "Spectral Power Distribution",
				"tightenX": True,
				"legend": True,
				"legendLocation": "upper left",
				"tickerX": True,
				"tickerY": True,
				"limits": [min(xLimitMin), max(xLimitMax), min(yLimitMin), max(yLimitMax)]}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

def singleColorMatchingFunctionsPlot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
	"""
	Plots given standard observer *CIE XYZ* color matching functions.

	Usage::

		>>> singleColorMatchingFunctionsPlot("Standard CIE 1931 2 Degree Observer")
		True

	:param cmfs: Standard observer color matching functions to plot.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	settings = {"title": "'{0}' - Color Matching Functions".format(cmfs)}
	settings.update(kwargs)

	return multiColorMatchingFunctionsPlot([cmfs], **settings)

def multiColorMatchingFunctionsPlot(cmfss=["Standard CIE 1931 2 Degree Observer",
										   "Standard CIE 1964 10 Degree Observer"],
									**kwargs):
	"""
	Plots given standard observers *CIE XYZ* color matching functions.

	Usage::

		>>> multiColorMatchingFunctionsPlot(["Standard CIE 1931 2 Degree Observer", "Standard CIE 1964 10 Degree Observer"])
		True

	:param cmfss: Standard observers color matching functions to plot.
	:type cmfss: list
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	xLimitMin, xLimitMax, yLimitMin, yLimitMax = [], [], [], []
	for axis, rgb in (("x", [1., 0., 0.]),
					  ("y", [0., 1., 0.]),
					  ("z", [0., 0., 1.])):
		for i, cmfs in enumerate(cmfss):
			cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
			if cmfs is None:
				raise color.exceptions.ProgrammingError(
					"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(
						name,
						sorted(
							color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

			rgb = map(lambda x: reduce(lambda y, _: y * 0.25, xrange(i), x), rgb)
			wavelengths, values = zip(*[(key, value) for key, value in getattr(cmfs, axis)])

			start, end, steps = cmfs.shape
			xLimitMin.append(start)
			xLimitMax.append(end)
			yLimitMin.append(min(values))
			yLimitMax.append(max(values))

			pylab.plot(wavelengths, values, color=rgb, label=u"{0} - {1}".format(cmfs.labels.get(axis), cmfs.name),
					   linewidth=2.)

	settings = {"title": "{0} - Color Matching Functions".format(", ".join(cmfss)),
				"xLabel": u"Wavelength λ (nm)",
				"yLabel": "Tristimulus Values",
				"tightenX": True,
				"legend": True,
				"legendLocation": "upper right",
				"tickerX": True,
				"tickerY": True,
				"grid": True,
				"axisYLine": True,
				"limits": [min(xLimitMin), max(xLimitMax), min(yLimitMin), max(yLimitMax)]}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

def singleIlluminantRelativeSpdPlot(illuminant="A", cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
	"""
	Plots given single illuminant relative spectral power distribution.

	Usage::

		>>> singleIlluminantRelativeSpdPlot("A")
		True

	:param illuminant: Factory illuminant to plot.
	:type illuminant: unicode
	:param cmfs: Standard observer color matching functions to plot.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	title = "Illuminant '{0}' - {1}".format(illuminant, cmfs)

	illuminant, name = color.spectral.ILLUMINANTS_RELATIVE_SPD.get(illuminant), illuminant
	if illuminant is None:
		raise color.exceptions.ProgrammingError(
			"Illuminant '{0}' not found in factory illuminants: '{1}'.".format(name,
																			   sorted(
																				   color.spectral.ILLUMINANTS_RELATIVE_SPD.keys())))

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	settings = {"title": title,
				"yLabel": "Relative Spectral Power Distribution"}
	settings.update(kwargs)

	return singleSpectralPowerDistributionPlot(illuminant, **settings)

def multiIlluminantsRelativeSpdPlot(illuminants=["A", "C", "D50"], **kwargs):
	"""
	Plots given illuminants relative spectral power distributions.

	Usage::

		>>> multiIlluminantsRelativeSpdPlot(["A", "C", "D50"])
		True

	:param illuminants: Factory illuminants to plot.
	:type illuminants: tuple or list
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	spds = []
	for illuminant in illuminants:
		illuminant = color.spectral.ILLUMINANTS_RELATIVE_SPD.get(illuminant)
		if illuminant is None:
			raise color.exceptions.ProgrammingError(
				"Illuminant '{0}' not found in factory illuminants: '{1}'.".format(illuminant.name,
																				   sorted(
																					   color.spectral.ILLUMINANTS_RELATIVE_SPD.keys())))

		spds.append(illuminant)

	settings = {"title": "{0}- Illuminants Relative Spectral Power Distribution".format(", ".join(illuminants)),
				"yLabel": "Relative Spectral Power Distribution"}
	settings.update(kwargs)

	return multiSpectralPowerDistributionPlot(spds, **settings)

def visibleSpectrumPlot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
	"""
	Plots the visible colors spectrum using given standard observer *CIE XYZ* color matching functions.

	Usage::

		>>> visibleSpectrumPlot("Standard CIE 1931 2 Degree Observer")
		True

	:param cmfs: Standard observer color matching functions used for spectrum creation.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	cmfs.resample(360, 830)

	start, end, steps = cmfs.shape
	wavelengths = numpy.arange(start, end + steps, steps)

	colors = []
	for i in wavelengths:
		XYZ = color.transformations.wavelength_to_XYZ(i, cmfs)
		colors.append(XYZ_to_sRGB(XYZ))

	colors = numpy.array(map(numpy.ravel, colors))
	colors *= 1. / numpy.max(colors)
	colors = numpy.clip(colors, 0, 1)

	settings = {"title": "The Visible Spectrum - {0}".format(name),
				"xLabel": u"Wavelength λ (nm)",
				"tightenX": True}
	settings.update(kwargs)

	return colorParametersPlot(map(lambda x: colorParameter(x=x[0], RGB=x[1]), zip(wavelengths, colors)), **settings)

@figureSize((32, 32))
def CIE_1931_chromaticityDiagramColorsPlot(surface=1.25,
										   spacing=0.00075,
										   cmfs="Standard CIE 1931 2 Degree Observer",
										   **kwargs):
	"""
	Plots the *CIE 1931 Chromaticity Diagram* colors.

	Usage::

		>>> CIE_1931_chromaticityDiagramColorsPlot()
		True

	:param surface: Generated markers surface.
	:type surface: float
	:param spacing: Spacing between markers.
	:type spacing: float
	:param cmfs: Standard observer color matching functions used for diagram bounds.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	illuminant = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("E")

	XYZs = [value for key, value in cmfs]

	x, y = zip(*(map(lambda x: color.transformations.XYZ_to_xy(x), XYZs)))

	path = matplotlib.path.Path(zip(x, y))
	dotX, dotY, colors = [], [], []
	for i in numpy.arange(0., 1., spacing):
		for j in numpy.arange(0., 1., spacing):
			if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
				dotX.append(i)
				dotY.append(j)

				XYZ = color.transformations.xy_to_XYZ((i, j))
				RGB = XYZ_to_sRGB(XYZ, illuminant)

				RGB = numpy.ravel(RGB)
				RGB *= 1. / numpy.max(RGB)
				RGB = numpy.clip(RGB, 0, 1)

				colors.append(RGB)

	pylab.scatter(dotX, dotY, color=colors, s=surface)

	settings = {"noTicks": True,
				"boundingBox": [0., 1., 0., 1.],
				"bbox_inches": "tight",
				"pad_inches": 0}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

@figureSize((8, 8))
def CIE_1931_chromaticityDiagramPlot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
	"""
	Plots the *CIE 1931 Chromaticity Diagram*.

	Usage::

		>>> CIE_1931_chromaticityDiagramPlot()
		True

	:param cmfs: Standard observer color matching functions used for diagram bounds.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	image = matplotlib.image.imread(os.path.join(RESOURCES_DIRECTORY,
												 "CIE_1931_Chromaticity_Diagram_{0}_Small.png".format(
													 cmfs.name.replace(" ", "_"))))
	pylab.imshow(image, interpolation="nearest", extent=(0, 1, 0, 1))

	labels = [390, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700]

	wavelengths = cmfs.wavelengths
	equalEnergy = numpy.array([1. / 3.] * 2)

	XYZs = [value for key, value in cmfs]

	x, y = zip(*(map(lambda x: color.transformations.XYZ_to_xy(x), XYZs)))

	wavelengthsChromaticityCoordinates = dict(zip(wavelengths, zip(x, y)))

	pylab.plot(x, y, color="black", linewidth=2.)
	pylab.plot((x[-1], x[0]), (y[-1], y[0]), color="black", linewidth=2.)

	for label in labels:
		x, y = wavelengthsChromaticityCoordinates.get(label)
		pylab.plot(x, y, "o", color="black", linewidth=2.)

		index = bisect.bisect(wavelengths, label)
		left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
		right = wavelengths[index] if index < len(wavelengths) else wavelengths[-1]

		dx = wavelengthsChromaticityCoordinates.get(right)[0] - wavelengthsChromaticityCoordinates.get(left)[0]
		dy = wavelengthsChromaticityCoordinates.get(right)[1] - wavelengthsChromaticityCoordinates.get(left)[1]

		normalize = lambda x: x / numpy.linalg.norm(x)

		xy = numpy.array([x, y])
		direction = numpy.array((-dy, dx))

		normal = numpy.array((-dy, dx)) if numpy.dot(normalize(xy - equalEnergy),
													 normalize(direction)) > 0 else numpy.array((dy, -dx))
		normal = normalize(normal)
		normal /= 25

		pylab.plot([x, x + normal[0] * 0.75], [y, y + normal[1] * 0.75], color="black", linewidth=1.5)
		pylab.text(x + normal[0], y + normal[1], label, ha="left" if normal[0] >= 0 else "right", va="center",
				   fontdict={"size": "small"})

	settings = {"title": "CIE 1931 Chromaticity Diagram - {0}".format(name),
				"xLabel": "CIE x",
				"yLabel": "CIE y",
				"tickerX": True,
				"tickerY": True,
				"grid": True,
				"boundingBox": [-0.1, 0.9, -0.1, 0.9],
				"bbox_inches": "tight",
				"pad_inches": 0}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

@figureSize((8, 8))
def colorspaces_CIE_1931_chromaticityDiagramPlot(colorspaces=["sRGB", "ACES RGB", "Pointer Gamut"],
												 cmfs="Standard CIE 1931 2 Degree Observer",
												 **kwargs):
	"""
	Plots given colorspaces in *CIE 1931 Chromaticity Diagram*.

	Usage::

		>>> colorspaces_CIE_1931_chromaticityDiagramPlot(["sRGB", "ACES RGB"])
		True

	:param colorspaces: Colorspaces to plot.
	:type colorspaces: list
	:param cmfs: Standard observer color matching functions used for diagram bounds.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))
	settings = {"title": "{0} - {1}".format(", ".join(colorspaces), name),
				"standalone": False}
	settings.update(kwargs)

	if not CIE_1931_chromaticityDiagramPlot(**settings):
		return

	xLimitMin, xLimitMax, yLimitMin, yLimitMax = [-0.1], [0.9], [-0.1], [0.9]
	for colorspace in colorspaces:
		if colorspace == "Pointer Gamut":
			x, y = zip(*color.colorspaces.POINTER_GAMUT_DATA)
			pylab.plot(x, y, label="Pointer Gamut", color="0.95", linewidth=2.)
			pylab.plot([x[-1], x[0]], [y[-1], y[0]], color="0.95", linewidth=2.)
		else:
			colorspace, name = color.colorspaces.COLORSPACES.get(colorspace), colorspace
			if colorspace is None:
				raise color.exceptions.ProgrammingError(
					"'{0}' colorspace not found in supported colorspaces: '{1}'.".format(name,
																						 sorted(
																							 color.colorspaces.COLORSPACES.keys())))

			randomColor = lambda: float(random.randint(64, 224)) / 255
			r, g, b = randomColor(), randomColor(), randomColor()

			primaries = colorspace.primaries
			whitepoint = colorspace.whitepoint

			pylab.plot([whitepoint[0], whitepoint[0]], [whitepoint[1], whitepoint[1]], color=(r, g, b),
					   label=colorspace.name, linewidth=2.)
			pylab.plot([whitepoint[0], whitepoint[0]], [whitepoint[1], whitepoint[1]], "o", color=(r, g, b),
					   linewidth=2.)
			pylab.plot([primaries[0, 0], primaries[1, 0]], [primaries[0, 1], primaries[1, 1]], "o-", color=(r, g, b),
					   linewidth=2.)
			pylab.plot([primaries[1, 0], primaries[2, 0]], [primaries[1, 1], primaries[2, 1]], "o-", color=(r, g, b),
					   linewidth=2.)
			pylab.plot([primaries[2, 0], primaries[0, 0]], [primaries[2, 1], primaries[0, 1]], "o-", color=(r, g, b),
					   linewidth=2.)

			xLimitMin.append(numpy.amin(primaries[:, 0]))
			yLimitMin.append(numpy.amin(primaries[:, 1]))
			xLimitMax.append(numpy.amax(primaries[:, 0]))
			yLimitMax.append(numpy.amax(primaries[:, 1]))

	settings.update({"legend": True,
					 "legendLocation": "upper right",
					 "tightenX": True,
					 "tightenY": True,
					 "limits": [min(xLimitMin), max(xLimitMax), min(yLimitMin), max(yLimitMax)],
					 "margins": [-0.05, 0.05, -0.05, 0.05],
					 "standalone": True})

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

@figureSize((8, 8))
def planckianLocus_CIE_1931_chromaticityDiagramPlot(illuminants=["A", "C", "E"],
													**kwargs):
	"""
	Plots the planckian locus and given illuminants in *CIE 1931 Chromaticity Diagram*.

	Usage::

		>>> planckianLocus_CIE_1931_chromaticityDiagramPlot(["A", "C", "E"])
		True

	:param illuminants: Factory illuminants to plot.
	:type illuminants: tuple or list
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")

	settings = {
	"title": "{0} Illuminants - Planckian Locus\n CIE 1931 Chromaticity Diagram - Standard CIE 1931 2 Degree Observer".format(
		", ".join(
			illuminants)) if illuminants else "Planckian Locus\n CIE 1931 Chromaticity Diagram - Standard CIE 1931 2 Degree Observer",
	"standalone": False}
	settings.update(kwargs)

	if not CIE_1931_chromaticityDiagramPlot(**settings):
		return

	start, end = 1667, 100000
	x, y = zip(*map(lambda x: color.transformations.UVW_uv_to_xy(color.temperature.cct_to_uv(x, 0., cmfs)),
					numpy.arange(start, end + 250, 250)))

	pylab.plot(x, y, color="black", linewidth=2.)

	for i in [1667, 2000, 2500, 3000, 4000, 6000, 10000]:
		x0, y0 = color.transformations.UVW_uv_to_xy(color.temperature.cct_to_uv(i, -0.025, cmfs))
		x1, y1 = color.transformations.UVW_uv_to_xy(color.temperature.cct_to_uv(i, 0.025, cmfs))
		pylab.plot([x0, x1], [y0, y1], color="black", linewidth=2.)
		pylab.annotate("{0}K".format(i),
					   xy=(x0, y0),
					   xytext=(0, -10),
					   textcoords="offset points",
					   size="x-small")

	for illuminant in illuminants:
		xy = color.illuminants.ILLUMINANTS.get(cmfs.name).get(illuminant)
		if xy is None:
			raise color.exceptions.ProgrammingError(
				"Illuminant '{0}' not found in factory illuminants: '{1}'.".format(illuminant,
																				   sorted(
																					   color.illuminants.ILLUMINANTS.get(
																						   cmfs.name).keys())))

		pylab.plot(xy[0], xy[1], "o", color="white", linewidth=2.)

		pylab.annotate(illuminant,
					   xy=(xy[0], xy[1]),
					   xytext=(-50, 30),
					   textcoords="offset points",
					   arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.2"))

	settings.update({"standalone": True})

	return display(**settings)

@figureSize((32, 32))
def CIE_1960_UCS_chromaticityDiagramColorsPlot(surface=1.25,
											   spacing=0.00075,
											   cmfs="Standard CIE 1931 2 Degree Observer",
											   **kwargs):
	"""
	Plots the *CIE 1960 UCS Chromaticity Diagram* colors.

	Usage::

		>>> CIE_1960_UCS_chromaticityDiagramColorsPlot()
		True

	:param surface: Generated markers surface.
	:type surface: float
	:param spacing: Spacing between markers.
	:type spacing: float
	:param cmfs: Standard observer color matching functions used for diagram bounds.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	illuminant = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("E")

	UVWs = [color.transformations.XYZ_to_UVW(value) for key, value in cmfs]

	u, v = zip(*(map(lambda x: color.transformations.UVW_to_uv(x), UVWs)))

	path = matplotlib.path.Path(zip(u, v))
	dotX, dotY, colors = [], [], []
	for i in numpy.arange(0., 1., spacing):
		for j in numpy.arange(0., 1., spacing):
			if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
				dotX.append(i)
				dotY.append(j)

				XYZ = color.transformations.xy_to_XYZ(color.transformations.UVW_uv_to_xy((i, j)))
				RGB = XYZ_to_sRGB(XYZ, illuminant)

				RGB = numpy.ravel(RGB)
				RGB *= 1. / numpy.max(RGB)
				RGB = numpy.clip(RGB, 0, 1)

				colors.append(RGB)

	pylab.scatter(dotX, dotY, color=colors, s=surface)

	settings = {"noTicks": True,
				"boundingBox": [0., 1., 0., 1.],
				"bbox_inches": "tight",
				"pad_inches": 0}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

@figureSize((8, 8))
def CIE_1960_UCS_chromaticityDiagramPlot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
	"""
	Plots the *CIE 1960 UCS Chromaticity Diagram*.

	Usage::

		>>> CIE_1960_UCS_chromaticityDiagramPlot()
		True

	:param cmfs: Standard observer color matching functions used for diagram bounds.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	image = matplotlib.image.imread(os.path.join(RESOURCES_DIRECTORY,
												 "CIE_1960_UCS_Chromaticity_Diagram_{0}_Small.png".format(
													 cmfs.name.replace(" ", "_"))))
	pylab.imshow(image, interpolation="nearest", extent=(0, 1, 0, 1))

	labels = [420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620,
			  630, 640, 680]

	wavelengths = cmfs.wavelengths
	equalEnergy = numpy.array([1. / 3.] * 2)

	UVWs = [color.transformations.XYZ_to_UVW(value) for key, value in cmfs]

	u, v = zip(*(map(lambda x: color.transformations.UVW_to_uv(x), UVWs)))

	wavelengthsChromaticityCoordinates = dict(zip(wavelengths, zip(u, v)))

	pylab.plot(u, v, color="black", linewidth=2.)
	pylab.plot((u[-1], u[0]), (v[-1], v[0]), color="black", linewidth=2.)

	for label in labels:
		u, v = wavelengthsChromaticityCoordinates.get(label)
		pylab.plot(u, v, "o", color="black", linewidth=2.)

		index = bisect.bisect(wavelengths, label)
		left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
		right = wavelengths[index] if index < len(wavelengths) else wavelengths[-1]

		dx = wavelengthsChromaticityCoordinates.get(right)[0] - wavelengthsChromaticityCoordinates.get(left)[0]
		dy = wavelengthsChromaticityCoordinates.get(right)[1] - wavelengthsChromaticityCoordinates.get(left)[1]

		normalize = lambda x: x / numpy.linalg.norm(x)

		uv = numpy.array([u, v])
		direction = numpy.array((-dy, dx))

		normal = numpy.array((-dy, dx)) if numpy.dot(normalize(uv - equalEnergy),
													 normalize(direction)) > 0 else numpy.array((dy, -dx))
		normal = normalize(normal)
		normal /= 25

		pylab.plot([u, u + normal[0] * 0.75], [v, v + normal[1] * 0.75], color="black", linewidth=1.5)
		pylab.text(u + normal[0], v + normal[1], label, ha="left" if normal[0] >= 0 else "right", va="center",
				   fontdict={"size": "small"})

	settings = {"title": "CIE 1960 UCS Chromaticity Diagram - {0}".format(name),
				"xLabel": "CIE u",
				"yLabel": "CIE v",
				"tickerX": True,
				"tickerY": True,
				"grid": True,
				"boundingBox": [-0.075, 0.675, -0.15, 0.6],
				"bbox_inches": "tight",
				"pad_inches": 0}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

@figureSize((8, 8))
def planckianLocus_CIE_1960_UCS_chromaticityDiagramPlot(illuminants=["A", "C", "E"],
														**kwargs):
	"""
	Plots the planckian locus and given illuminants in *CIE 1960 UCS Chromaticity Diagram*.

	Usage::

		>>> planckianLocus_CIE_1960_UCS_chromaticityDiagramPlot(["A", "C", "E"])
		True

	:param illuminants: Factory illuminants to plot.
	:type illuminants: tuple or list
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")

	settings = {
	"title": "{0} Illuminants - Planckian Locus\nCIE 1960 UCS Chromaticity Diagram - Standard CIE 1931 2 Degree Observer".format(
		", ".join(
			illuminants)) if illuminants else "Planckian Locus\nCIE 1960 UCS Chromaticity Diagram - Standard CIE 1931 2 Degree Observer",
	"standalone": False}
	settings.update(kwargs)

	if not CIE_1960_UCS_chromaticityDiagramPlot(**settings):
		return

	xy_to_uv = lambda x: color.transformations.UVW_to_uv(
		color.transformations.XYZ_to_UVW(
			color.transformations.xy_to_XYZ(x)))

	start, end = 1667, 100000
	u, v = zip(*map(lambda x: color.temperature.cct_to_uv(x, 0., cmfs),	numpy.arange(start, end + 250, 250)))

	pylab.plot(u, v, color="black", linewidth=2.)

	for i in [1667, 2000, 2500, 3000, 4000, 6000, 10000]:
		u0, v0 = color.temperature.cct_to_uv(i, -0.05)
		u1, v1 = color.temperature.cct_to_uv(i, 0.05)
		pylab.plot([u0, u1], [v0, v1], color="black", linewidth=2.)
		pylab.annotate("{0}K".format(i),
					   xy=(u0, v0),
					   xytext=(0, -10),
					   textcoords="offset points",
					   size="x-small")

	for illuminant in illuminants:
		uv = xy_to_uv(color.illuminants.ILLUMINANTS.get(cmfs.name).get(illuminant))
		if uv is None:
			raise color.exceptions.ProgrammingError(
				"Illuminant '{0}' not found in factory illuminants: '{1}'.".format(illuminant,
																				   sorted(
																					   color.illuminants.ILLUMINANTS.get(
																						   cmfs.name).keys())))

		pylab.plot(uv[0], uv[1], "o", color="white", linewidth=2.)

		pylab.annotate(illuminant,
					   xy=(uv[0], uv[1]),
					   xytext=(-50, 30),
					   textcoords="offset points",
					   arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.2"))

	settings.update({"standalone": True})

	return display(**settings)

@figureSize((32, 32))
def CIE_1976_UCS_chromaticityDiagramColorsPlot(surface=1.25,
											   spacing=0.00075,
											   cmfs="Standard CIE 1931 2 Degree Observer",
											   **kwargs):
	"""
	Plots the *CIE 1976 UCS Chromaticity Diagram* colors.

	Usage::

		>>> CIE_1976_UCS_chromaticityDiagramColorsPlot()
		True

	:param surface: Generated markers surface.
	:type surface: float
	:param spacing: Spacing between markers.
	:type spacing: float
	:param cmfs: Standard observer color matching functions used for diagram bounds.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	illuminant = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

	Luvs = [color.transformations.XYZ_to_Luv(value, illuminant) for key, value in cmfs]

	u, v = zip(*(map(lambda x: color.transformations.Luv_to_uv(x), Luvs)))

	path = matplotlib.path.Path(zip(u, v))
	dotX, dotY, colors = [], [], []
	for i in numpy.arange(0., 1., spacing):
		for j in numpy.arange(0., 1., spacing):
			if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
				dotX.append(i)
				dotY.append(j)

				XYZ = color.transformations.xy_to_XYZ(color.transformations.Luv_uv_to_xy((i, j)))
				RGB = XYZ_to_sRGB(XYZ, illuminant)

				RGB = numpy.ravel(RGB)
				RGB *= 1. / numpy.max(RGB)
				RGB = numpy.clip(RGB, 0, 1)

				colors.append(RGB)

	pylab.scatter(dotX, dotY, color=colors, s=surface)

	settings = {"noTicks": True,
				"boundingBox": [0., 1., 0., 1.],
				"bbox_inches": "tight",
				"pad_inches": 0}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

@figureSize((8, 8))
def CIE_1976_UCS_chromaticityDiagramPlot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
	"""
	Plots the *CIE 1976 UCS Chromaticity Diagram*.

	Usage::

		>>> CIE_1976_UCS_chromaticityDiagramPlot()
		True

	:param cmfs: Standard observer color matching functions used for diagram bounds.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	image = matplotlib.image.imread(os.path.join(RESOURCES_DIRECTORY,
												 "CIE_1976_UCS_Chromaticity_Diagram_{0}_Small.png".format(
													 cmfs.name.replace(" ", "_"))))
	pylab.imshow(image, interpolation="nearest", extent=(0, 1, 0, 1))

	labels = [420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620,
			  630, 640, 680]

	wavelengths = cmfs.wavelengths
	equalEnergy = numpy.array([1. / 3.] * 2)

	illuminant = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

	Luvs = [color.transformations.XYZ_to_Luv(value, illuminant) for key, value in cmfs]

	u, v = zip(*(map(lambda x: color.transformations.Luv_to_uv(x), Luvs)))

	wavelengthsChromaticityCoordinates = dict(zip(wavelengths, zip(u, v)))

	pylab.plot(u, v, color="black", linewidth=2.)
	pylab.plot((u[-1], u[0]), (v[-1], v[0]), color="black", linewidth=2.)

	for label in labels:
		u, v = wavelengthsChromaticityCoordinates.get(label)
		pylab.plot(u, v, "o", color="black", linewidth=2.)

		index = bisect.bisect(wavelengths, label)
		left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
		right = wavelengths[index] if index < len(wavelengths) else wavelengths[-1]

		dx = wavelengthsChromaticityCoordinates.get(right)[0] - wavelengthsChromaticityCoordinates.get(left)[0]
		dy = wavelengthsChromaticityCoordinates.get(right)[1] - wavelengthsChromaticityCoordinates.get(left)[1]

		normalize = lambda x: x / numpy.linalg.norm(x)

		uv = numpy.array([u, v])
		direction = numpy.array((-dy, dx))

		normal = numpy.array((-dy, dx)) if numpy.dot(normalize(uv - equalEnergy),
													 normalize(direction)) > 0 else numpy.array((dy, -dx))
		normal = normalize(normal)
		normal /= 25

		pylab.plot([u, u + normal[0] * 0.75], [v, v + normal[1] * 0.75], color="black", linewidth=1.5)
		pylab.text(u + normal[0], v + normal[1], label, ha="left" if normal[0] >= 0 else "right", va="center",
				   fontdict={"size": "small"})

	settings = {"title": "CIE 1976 UCS Chromaticity Diagram - {0}".format(name),
				"xLabel": "CIE u'",
				"yLabel": "CIE v'",
				"tickerX": True,
				"tickerY": True,
				"grid": True,
				"boundingBox": [-0.1, .7, -.1, .7],
				"bbox_inches": "tight",
				"pad_inches": 0}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

def singleTransferFunctionPlot(colorspace="sRGB",
							   **kwargs):
	"""
	Plots given colorspace transfer function.

	Usage::

		>>> singleTransferFunctionPlot("sRGB")
		True

	:param colorspace: Colorspace transfer function to plot.
	:type colorspace: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	settings = {"title": "{0} - Transfer Function".format(colorspace)}
	settings.update(kwargs)

	return multiTransferFunctionPlot([colorspace], **settings)

@figureSize((8, 8))
def multiTransferFunctionPlot(colorspaces=["sRGB", "Rec. 709"],
							  **kwargs):
	"""
	Plots given colorspaces transfer functions.

	Usage::

		>>> multiTransferFunctionPlot(["sRGB", "Rec. 709"])
		True

	:param colorspaces: Colorspaces transfer functions to plot.
	:type colorspaces: list
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	samples = numpy.linspace(0., 1., 100)
	for i, colorspace in enumerate(colorspaces):
		colorspace, name = color.colorspaces.COLORSPACES.get(colorspace), colorspace
		if colorspace is None:
			raise color.exceptions.ProgrammingError(
				"'{0}' colorspace not found in supported colorspaces: '{1}'.".format(name,
																					 sorted(
																						 color.colorspaces.COLORSPACES.keys())))

		RGBs = numpy.array(map(colorspace.inverseTransferFunction, zip(samples, samples, samples)))
		for j, data in enumerate((("R", [1., 0., 0.]),
								  ("G", [0., 1., 0.]),
								  ("B", [0., 0., 1.]))):
			axis, rgb = data
			rgb = map(lambda x: reduce(lambda y, _: y * 0.25, xrange(i), x), rgb)
			pylab.plot(samples, RGBs[:, j], color=rgb,
					   label=u"{0} - {1}".format(axis, colorspace.name),
					   linewidth=2.)

	settings = {"title": "{0} - Transfer Functions".format(", ".join(colorspaces)),
				"tightenX": True,
				"legend": True,
				"legendLocation": "upper left",
				"tickerX": True,
				"tickerY": True,
				"grid": True,
				"limits": [0., 1., 0., 1.]}

	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)

	return display(**settings)

def blackbodySpectralRadiancePlot(temperature=3500,
								  cmfs="Standard CIE 1931 2 Degree Observer",
								  blackbody="VY Canis Major",
								  **kwargs):
	"""
	Plots given blackbody spectral radiance.

	Usage::

		>>> blackbodySpectralRadiancePlot(3500)
		True

	:param temperature: Blackbody temperature.
	:type temperature: float
	:param cmfs: Standard observer color matching functions.
	:type cmfs: unicode
	:param blackbody: Blackbody name.
	:type blackbody: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	matplotlib.pyplot.subplots_adjust(hspace=0.4)

	spd = color.blackbody.blackbodySpectralPowerDistribution(temperature, *cmfs.shape)

	matplotlib.pyplot.figure(1)
	matplotlib.pyplot.subplot(211)

	settings = {"title": "{0} - Spectral Radiance".format(blackbody),
				"yLabel": u"W / (sr m²) / m",
				"standalone": False}
	settings.update(kwargs)

	singleSpectralPowerDistributionPlot(spd, cmfs, **settings)

	XYZ = color.transformations.spectral_to_XYZ(spd, cmfs)
	RGB = XYZ_to_sRGB(XYZ)
	RGB *= 1. / numpy.max(RGB)

	matplotlib.pyplot.subplot(212)

	settings = {"title": "{0} - Color".format(blackbody),
				"xLabel": "{0}K".format(temperature),
				"yLabel": "",
				"aspect": None,
				"standalone": False}

	singleColorPlot(colorParameter(name="", RGB=RGB), **settings)

	settings = {"standalone": True}
	settings.update(kwargs)

	boundingBox(**settings)
	aspect(**settings)
	return display(**settings)

def blackbodyColorsPlot(start=1000,
						end=15000,
						steps=25,
						cmfs="Standard CIE 1931 2 Degree Observer",
						**kwargs):
	"""
	Plots blackbody colors.

	Usage::

		>>> blackbodyColorsPlot()
		True

	:param start: Temperature range start in kelvins.
	:type start: float
	:param end: Temperature range end in kelvins.
	:type end: float
	:param steps: Temperature range steps.
	:type steps: float
	:param cmfs: Standard observer color matching functions.
	:type cmfs: unicode
	:param \*\*kwargs: Keywords arguments.
	:type \*\*kwargs: \*\*
	:return: Definition success.
	:rtype: bool
	"""

	cmfs, name = color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(cmfs), cmfs
	if cmfs is None:
		raise color.exceptions.ProgrammingError(
			"Standard observer '{0}' not found in standard observers color matching functions: '{1}'.".format(name,
																											  sorted(
																												  color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.keys())))

	colors = []
	temperatures = []

	for temperature in numpy.arange(start, end + steps, steps):
		spd = color.blackbody.blackbodySpectralPowerDistribution(temperature, *cmfs.shape)

		XYZ = color.transformations.spectral_to_XYZ(spd, cmfs)
		RGB = XYZ_to_sRGB(XYZ)

		RGB = numpy.ravel(RGB)
		RGB *= 1. / numpy.max(RGB)
		RGB = numpy.clip(RGB, 0, 1)

		colors.append(RGB)
		temperatures.append(temperature)

	settings = {"title": "Blackbody Colors",
				"xLabel": "Temperature K",
				"yLabel": "",
				"tightenX": True,
				"tickerX": True,
				"tickerY": False}

	settings.update(kwargs)
	return colorParametersPlot(map(lambda x: colorParameter(x=x[0], RGB=x[1]), zip(temperatures, colors)),
							   **settings)