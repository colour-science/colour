#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**plots.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package **matplotlib** plotting objects.

**Others:**

"""

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

import color.algebra.matrix
import color.blackbody
import color.color_checkers
import color.colorspaces
import color.illuminants
import color.exceptions
import color.lightness
import color.spectral
import color.temperature
import color.transformations
import color.data_structures
import color.verbose

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
           "figure_size",
           "aspect",
           "bounding_box",
           "display",
           "color_parameter",
           "color_parameters_plot",
           "single_color_plot",
           "multi_color_plot",
           "color_checker_plot",
           "single_spectral_power_distribution_plot",
           "multi_spectral_power_distribution_plot",
           "single_color_matching_functions_plot",
           "multi_color_matching_functions_plot",
           "single_illuminant_relative_spd_plot",
           "multi_illuminants_relative_spd_plot",
           "visible_spectrum_plot",
           "CIE_1931_chromaticity_diagram_colors_plot",
           "CIE_1931_chromaticity_diagram_plot",
           "colorspaces_CIE_1931_chromaticity_diagram_plot",
           "planckian_locus_CIE_1931_chromaticity_diagram_plot",
           "CIE_1960_UCS_chromaticity_diagram_colors_plot",
           "CIE_1960_UCS_chromaticity_diagram_plot",
           "planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot",
           "CIE_1976_UCS_chromaticity_diagram_colors_plot",
           "CIE_1976_UCS_chromaticity_diagram_plot",
           "single_munsell_value_function_plot",
           "multi_munsell_value_function_plot",
           "single_lightness_function_plot",
           "multi_lightness_function_plot",
           "single_transfer_function_plot",
           "multi_transfer_function_plot",
           "blackbody_spectral_radiance_plot",
           "blackbody_colors_plot"]

LOGGER = color.verbose.install_logger()

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")

DEFAULT_FIGURE_SIZE = 14, 7

COLOR_PARAMETER = namedtuple("ColorParameter", ("name", "RGB", "x", "y0", "y1"))

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
                                            color.colorspaces.sRGB_COLORSPACE.from_XYZ,
                                            color.colorspaces.sRGB_COLORSPACE.transfer_function)


def figure_size(size=DEFAULT_FIGURE_SIZE):
    """
    Sets figures sizes.

    :param size: Figure size.
    :type size: tuple
    :return: Object.
    :rtype: object
    """

    def figure_size_decorator(object):
        """
        Sets figures sizes.

        :param object: Object to decorate.
        :type object: object
        :return: Object.
        :rtype: object
        """

        @functools.wraps(object)
        def figure_size_wrapper(*args, **kwargs):
            """
            Sets figures sizes.

            :param \*args: Arguments.
            :type \*args: \*
            :param \*\*kwargs: Keywords arguments.
            :type \*\*kwargs: \*\*
            :return: Object.
            :rtype: object
            """

            pylab.rcParams["figure.figsize"] = kwargs.get("figure_size") if kwargs.get(
                "figure_size") is not None else size

            try:
                return object(*args, **kwargs)
            finally:
                pylab.rcParams["figure.figsize"] = DEFAULT_FIGURE_SIZE

        return figure_size_wrapper

    return figure_size_decorator


def aspect(**kwargs):
    """
    Sets the figure aspect.

    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    settings = color.data_structures.Structure(**{"title": None,
                                                  "x_label": None,
                                                  "y_label": None,
                                                  "legend": False,
                                                  "legend_location": "upper right",
                                                  "x_ticker": False,
                                                  "y_ticker": False,
                                                  "x_ticker_locator": matplotlib.ticker.AutoMinorLocator(2),
                                                  "y_ticker_locator": matplotlib.ticker.AutoMinorLocator(2),
                                                  "no_ticks": False,
                                                  "grid": False,
                                                  "axis_grid": "both",
                                                  "x_axis_line": False,
                                                  "y_axis_line": False,
                                                  "aspect": None})
    settings.update(kwargs)

    settings.title and pylab.title(settings.title)
    settings.x_label and pylab.xlabel(settings.x_label)
    settings.y_label and pylab.ylabel(settings.y_label)
    settings.legend and pylab.legend(loc=settings.legend_location)
    settings.x_ticker and matplotlib.pyplot.gca().xaxis.set_minor_locator(settings.x_ticker_locator)
    settings.y_ticker and matplotlib.pyplot.gca().yaxis.set_minor_locator(settings.y_ticker_locator)
    if settings.no_ticks:
        matplotlib.pyplot.gca().set_xticks([])
        matplotlib.pyplot.gca().set_yticks([])
    settings.grid and pylab.grid(which=settings.axis_grid)
    settings.x_axis_line and pylab.axvline(0, color="black", linestyle="--")
    settings.y_axis_line and pylab.axhline(0, color="black", linestyle="--")
    settings.aspect and matplotlib.pyplot.axes().set_aspect(settings.aspect)

    return True


def bounding_box(**kwargs):
    """
    Sets the plot bounding box.

    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    settings = color.data_structures.Structure(**{"bounding_box": None,
                                                  "x_tighten": False,
                                                  "y_tighten": False,
                                                  "limits": [0., 1., 0., 1.],
                                                  "margins": [0., 0., 0., 0.]})
    settings.update(kwargs)

    if settings.bounding_box is None:
        x_limit_min, x_limit_max, y_limit_min, y_limit_max = settings.limits
        x_margin_min, x_margin_max, y_margin_min, y_margin_max = settings.margins
        settings.x_tighten and pylab.xlim(x_limit_min + x_margin_min, x_limit_max + x_margin_max)
        settings.y_tighten and pylab.ylim(y_limit_min + y_margin_min, y_limit_max + y_margin_max)
    else:
        pylab.xlim(settings.bounding_box[0], settings.bounding_box[1])
        pylab.ylim(settings.bounding_box[2], settings.bounding_box[3])

    return True


def display(**kwargs):
    """
    Sets the figure display.

    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    settings = color.data_structures.Structure(**{"standalone": True,
                                                  "filename": None})
    settings.update(kwargs)

    if settings.standalone:
        if settings.filename is not None:
            pylab.savefig(**kwargs)
        else:
            pylab.show()
        pylab.close()

    return True


def color_parameter(name=None, RGB=None, x=None, y0=None, y1=None):
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


def color_parameters_plot(color_parameters,
                          y0_plot=True,
                          y1_plot=True,
                          **kwargs):
    """
    Plots given color color_parameters.

    Usage::

        >>> cp1 = color_parameter(x=390, RGB=[0.03009021, 0., 0.12300545])
        >>> cp2 = color_parameter(x=391, RGB=[0.03434063, 0., 0.13328537], y0=0, y1=0.25)
        >>> cp3 = color_parameter(x=392, RGB=[0.03826312, 0., 0.14276247], y0=0, y1=0.35)
        >>> cp4 = color_parameter(x=393, RGB=[0.04191844, 0., 0.15158707], y0=0, y1=0.05)
        >>> cp5 = color_parameter(x=394, RGB=[0.04535085, 0., 0.15986838], y0=0, y1=-.25)
        >>> color_parameters_plot([cp1, cp2, cp3, cp3, cp4, cp5])
        True

    :param color_parameters: ColorParameter sequence.
    :type color_parameters: list
    :param y0_plot: Plot y0 line.
    :type y0_plot: bool
    :param y1_plot: Plot y1 line.
    :type y1_plot: bool
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    for i in xrange(len(color_parameters) - 1):
        x0 = color_parameters[i].x
        x01 = color_parameters[i + 1].x
        y0 = 0. if color_parameters[i].y0 is None else color_parameters[i].y0
        y1 = 1. if color_parameters[i].y1 is None else color_parameters[i].y1
        y01 = 0. if color_parameters[i].y0 is None else color_parameters[i + 1].y0
        y11 = 1. if color_parameters[i].y1 is None else color_parameters[i + 1].y1
        x_polygon = [x0, x01, x01, x0]
        y_polygon = [y0, y01, y11, y1]
        pylab.fill(x_polygon, y_polygon, color=color_parameters[i].RGB, edgecolor=color_parameters[i].RGB)

    if all(map(lambda x: x.y0 is not None, color_parameters)):
        y0_plot and pylab.plot(map(lambda x: x.x, color_parameters), map(lambda x: x.y0, color_parameters),
                               color="black",
                               linewidth=2.)

    if all(map(lambda x: x.y1 is not None, color_parameters)):
        y1_plot and pylab.plot(map(lambda x: x.x, color_parameters), map(lambda x: x.y1, color_parameters),
                               color="black",
                               linewidth=2.)

    y_limit_min0, y_limit_max0 = min(map(lambda x: 0. if x.y0 is None else x.y0, color_parameters)), \
                                 max(map(lambda x: 1. if x.y0 is None else x.y0, color_parameters))
    y_limit_min1, y_limit_max1 = min(map(lambda x: 0. if x.y1 is None else x.y1, color_parameters)), \
                                 max(map(lambda x: 1. if x.y1 is None else x.y1, color_parameters))

    settings = {"x_label": "Parameter",
                "y_label": "Color",
                "limits": [min(map(lambda x: 0. if x.x is None else x.x, color_parameters)),
                           max(map(lambda x: 1. if x.x is None else x.x, color_parameters)),
                           y_limit_min0,
                           y_limit_max1]}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def single_color_plot(color_parameter,
                      **kwargs):
    """
    Plots given color.

    Usage::

        >>> single_color_plot(color_parameter(RGB=(0.32315746, 0.32983556, 0.33640183)))
        True

    :param color_parameter: ColorParameter.
    :type color_parameter: ColorParameter
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    return multi_color_plot([color_parameter], **kwargs)


def multi_color_plot(color_parameters,
                     width=1.,
                     height=1.,
                     spacing=0.,
                     across=3,
                     text_display=True,
                     text_size="large",
                     text_offset=0.075,
                     **kwargs):
    """
    Plots given colors.

    Usage::

        >>> cp1 = color_parameter(RGB=(0.45293517, 0.31732158, 0.26414773))
        >>> cp2 = color_parameter(RGB=(0.77875824, 0.5772645,  0.50453169)
        >>> multi_color_plot([cp1, cp2])
        True

    :param color_parameters: ColorParameter sequence.
    :type color_parameters: list
    :param width: Color polygon width.
    :type width: float
    :param height: Color polygon height.
    :type height: float
    :param spacing: Color polygons spacing.
    :type spacing: float
    :param across: Color polygons count per row.
    :type across: int
    :param text_display: Display color text.
    :type text_display: bool
    :param text_size: Color text size.
    :type text_size: float
    :param text_offset: Color text offset.
    :type text_offset: float
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    offsetX = offsetY = 0
    x_limit_min, x_limit_max, y_limit_min, y_limit_max = 0, width, 0, height
    for i, color_parameter in enumerate(color_parameters):
        if i % across == 0 and i != 0:
            offsetX = 0
            offsetY -= height + spacing

        x0 = offsetX
        x1 = offsetX + width
        y0 = offsetY
        y1 = offsetY + height

        x_polygon = [x0, x1, x1, x0]
        y_polygon = [y0, y0, y1, y1]
        pylab.fill(x_polygon, y_polygon, color=color_parameters[i].RGB)
        if color_parameter.name is not None and text_display:
            pylab.text(x0 + text_offset, y0 + text_offset, color_parameter.name, size=text_size)

        offsetX += width + spacing

    x_limit_max = min(len(color_parameters), across)
    x_limit_max = x_limit_max * width + x_limit_max * spacing - spacing
    y_limit_min = offsetY

    settings = {"x_tighten": True,
                "y_tighten": True,
                "no_ticks": True,
                "limits": [x_limit_min, x_limit_max, y_limit_min, y_limit_max],
                "aspect": "equal"}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def color_checker_plot(color_checker="ColorChecker 2005",
                       **kwargs):
    """
    Plots given color checker.

    Usage::

        >>> color_checker_plot()
        True

    :param color_checker: Color checker name.
    :type color_checker: unicode
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    color_checker, name = color.color_checkers.COLORCHECKERS.get(color_checker), color_checker
    if color_checker is None:
        raise color.exceptions.ProgrammingError(
            "Color checker '{0}' not found in color checkers: '{1}'.".format(name,
                                                                             sorted(
                                                                                 color.color_checkers.COLORCHECKERS.keys())))

    _, data, illuminant = color_checker
    color_parameters = []
    for index, label, x, y, Y in data:
        XYZ = color.transformations.xyY_to_XYZ((x, y, Y))
        RGB = XYZ_to_sRGB(XYZ, illuminant)

        color_parameters.append(color_parameter(label.title(), numpy.clip(numpy.ravel(RGB), 0, 1)))

    background_color = "0.1"
    matplotlib.pyplot.gca().patch.set_facecolor(background_color)

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

    multi_color_plot(color_parameters, **settings)

    text_x = width * (across / 2) + (across * (spacing / 2)) - spacing / 2
    text_y = -(len(color_parameters) / across + spacing / 2)

    pylab.text(text_x,
               text_y,
               "{0} - {1} - Color Rendition Chart".format(name, color.colorspaces.sRGB_COLORSPACE.name),
               color="0.95",
               ha="center")

    settings.update({"title": name,
                     "facecolor": background_color,
                     "edgecolor": None,
                     "standalone": True})

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def single_spectral_power_distribution_plot(spd,
                                            cmfs=color.spectral.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get(
                                                "Standard CIE 1931 2 Degree Observer"),
                                            **kwargs):
    """
    Plots given spectral power distribution.

    Usage::

        >>> spd = color.SpectralPowerDistribution(name="Custom", spd={400: 0.0641, 420: 0.0645, 440: 0.0562})
        >>> single_spectral_power_distribution_plot(spd)
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
                "x_label": u"Wavelength λ (nm)",
                "y_label": "Spectral Power Distribution",
                "x_tighten": True,
                "x_ticker": True,
                "y_ticker": True}

    settings.update(kwargs)
    return color_parameters_plot(
        map(lambda x: color_parameter(x=x[0], y1=x[1], RGB=x[2]), zip(wavelengths, y1, colors)),
        **settings)


def multi_spectral_power_distribution_plot(spds,
                                           **kwargs):
    """
    Plots given spectral power distributions.

    Usage::

        >>> spd1 = color.SpectralPowerDistribution(name="Custom1", spd={400: 0.0641, 420: 0.0645, 440: 0.0562})
        >>> spd2 = color.SpectralPowerDistribution(name="Custom2", spd={400: 0.134, 420: 0.789, 440: 1.289})
        >>> multi_spectral_power_distribution_plot([spd1, spd2]))
        True

    :param spds: Spectral power distributions to plot.
    :type spds: list
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for spd in spds:
        wavelengths, values = zip(*[(key, value) for key, value in spd])

        start, end, steps = spd.shape
        x_limit_min.append(start)
        x_limit_max.append(end)
        y_limit_min.append(min(values))
        y_limit_max.append(max(values))

        pylab.plot(wavelengths, values, label=spd.name, linewidth=2.)

    settings = {"x_label": u"Wavelength λ (nm)",
                "y_label": "Spectral Power Distribution",
                "x_tighten": True,
                "legend": True,
                "legend_location": "upper left",
                "x_ticker": True,
                "y_ticker": True,
                "limits": [min(x_limit_min), max(x_limit_max), min(y_limit_min), max(y_limit_max)]}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def single_color_matching_functions_plot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
    """
    Plots given standard observer *CIE XYZ* color matching functions.

    Usage::

        >>> single_color_matching_functions_plot("Standard CIE 1931 2 Degree Observer")
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

    return multi_color_matching_functions_plot([cmfs], **settings)


def multi_color_matching_functions_plot(cmfss=["Standard CIE 1931 2 Degree Observer",
                                               "Standard CIE 1964 10 Degree Observer"],
                                        **kwargs):
    """
    Plots given standard observers *CIE XYZ* color matching functions.

    Usage::

        >>> multi_color_matching_functions_plot(["Standard CIE 1931 2 Degree Observer", "Standard CIE 1964 10 Degree Observer"])
        True

    :param cmfss: Standard observers color matching functions to plot.
    :type cmfss: list
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
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
            x_limit_min.append(start)
            x_limit_max.append(end)
            y_limit_min.append(min(values))
            y_limit_max.append(max(values))

            pylab.plot(wavelengths, values, color=rgb, label=u"{0} - {1}".format(cmfs.labels.get(axis), cmfs.name),
                       linewidth=2.)

    settings = {"title": "{0} - Color Matching Functions".format(", ".join(cmfss)),
                "x_label": u"Wavelength λ (nm)",
                "y_label": "Tristimulus Values",
                "x_tighten": True,
                "legend": True,
                "legend_location": "upper right",
                "x_ticker": True,
                "y_ticker": True,
                "grid": True,
                "y_axis_line": True,
                "limits": [min(x_limit_min), max(x_limit_max), min(y_limit_min), max(y_limit_max)]}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def single_illuminant_relative_spd_plot(illuminant="A", cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
    """
    Plots given single illuminant relative spectral power distribution.

    Usage::

        >>> single_illuminant_relative_spd_plot("A")
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
                "y_label": "Relative Spectral Power Distribution"}
    settings.update(kwargs)

    return single_spectral_power_distribution_plot(illuminant, **settings)


def multi_illuminants_relative_spd_plot(illuminants=["A", "C", "D50"], **kwargs):
    """
    Plots given illuminants relative spectral power distributions.

    Usage::

        >>> multi_illuminants_relative_spd_plot(["A", "C", "D50"])
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
                "y_label": "Relative Spectral Power Distribution"}
    settings.update(kwargs)

    return multi_spectral_power_distribution_plot(spds, **settings)


def visible_spectrum_plot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
    """
    Plots the visible colors spectrum using given standard observer *CIE XYZ* color matching functions.

    Usage::

        >>> visible_spectrum_plot("Standard CIE 1931 2 Degree Observer")
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
                "x_label": u"Wavelength λ (nm)",
                "x_tighten": True}
    settings.update(kwargs)

    return color_parameters_plot(map(lambda x: color_parameter(x=x[0], RGB=x[1]), zip(wavelengths, colors)), **settings)


@figure_size((32, 32))
def CIE_1931_chromaticity_diagram_colors_plot(surface=1.25,
                                              spacing=0.00075,
                                              cmfs="Standard CIE 1931 2 Degree Observer",
                                              **kwargs):
    """
    Plots the *CIE 1931 Chromaticity Diagram* colors.

    Usage::

        >>> CIE_1931_chromaticity_diagram_colors_plot()
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
    x_dot, y_dot, colors = [], [], []
    for i in numpy.arange(0., 1., spacing):
        for j in numpy.arange(0., 1., spacing):
            if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
                x_dot.append(i)
                y_dot.append(j)

                XYZ = color.transformations.xy_to_XYZ((i, j))
                RGB = XYZ_to_sRGB(XYZ, illuminant)

                RGB = numpy.ravel(RGB)
                RGB *= 1. / numpy.max(RGB)
                RGB = numpy.clip(RGB, 0, 1)

                colors.append(RGB)

    pylab.scatter(x_dot, y_dot, color=colors, s=surface)

    settings = {"no_ticks": True,
                "bounding_box": [0., 1., 0., 1.],
                "bbox_inches": "tight",
                "pad_inches": 0}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


@figure_size((8, 8))
def CIE_1931_chromaticity_diagram_plot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
    """
    Plots the *CIE 1931 Chromaticity Diagram*.

    Usage::

        >>> CIE_1931_chromaticity_diagram_plot()
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
    equal_energy = numpy.array([1. / 3.] * 2)

    XYZs = [value for key, value in cmfs]

    x, y = zip(*(map(lambda x: color.transformations.XYZ_to_xy(x), XYZs)))

    wavelengths_chromaticity_coordinates = dict(zip(wavelengths, zip(x, y)))

    pylab.plot(x, y, color="black", linewidth=2.)
    pylab.plot((x[-1], x[0]), (y[-1], y[0]), color="black", linewidth=2.)

    for label in labels:
        x, y = wavelengths_chromaticity_coordinates.get(label)
        pylab.plot(x, y, "o", color="black", linewidth=2.)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = wavelengths[index] if index < len(wavelengths) else wavelengths[-1]

        dx = wavelengths_chromaticity_coordinates.get(right)[0] - wavelengths_chromaticity_coordinates.get(left)[0]
        dy = wavelengths_chromaticity_coordinates.get(right)[1] - wavelengths_chromaticity_coordinates.get(left)[1]

        normalize = lambda x: x / numpy.linalg.norm(x)

        xy = numpy.array([x, y])
        direction = numpy.array((-dy, dx))

        normal = numpy.array((-dy, dx)) if numpy.dot(normalize(xy - equal_energy),
                                                     normalize(direction)) > 0 else numpy.array((dy, -dx))
        normal = normalize(normal)
        normal /= 25

        pylab.plot([x, x + normal[0] * 0.75], [y, y + normal[1] * 0.75], color="black", linewidth=1.5)
        pylab.text(x + normal[0], y + normal[1], label, ha="left" if normal[0] >= 0 else "right", va="center",
                   fontdict={"size": "small"})

    settings = {"title": "CIE 1931 Chromaticity Diagram - {0}".format(name),
                "x_label": "CIE x",
                "y_label": "CIE y",
                "x_ticker": True,
                "y_ticker": True,
                "grid": True,
                "bounding_box": [-0.1, 0.9, -0.1, 0.9],
                "bbox_inches": "tight",
                "pad_inches": 0}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


@figure_size((8, 8))
def colorspaces_CIE_1931_chromaticity_diagram_plot(colorspaces=["sRGB", "ACES RGB", "Pointer Gamut"],
                                                   cmfs="Standard CIE 1931 2 Degree Observer",
                                                   **kwargs):
    """
    Plots given colorspaces in *CIE 1931 Chromaticity Diagram*.

    Usage::

        >>> colorspaces_CIE_1931_chromaticity_diagram_plot(["sRGB", "ACES RGB"])
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

    if not CIE_1931_chromaticity_diagram_plot(**settings):
        return

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [-0.1], [0.9], [-0.1], [0.9]
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

            random_color = lambda: float(random.randint(64, 224)) / 255
            r, g, b = random_color(), random_color(), random_color()

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

            x_limit_min.append(numpy.amin(primaries[:, 0]))
            y_limit_min.append(numpy.amin(primaries[:, 1]))
            x_limit_max.append(numpy.amax(primaries[:, 0]))
            y_limit_max.append(numpy.amax(primaries[:, 1]))

    settings.update({"legend": True,
                     "legend_location": "upper right",
                     "x_tighten": True,
                     "y_tighten": True,
                     "limits": [min(x_limit_min), max(x_limit_max), min(y_limit_min), max(y_limit_max)],
                     "margins": [-0.05, 0.05, -0.05, 0.05],
                     "standalone": True})

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


@figure_size((8, 8))
def planckian_locus_CIE_1931_chromaticity_diagram_plot(illuminants=["A", "C", "E"],
                                                       **kwargs):
    """
    Plots the planckian locus and given illuminants in *CIE 1931 Chromaticity Diagram*.

    Usage::

        >>> planckian_locus_CIE_1931_chromaticity_diagram_plot(["A", "C", "E"])
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

    if not CIE_1931_chromaticity_diagram_plot(**settings):
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


@figure_size((32, 32))
def CIE_1960_UCS_chromaticity_diagram_colors_plot(surface=1.25,
                                                  spacing=0.00075,
                                                  cmfs="Standard CIE 1931 2 Degree Observer",
                                                  **kwargs):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram* colors.

    Usage::

        >>> CIE_1960_UCS_chromaticity_diagram_colors_plot()
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
    x_dot, y_dot, colors = [], [], []
    for i in numpy.arange(0., 1., spacing):
        for j in numpy.arange(0., 1., spacing):
            if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
                x_dot.append(i)
                y_dot.append(j)

                XYZ = color.transformations.xy_to_XYZ(color.transformations.UVW_uv_to_xy((i, j)))
                RGB = XYZ_to_sRGB(XYZ, illuminant)

                RGB = numpy.ravel(RGB)
                RGB *= 1. / numpy.max(RGB)
                RGB = numpy.clip(RGB, 0, 1)

                colors.append(RGB)

    pylab.scatter(x_dot, y_dot, color=colors, s=surface)

    settings = {"no_ticks": True,
                "bounding_box": [0., 1., 0., 1.],
                "bbox_inches": "tight",
                "pad_inches": 0}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


@figure_size((8, 8))
def CIE_1960_UCS_chromaticity_diagram_plot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram*.

    Usage::

        >>> CIE_1960_UCS_chromaticity_diagram_plot()
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
    equal_energy = numpy.array([1. / 3.] * 2)

    UVWs = [color.transformations.XYZ_to_UVW(value) for key, value in cmfs]

    u, v = zip(*(map(lambda x: color.transformations.UVW_to_uv(x), UVWs)))

    wavelengths_chromaticity_coordinates = dict(zip(wavelengths, zip(u, v)))

    pylab.plot(u, v, color="black", linewidth=2.)
    pylab.plot((u[-1], u[0]), (v[-1], v[0]), color="black", linewidth=2.)

    for label in labels:
        u, v = wavelengths_chromaticity_coordinates.get(label)
        pylab.plot(u, v, "o", color="black", linewidth=2.)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = wavelengths[index] if index < len(wavelengths) else wavelengths[-1]

        dx = wavelengths_chromaticity_coordinates.get(right)[0] - wavelengths_chromaticity_coordinates.get(left)[0]
        dy = wavelengths_chromaticity_coordinates.get(right)[1] - wavelengths_chromaticity_coordinates.get(left)[1]

        normalize = lambda x: x / numpy.linalg.norm(x)

        uv = numpy.array([u, v])
        direction = numpy.array((-dy, dx))

        normal = numpy.array((-dy, dx)) if numpy.dot(normalize(uv - equal_energy),
                                                     normalize(direction)) > 0 else numpy.array((dy, -dx))
        normal = normalize(normal)
        normal /= 25

        pylab.plot([u, u + normal[0] * 0.75], [v, v + normal[1] * 0.75], color="black", linewidth=1.5)
        pylab.text(u + normal[0], v + normal[1], label, ha="left" if normal[0] >= 0 else "right", va="center",
                   fontdict={"size": "small"})

    settings = {"title": "CIE 1960 UCS Chromaticity Diagram - {0}".format(name),
                "x_label": "CIE u",
                "y_label": "CIE v",
                "x_ticker": True,
                "y_ticker": True,
                "grid": True,
                "bounding_box": [-0.075, 0.675, -0.15, 0.6],
                "bbox_inches": "tight",
                "pad_inches": 0}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


@figure_size((8, 8))
def planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot(illuminants=["A", "C", "E"],
                                                           **kwargs):
    """
    Plots the planckian locus and given illuminants in *CIE 1960 UCS Chromaticity Diagram*.

    Usage::

        >>> planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot(["A", "C", "E"])
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

    if not CIE_1960_UCS_chromaticity_diagram_plot(**settings):
        return

    xy_to_uv = lambda x: color.transformations.UVW_to_uv(
        color.transformations.XYZ_to_UVW(
            color.transformations.xy_to_XYZ(x)))

    start, end = 1667, 100000
    u, v = zip(*map(lambda x: color.temperature.cct_to_uv(x, 0., cmfs), numpy.arange(start, end + 250, 250)))

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


@figure_size((32, 32))
def CIE_1976_UCS_chromaticity_diagram_colors_plot(surface=1.25,
                                                  spacing=0.00075,
                                                  cmfs="Standard CIE 1931 2 Degree Observer",
                                                  **kwargs):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram* colors.

    Usage::

        >>> CIE_1976_UCS_chromaticity_diagram_colors_plot()
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
    x_dot, y_dot, colors = [], [], []
    for i in numpy.arange(0., 1., spacing):
        for j in numpy.arange(0., 1., spacing):
            if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
                x_dot.append(i)
                y_dot.append(j)

                XYZ = color.transformations.xy_to_XYZ(color.transformations.Luv_uv_to_xy((i, j)))
                RGB = XYZ_to_sRGB(XYZ, illuminant)

                RGB = numpy.ravel(RGB)
                RGB *= 1. / numpy.max(RGB)
                RGB = numpy.clip(RGB, 0, 1)

                colors.append(RGB)

    pylab.scatter(x_dot, y_dot, color=colors, s=surface)

    settings = {"no_ticks": True,
                "bounding_box": [0., 1., 0., 1.],
                "bbox_inches": "tight",
                "pad_inches": 0}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


@figure_size((8, 8))
def CIE_1976_UCS_chromaticity_diagram_plot(cmfs="Standard CIE 1931 2 Degree Observer", **kwargs):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram*.

    Usage::

        >>> CIE_1976_UCS_chromaticity_diagram_plot()
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
    equal_energy = numpy.array([1. / 3.] * 2)

    illuminant = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

    Luvs = [color.transformations.XYZ_to_Luv(value, illuminant) for key, value in cmfs]

    u, v = zip(*(map(lambda x: color.transformations.Luv_to_uv(x), Luvs)))

    wavelengths_chromaticity_coordinates = dict(zip(wavelengths, zip(u, v)))

    pylab.plot(u, v, color="black", linewidth=2.)
    pylab.plot((u[-1], u[0]), (v[-1], v[0]), color="black", linewidth=2.)

    for label in labels:
        u, v = wavelengths_chromaticity_coordinates.get(label)
        pylab.plot(u, v, "o", color="black", linewidth=2.)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = wavelengths[index] if index < len(wavelengths) else wavelengths[-1]

        dx = wavelengths_chromaticity_coordinates.get(right)[0] - wavelengths_chromaticity_coordinates.get(left)[0]
        dy = wavelengths_chromaticity_coordinates.get(right)[1] - wavelengths_chromaticity_coordinates.get(left)[1]

        normalize = lambda x: x / numpy.linalg.norm(x)

        uv = numpy.array([u, v])
        direction = numpy.array((-dy, dx))

        normal = numpy.array((-dy, dx)) if numpy.dot(normalize(uv - equal_energy),
                                                     normalize(direction)) > 0 else numpy.array((dy, -dx))
        normal = normalize(normal)
        normal /= 25

        pylab.plot([u, u + normal[0] * 0.75], [v, v + normal[1] * 0.75], color="black", linewidth=1.5)
        pylab.text(u + normal[0], v + normal[1], label, ha="left" if normal[0] >= 0 else "right", va="center",
                   fontdict={"size": "small"})

    settings = {"title": "CIE 1976 UCS Chromaticity Diagram - {0}".format(name),
                "x_label": "CIE u'",
                "y_label": "CIE v'",
                "x_ticker": True,
                "y_ticker": True,
                "grid": True,
                "bounding_box": [-0.1, .7, -.1, .7],
                "bbox_inches": "tight",
                "pad_inches": 0}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def single_munsell_value_function_plot(function="Munsell Value 1955",
                                       **kwargs):
    """
    Plots given *Lightness* function.

    Usage::

        >>> single_munsell_value_function_plot("Munsell Value 1955")
        True

    :param function: *Munsell value* function to plot.
    :type function: unicode
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    settings = {"title": "{0} - Munsell Value Function".format(function)}
    settings.update(kwargs)

    return multi_munsell_value_function_plot([function], **settings)


@figure_size((8, 8))
def multi_munsell_value_function_plot(functions=["Munsell Value 1955", "Munsell Value 1944"],
                                      **kwargs):
    """
    Plots given *Munsell value* functions.

    Usage::

        >>> multi_transfer_function_plot(["Lightness 1976", "Lightness 1964"])
        True

    :param functions: *Munsell value* functions to plot.
    :type functions: list
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    samples = numpy.linspace(0., 100., 1000)
    for i, function in enumerate(functions):
        function, name = color.lightness.MUNSELL_VALUE_FUNCTIONS.get(function), function
        if function is None:
            raise color.exceptions.ProgrammingError(
                "'{0}' 'Munsell value' function not found in supported 'Munsell value': '{1}'.".format(name,
                                                                                                       sorted(
                                                                                                           color.lightness.MUNSELL_VALUE_FUNCTIONS.keys())))

        pylab.plot(samples, map(function, samples), label=u"{0}".format(name), linewidth=2.)

    settings = {"title": "{0} - Munsell Functions".format(", ".join(functions)),
                "x_label": "Luminance Y",
                "y_label": "Munsell Value V",
                "x_tighten": True,
                "legend": True,
                "legend_location": "upper left",
                "x_ticker": True,
                "y_ticker": True,
                "grid": True,
                "limits": [0., 100., 0., 100.]}

    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def single_lightness_function_plot(function="Lightness 1976",
                                   **kwargs):
    """
    Plots given *Lightness* function.

    Usage::

        >>> single_transfer_function_plot("Lightness 1976")
        True

    :param function: *Lightness* function to plot.
    :type function: unicode
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    settings = {"title": "{0} - Lightness Function".format(function)}
    settings.update(kwargs)

    return multi_lightness_function_plot([function], **settings)


@figure_size((8, 8))
def multi_lightness_function_plot(functions=["Lightness 1976", "Lightness 1964", "Lightness 1958"],
                                  **kwargs):
    """
    Plots given *Lightness* functions.

    Usage::

        >>> multi_transfer_function_plot(["Lightness 1976", "Lightness 1964"])
        True

    :param functions: *Lightness* functions to plot.
    :type functions: list
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    samples = numpy.linspace(0., 100., 1000)
    for i, function in enumerate(functions):
        function, name = color.lightness.LIGHTNESS_FUNCTIONS.get(function), function
        if function is None:
            raise color.exceptions.ProgrammingError(
                "'{0}' 'Lightness' function not found in supported 'Lightness': '{1}'.".format(name,
                                                                                               sorted(
                                                                                                   color.lightness.LIGHTNESS_FUNCTIONS.keys())))

        pylab.plot(samples, map(function, samples), label=u"{0}".format(name), linewidth=2.)

    settings = {"title": "{0} - Lightness Functions".format(", ".join(functions)),
                "x_label": "Luminance Y",
                "y_label": "Lightness L*",
                "x_tighten": True,
                "legend": True,
                "legend_location": "upper left",
                "x_ticker": True,
                "y_ticker": True,
                "grid": True,
                "limits": [0., 100., 0., 100.]}

    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def single_transfer_function_plot(colorspace="sRGB",
                                  **kwargs):
    """
    Plots given colorspace transfer function.

    Usage::

        >>> single_transfer_function_plot("sRGB")
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

    return multi_transfer_function_plot([colorspace], **settings)


@figure_size((8, 8))
def multi_transfer_function_plot(colorspaces=["sRGB", "Rec. 709"],
                                 inverse=False,
                                 **kwargs):
    """
    Plots given colorspaces transfer functions.

    Usage::

        >>> multi_transfer_function_plot(["sRGB", "Rec. 709"])
        True

    :param colorspaces: Colorspaces transfer functions to plot.
    :type colorspaces: list
    :param inverse: Plot inverse transfer functions.
    :type inverse: bool
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Definition success.
    :rtype: bool
    """

    samples = numpy.linspace(0., 1., 1000)
    for i, colorspace in enumerate(colorspaces):
        colorspace, name = color.colorspaces.COLORSPACES.get(colorspace), colorspace
        if colorspace is None:
            raise color.exceptions.ProgrammingError(
                "'{0}' colorspace not found in supported colorspaces: '{1}'.".format(name,
                                                                                     sorted(
                                                                                         color.colorspaces.COLORSPACES.keys())))

        RGBs = numpy.array(map(colorspace.inverse_transfer_function if inverse else colorspace.transfer_function, samples))
        pylab.plot(samples, RGBs, label=u"{0}".format(colorspace.name), linewidth=2.)

    settings = {"title": "{0} - Transfer Functions".format(", ".join(colorspaces)),
                "x_tighten": True,
                "legend": True,
                "legend_location": "upper left",
                "x_ticker": True,
                "y_ticker": True,
                "grid": True,
                "limits": [0., 1., 0., 1.]}

    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)


def blackbody_spectral_radiance_plot(temperature=3500,
                                     cmfs="Standard CIE 1931 2 Degree Observer",
                                     blackbody="VY Canis Major",
                                     **kwargs):
    """
    Plots given blackbody spectral radiance.

    Usage::

        >>> blackbody_spectral_radiance_plot(3500)
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

    spd = color.blackbody.blackbody_spectral_power_distribution(temperature, *cmfs.shape)

    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.subplot(211)

    settings = {"title": "{0} - Spectral Radiance".format(blackbody),
                "y_label": u"W / (sr m²) / m",
                "standalone": False}
    settings.update(kwargs)

    single_spectral_power_distribution_plot(spd, cmfs, **settings)

    XYZ = color.transformations.spectral_to_XYZ(spd, cmfs)
    RGB = XYZ_to_sRGB(XYZ)
    RGB *= 1. / numpy.max(RGB)

    matplotlib.pyplot.subplot(212)

    settings = {"title": "{0} - Color".format(blackbody),
                "x_label": "{0}K".format(temperature),
                "y_label": "",
                "aspect": None,
                "standalone": False}

    single_color_plot(color_parameter(name="", RGB=RGB), **settings)

    settings = {"standalone": True}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)
    return display(**settings)


def blackbody_colors_plot(start=1000,
                          end=15000,
                          steps=25,
                          cmfs="Standard CIE 1931 2 Degree Observer",
                          **kwargs):
    """
    Plots blackbody colors.

    Usage::

        >>> blackbody_colors_plot()
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
        spd = color.blackbody.blackbody_spectral_power_distribution(temperature, *cmfs.shape)

        XYZ = color.transformations.spectral_to_XYZ(spd, cmfs)
        RGB = XYZ_to_sRGB(XYZ)

        RGB = numpy.ravel(RGB)
        RGB *= 1. / numpy.max(RGB)
        RGB = numpy.clip(RGB, 0, 1)

        colors.append(RGB)
        temperatures.append(temperature)

    settings = {"title": "Blackbody Colors",
                "x_label": "Temperature K",
                "y_label": "",
                "x_tighten": True,
                "x_ticker": True,
                "y_ticker": False}

    settings.update(kwargs)
    return color_parameters_plot(map(lambda x: color_parameter(x=x[0], RGB=x[1]), zip(temperatures, colors)),
                                 **settings)
