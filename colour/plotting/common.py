#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Plotting
===============

Defines the common plotting objects:

-   :func:`colour_cycle`
-   :func:`canvas`
-   :func:`decorate`
-   :func:`boundaries`
-   :func:`display`
-   :func:`colour_parameter`
-   :func:`colour_parameters_plot`
-   :func:`single_colour_plot`
-   :func:`multi_colour_plot`
-   :func:`image_plot`
"""

from __future__ import division

import itertools
import os
from collections import namedtuple

import matplotlib
import matplotlib.image
import matplotlib.path
import matplotlib.pyplot
import matplotlib.ticker
import numpy as np
import pylab

from colour.utilities import Structure

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['PLOTTING_RESOURCES_DIRECTORY',
           'DEFAULT_FIGURE_ASPECT_RATIO',
           'DEFAULT_FIGURE_WIDTH',
           'DEFAULT_FIGURE_HEIGHT',
           'DEFAULT_FIGURE_SIZE',
           'DEFAULT_FONT_SIZE',
           'DEFAULT_PARAMETERS',
           'DEFAULT_COLOUR_CYCLE',
           'ColourParameter',
           'ColourParameter',
           'colour_cycle',
           'canvas',
           'decorate',
           'boundaries',
           'display',
           'colour_parameter',
           'colour_parameters_plot',
           'single_colour_plot',
           'multi_colour_plot',
           'image_plot']

PLOTTING_RESOURCES_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'resources')
"""
Resources directory.

RESOURCES_DIRECTORY : unicode
"""

DEFAULT_FIGURE_ASPECT_RATIO = (np.sqrt(5) - 1) / 2
"""
Default figure aspect ratio (Golden Number).

DEFAULT_FIGURE_ASPECT_RATIO : float
"""

DEFAULT_FIGURE_WIDTH = 18
"""
Default figure width.

DEFAULT_FIGURE_WIDTH : integer
"""

if 'Qt4Agg' in matplotlib.get_backend():
    DEFAULT_FIGURE_WIDTH = 12

DEFAULT_FIGURE_HEIGHT = DEFAULT_FIGURE_WIDTH * DEFAULT_FIGURE_ASPECT_RATIO
"""
Default figure height.

DEFAULT_FIGURE_HEIGHT : integer
"""

DEFAULT_FIGURE_SIZE = DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT
"""
Default figure size.

DEFAULT_FIGURE_SIZE : tuple
"""

DEFAULT_FONT_SIZE = 12
"""
Default figure font size.

DEFAULT_FONT_SIZE : numeric
"""

if 'Qt4Agg' in matplotlib.get_backend():
    DEFAULT_FONT_SIZE = 10

DEFAULT_PARAMETERS = {
    'figure.figsize': DEFAULT_FIGURE_SIZE,
    'font.size': DEFAULT_FONT_SIZE,
    'axes.titlesize': DEFAULT_FONT_SIZE * 1.25,
    'axes.labelsize': DEFAULT_FONT_SIZE * 1.25,
    'legend.fontsize': DEFAULT_FONT_SIZE * 0.9,
    'xtick.labelsize': DEFAULT_FONT_SIZE,
    'ytick.labelsize': DEFAULT_FONT_SIZE
}
"""
Default plotting parameters.

DEFAULT_PARAMETERS : dict
"""

pylab.rcParams.update(DEFAULT_PARAMETERS)

DEFAULT_COLOUR_CYCLE = ('r', 'g', 'b', 'c', 'm', 'y', 'k')

ColourParameter = namedtuple('ColourParameter',
                             ('name', 'RGB', 'x', 'y0', 'y1'))


def colour_cycle(**kwargs):
    """
    Returns a colour cycle iterator using given colour map.

   Parameters
    ----------
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    cycle
        Colour cycle iterator.
    """

    settings = Structure(
        **{'colour_cycle_map': 'hsv',
           'colour_cycle_count': len(DEFAULT_COLOUR_CYCLE)})
    settings.update(kwargs)

    if settings.colour_cycle_map is None:
        cycle = DEFAULT_COLOUR_CYCLE
    else:
        cycle = getattr(matplotlib.pyplot.cm,
                        settings.colour_cycle_map)(
            np.linspace(0, 1, settings.colour_cycle_count))

    return itertools.cycle(cycle)


def canvas(**kwargs):
    """
    Sets the figure size and aspect.

    Parameters
    ----------
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    Figure
        Current figure.
    """

    settings = Structure(
        **{'figure_size': DEFAULT_FIGURE_SIZE})
    settings.update(kwargs)

    figure = matplotlib.pyplot.gcf()
    if figure is None:
        figure = matplotlib.pyplot.figure(figsize=settings.figure_size)
    else:
        figure.set_size_inches(settings.figure_size)

    return figure


def decorate(**kwargs):
    """
    Sets the figure decorations.

    Parameters
    ----------
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.
    """

    settings = Structure(
        **{'title': None,
           'x_label': None,
           'y_label': None,
           'legend': False,
           'legend_location': 'upper right',
           'x_ticker': False,
           'y_ticker': False,
           'x_ticker_locator': matplotlib.ticker.AutoMinorLocator(2),
           'y_ticker_locator': matplotlib.ticker.AutoMinorLocator(2),
           'no_ticks': False,
           'no_x_ticks': False,
           'no_y_ticks': False,
           'grid': False,
           'grid_which': 'both',
           'grid_axis': 'both',
           'x_axis_line': False,
           'y_axis_line': False,
           'aspect': None})
    settings.update(kwargs)

    if settings.title:
        pylab.title(settings.title)
    if settings.x_label:
        pylab.xlabel(settings.x_label)
    if settings.y_label:
        pylab.ylabel(settings.y_label)
    if settings.legend:
        pylab.legend(loc=settings.legend_location)
    if settings.x_ticker:
        matplotlib.pyplot.gca().xaxis.set_minor_locator(
            settings.x_ticker_locator)
    if settings.y_ticker:
        matplotlib.pyplot.gca().yaxis.set_minor_locator(
            settings.y_ticker_locator)
    if settings.no_ticks:
        matplotlib.pyplot.gca().set_xticks([])
        matplotlib.pyplot.gca().set_yticks([])
    if settings.no_x_ticks:
        matplotlib.pyplot.gca().set_xticks([])
    if settings.no_y_ticks:
        matplotlib.pyplot.gca().set_yticks([])
    if settings.grid:
        pylab.grid(which=settings.grid_which, axis=settings.grid_axis)
    if settings.x_axis_line:
        pylab.axvline(color='black', linestyle='--')
    if settings.y_axis_line:
        pylab.axhline(color='black', linestyle='--')
    if settings.aspect:
        matplotlib.pyplot.axes().set_aspect(settings.aspect)

    return True


def boundaries(**kwargs):
    """
    Sets the plot boundaries.

    Parameters
    ----------
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.
    """

    settings = Structure(
        **{'bounding_box': None,
           'x_tighten': False,
           'y_tighten': False,
           'limits': (0, 1, 0, 1),
           'margins': (0, 0, 0, 0)})
    settings.update(kwargs)

    if settings.bounding_box is None:
        x_limit_min, x_limit_max, y_limit_min, y_limit_max = (
            settings.limits)
        x_margin_min, x_margin_max, y_margin_min, y_margin_max = (
            settings.margins)
        if settings.x_tighten:
            pylab.xlim(x_limit_min + x_margin_min, x_limit_max + x_margin_max)
        if settings.y_tighten:
            pylab.ylim(y_limit_min + y_margin_min, y_limit_max + y_margin_max)
    else:
        pylab.xlim(settings.bounding_box[0], settings.bounding_box[1])
        pylab.ylim(settings.bounding_box[2], settings.bounding_box[3])

    return True


def display(**kwargs):
    """
    Sets the figure display.

    Parameters
    ----------
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.
    """

    settings = Structure(
        **{'standalone': True,
           'filename': None})
    settings.update(kwargs)

    if settings.standalone:
        if settings.filename is not None:
            pylab.savefig(**kwargs)
        else:
            pylab.show()
        pylab.close()

    return True


def colour_parameter(name=None, RGB=None, x=None, y0=None, y1=None):
    """
    Defines a factory for
    :attr:`colour.plotting.plots.COLOUR_PARAMETER` attribute.

    Parameters
    ----------
    name : unicode, optional
        Colour name.
    RGB : array_like, optional
        RGB Colour.
    x : numeric, optional
        X data.
    y0 : numeric, optional
        Y0 data.
    y1 : numeric, optional
        Y1 data.

    Returns
    -------
    ColourParameter
        ColourParameter.
    """

    return ColourParameter(name, RGB, x, y0, y1)


def colour_parameters_plot(colour_parameters,
                           y0_plot=True,
                           y1_plot=True,
                           **kwargs):
    """
    Plots given colour colour_parameters.

    Parameters
    ----------
    colour_parameters : list
        ColourParameter sequence.
    y0_plot : bool, optional
        Plot y0 line.
    y1_plot : bool, optional
        Plot y1 line.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> cp1 = colour_parameter(x=390, RGB=[0.03009021, 0, 0.12300545])
    >>> cp2 = colour_parameter(x=391, RGB=[0.03434063, 0, 0.13328537], y0=0, y1=0.25)  # noqa
    >>> cp3 = colour_parameter(x=392, RGB=[0.03826312, 0, 0.14276247], y0=0, y1=0.35)  # noqa
    >>> cp4 = colour_parameter(x=393, RGB=[0.04191844, 0, 0.15158707], y0=0, y1=0.05)  # noqa
    >>> cp5 = colour_parameter(x=394, RGB=[0.04535085, 0, 0.15986838], y0=0, y1=-.25)  # noqa
    >>> colour_parameters_plot([cp1, cp2, cp3, cp3, cp4, cp5])  # noqa  # doctest: +SKIP
    True
    """

    canvas(**kwargs)

    for i in range(len(colour_parameters) - 1):
        x0 = colour_parameters[i].x
        x01 = colour_parameters[i + 1].x
        y0 = (0
              if colour_parameters[i].y0 is None else
              colour_parameters[i].y0)
        y1 = (1
              if colour_parameters[i].y1 is None else
              colour_parameters[i].y1)
        y01 = (0
               if colour_parameters[i].y0 is None else
               colour_parameters[i + 1].y0)
        y11 = (1
               if colour_parameters[i].y1 is None else
               colour_parameters[i + 1].y1)

        x_polygon = (x0, x01, x01, x0)
        y_polygon = (y0, y01, y11, y1)
        pylab.fill(x_polygon,
                   y_polygon,
                   color=colour_parameters[i].RGB,
                   edgecolor=colour_parameters[i].RGB)

    if all([x.y0 is not None for x in colour_parameters]):
        if y0_plot:
            pylab.plot([x.x for x in colour_parameters],
                       [x.y0 for x in colour_parameters],
                       color='black',
                       linewidth=2)

    if all([x.y1 is not None for x in colour_parameters]):
        if y1_plot:
            pylab.plot([x.x for x in colour_parameters],
                       [x.y1 for x in colour_parameters],
                       color='black',
                       linewidth=2)

    y_limit_min0 = min(
        [0 if x.y0 is None else x.y0 for x in colour_parameters])
    # y_limit_max0 = max(
    # [1 if x.y0 is None else x.y0 for x in colour_parameters])
    # y_limit_min1 = min(
    # [0 if x.y1 is None else x.y1 for x in colour_parameters])
    y_limit_max1 = max(
        [1 if x.y1 is None else x.y1 for x in colour_parameters])

    settings = {
        'x_label': 'Parameter',
        'y_label': 'Colour',
        'limits': (min([0 if x.x is None else x.x for x in colour_parameters]),
                   max([1 if x.x is None else x.x for x in colour_parameters]),
                   y_limit_min0,
                   y_limit_max1)}
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def single_colour_plot(colour_parameter, **kwargs):
    """
    Plots given colour.

    Parameters
    ----------
    colour_parameter : ColourParameter
        ColourParameter.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> RGB = (0.32315746, 0.32983556, 0.33640183)
    >>> single_colour_plot(colour_parameter(RGB))  # doctest: +SKIP
    True
    """

    return multi_colour_plot((colour_parameter, ), **kwargs)


def multi_colour_plot(colour_parameters,
                      width=1,
                      height=1,
                      spacing=0,
                      across=3,
                      text_display=True,
                      text_size='large',
                      text_offset=0.075,
                      **kwargs):
    """
    Plots given colours.

    Parameters
    ----------
    colour_parameters : list
        ColourParameter sequence.
    width : numeric, optional
        Colour polygon width.
    height : numeric, optional
        Colour polygon height.
    spacing : numeric, optional
        Colour polygons spacing.
    across : int, optional
        Colour polygons count per row.
    text_display : bool, optional
        Display colour text.
    text_size : numeric, optional
        Colour text size.
    text_offset : numeric, optional
        Colour text offset.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> cp1 = colour_parameter(RGB=(0.45293517, 0.31732158, 0.26414773))
    >>> cp2 = colour_parameter(RGB=(0.77875824, 0.57726450, 0.50453169))
    >>> multi_colour_plot([cp1, cp2])  # doctest: +SKIP
    True
    """

    canvas(**kwargs)

    offsetX = offsetY = 0
    x_limit_min, x_limit_max, y_limit_min, y_limit_max = 0, width, 0, height
    for i, colour_parameter in enumerate(colour_parameters):
        if i % across == 0 and i != 0:
            offsetX = 0
            offsetY -= height + spacing

        x0 = offsetX
        x1 = offsetX + width
        y0 = offsetY
        y1 = offsetY + height

        x_polygon = (x0, x1, x1, x0)
        y_polygon = (y0, y0, y1, y1)
        pylab.fill(x_polygon, y_polygon, color=colour_parameters[i].RGB)
        if colour_parameter.name is not None and text_display:
            pylab.text(x0 + text_offset, y0 + text_offset,
                       colour_parameter.name, clip_on=True, size=text_size)

        offsetX += width + spacing

    x_limit_max = min(len(colour_parameters), across)
    x_limit_max = x_limit_max * width + x_limit_max * spacing - spacing
    y_limit_min = offsetY

    settings = {
        'x_tighten': True,
        'y_tighten': True,
        'no_ticks': True,
        'limits': (x_limit_min, x_limit_max, y_limit_min, y_limit_max),
        'aspect': 'equal'}
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def image_plot(image,
               label=None,
               label_size=15,
               label_colour=None,
               label_alpha=0.85,
               **kwargs):
    """
    Plots given image.

    Parameters
    ----------
    image : array_like
        Image to plot.
    label: unicode, optional
        Image label.
    label_size: int, optional
        Image label font size.
    label_colour: array_like or unicode, optional
        Image label colour.
    label_alpha: numeric, optional
        Image label alpha.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> path = os.path.join('resources', 'CIE_1931_Chromaticity_Diagram_CIE_1931_2_Degree_Standard_Observer.png')  # noqa
    >>> image = read_image(path)  # doctest: +SKIP
    >>> image_plot(image)  # doctest: +SKIP
    True
    """

    image = np.asarray(image)

    pylab.imshow(np.clip(image, 0, 1))

    height, _width, _channels = image.shape

    pylab.text(0 + label_size,
               height - label_size,
               label,
               color=label_colour if label_colour is not None else (1, 1, 1),
               alpha=label_alpha,
               fontsize=label_size)

    settings = {'no_ticks': True,
                'bounding_box': (0, 1, 0, 1),
                'bbox_inches': 'tight',
                'pad_inches': 0}
    settings.update(kwargs)

    canvas(**settings)
    decorate(**settings)

    return display(**settings)
