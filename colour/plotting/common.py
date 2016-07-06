#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Plotting
===============

Defines the common plotting objects:

-   :func:`colour_cycle`
-   :func:`canvas`
-   :func:`camera`
-   :func:`decorate`
-   :func:`boundaries`
-   :func:`display`
-   :func:`label_rectangles`
-   :func:`equal_axes3d`
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
import matplotlib.cm
import matplotlib.pyplot
import matplotlib.ticker
import numpy as np
import pylab

from colour.colorimetry import CMFS, ILLUMINANTS, ILLUMINANTS_RELATIVE_SPDS
from colour.models import RGB_COLOURSPACES
from colour.utilities import Structure

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
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
           'DEFAULT_COLOUR_CYCLE',
           'DEFAULT_HATCH_PATTERNS',
           'DEFAULT_PARAMETERS',
           'DEFAULT_PLOTTING_ILLUMINANT',
           'DEFAULT_PLOTTING_ENCODING_CCTF',
           'ColourParameter',
           'colour_cycle',
           'canvas',
           'camera',
           'decorate',
           'boundaries',
           'display',
           'label_rectangles',
           'equal_axes3d',
           'get_RGB_colourspace',
           'get_cmfs',
           'get_illuminant',
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

DEFAULT_COLOUR_CYCLE = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
"""
Default colour cycle for plots.

DEFAULT_COLOUR_CYCLE : tuple
**{'r', 'g', 'b', 'c', 'm', 'y', 'k'}**
"""

DEFAULT_HATCH_PATTERNS = ('\\\\', 'o', 'x', '.', '*', '//')
"""
Default hatch patterns for bar plots.

DEFAULT_HATCH_PATTERNS : tuple
{'\\\\', 'o', 'x', '.', '*', '//'}
"""

DEFAULT_PARAMETERS = {
    'figure.figsize': DEFAULT_FIGURE_SIZE,
    'font.size': DEFAULT_FONT_SIZE,
    'axes.titlesize': DEFAULT_FONT_SIZE * 1.25,
    'axes.labelsize': DEFAULT_FONT_SIZE * 1.25,
    'legend.fontsize': DEFAULT_FONT_SIZE * 0.9,
    'xtick.labelsize': DEFAULT_FONT_SIZE,
    'ytick.labelsize': DEFAULT_FONT_SIZE,
    'axes.color_cycle': DEFAULT_COLOUR_CYCLE}
"""
Default plotting parameters.

DEFAULT_PARAMETERS : dict
"""

pylab.rcParams.update(DEFAULT_PARAMETERS)

DEFAULT_PLOTTING_ILLUMINANT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get('D65')
"""
Default plotting illuminant: *CIE Illuminant D Series* *D65*.

DEFAULT_PLOTTING_ILLUMINANT : tuple
"""

DEFAULT_PLOTTING_ENCODING_CCTF = RGB_COLOURSPACES['sRGB'].encoding_cctf
"""
Default plotting encoding colour component transfer function / opto-electronic
transfer function: *sRGB*.

DEFAULT_PLOTTING_ENCODING_CCTF : object
"""


class ColourParameter(
    namedtuple('ColourParameter',
               ('name', 'RGB', 'x', 'y0', 'y1'))):
    """
    Defines a data structure for plotting a colour polygon in various spectral
    figures.

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
    """

    def __new__(cls, name=None, RGB=None, x=None, y0=None, y1=None):
        """
        Returns a new instance of the :class:`ColourParameter` class.
        """

        return super(ColourParameter, cls).__new__(
            cls, name, RGB, x, y0, y1)


def colour_cycle(**kwargs):
    """
    Returns a colour cycle iterator using given colour map.

    Parameters
    ----------
    \**kwargs : dict, optional
        **{'colour_cycle_map', 'colour_cycle_count'}**
        Keywords arguments such as ``{'colour_cycle_map': unicode
        (Matplotlib colormap name), 'colour_cycle_count': int}``

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
    Sets the figure size.

    Parameters
    ----------
    \**kwargs : dict, optional
        **{'figure_size', }**
        Keywords arguments such as ``{'figure_size': array_like
        (width, height), }``

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


def camera(**kwargs):
    """
    Sets the camera settings.

    Parameters
    ----------
    \**kwargs : dict, optional
        **{'camera_aspect', 'elevation', 'azimuth'}**
        Keywords arguments such as ``{'camera_aspect': unicode
        (Matplotlib axes aspect), 'elevation' : numeric, 'azimuth' : numeric}``

    Returns
    -------
    Axes
        Current axes.
    """

    settings = Structure(
        **{'camera_aspect': 'equal',
           'elevation': None,
           'azimuth': None})
    settings.update(kwargs)

    axes = matplotlib.pyplot.gca()
    if settings.camera_aspect == 'equal':
        equal_axes3d(axes)

    axes.view_init(elev=settings.elevation, azim=settings.azimuth)

    return axes


def decorate(**kwargs):
    """
    Sets the figure decorations.

    Parameters
    ----------
    \**kwargs : dict, optional
        **{'title', 'x_label', 'y_label', 'legend', 'legend_columns',
        'legend_location', 'x_ticker', 'y_ticker', 'x_ticker_locator',
        'y_ticker_locator', 'grid', 'grid_which', 'grid_axis', 'x_axis_line',
        'y_axis_line', 'aspect', 'no_axes'}**
        Keywords arguments such as ``{'title': unicode (figure title),
        'x_label': unicode (X axis label), 'y_label': unicode (Y axis label),
        'legend': bool, 'legend_columns': int, 'legend_location': unicode
        (Matplotlib legend location), 'x_ticker': bool, 'y_ticker': bool,
        'x_ticker_locator': Locator, 'y_ticker_locator': Locator, 'grid': bool,
        'grid_which': unicode, 'grid_axis': unicode, 'x_axis_line': bool,
        'y_axis_line': bool, 'aspect': unicode (Matplotlib axes aspect),
        'no_axes': bool}``

    Returns
    -------
    Axes
        Current axes.
    """

    settings = Structure(
        **{'title': None,
           'x_label': None,
           'y_label': None,
           'legend': False,
           'legend_columns': 1,
           'legend_location': 'upper right',
           'x_ticker': True,
           'y_ticker': True,
           'x_ticker_locator': matplotlib.ticker.AutoMinorLocator(2),
           'y_ticker_locator': matplotlib.ticker.AutoMinorLocator(2),
           'grid': False,
           'grid_which': 'both',
           'grid_axis': 'both',
           'x_axis_line': False,
           'y_axis_line': False,
           'aspect': None,
           'no_axes': False})
    settings.update(kwargs)

    axes = matplotlib.pyplot.gca()
    if settings.title:
        pylab.title(settings.title)
    if settings.x_label:
        pylab.xlabel(settings.x_label)
    if settings.y_label:
        pylab.ylabel(settings.y_label)
    if settings.legend:
        pylab.legend(loc=settings.legend_location,
                     ncol=settings.legend_columns)
    if settings.x_ticker:
        axes.xaxis.set_minor_locator(
            settings.x_ticker_locator)
    else:
        axes.set_xticks([])
    if settings.y_ticker:
        axes.yaxis.set_minor_locator(
            settings.y_ticker_locator)
    else:
        axes.set_yticks([])
    if settings.grid:
        pylab.grid(which=settings.grid_which, axis=settings.grid_axis)
    if settings.x_axis_line:
        pylab.axvline(color='black', linestyle='--')
    if settings.y_axis_line:
        pylab.axhline(color='black', linestyle='--')
    if settings.aspect:
        matplotlib.pyplot.axes().set_aspect(settings.aspect)
    if settings.no_axes:
        axes.set_axis_off()

    return axes


def boundaries(**kwargs):
    """
    Sets the plot boundaries.

    Parameters
    ----------
    \**kwargs : dict, optional
        **{'bounding_box', 'x_tighten', 'y_tighten', 'limits', 'margins'}**
        Keywords arguments such as ``{'bounding_box': array_like
        (x min, x max, y min, y max), 'x_tighten': bool, 'y_tighten': bool,
        'limits': array_like (x min, x max, y min, y max), 'limits': array_like
        (x min, x max, y min, y max)}``

    Returns
    -------
    Axes
        Current axes.
    """

    settings = Structure(
        **{'bounding_box': None,
           'x_tighten': False,
           'y_tighten': False,
           'limits': (0, 1, 0, 1),
           'margins': (0, 0, 0, 0)})
    settings.update(kwargs)

    axes = matplotlib.pyplot.gca()
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

    return axes


def display(**kwargs):
    """
    Sets the figure display.

    Parameters
    ----------
    \**kwargs : dict, optional
        **{'standalone', 'filename'}**
        Keywords arguments such as ``{'standalone': bool (figure is shown),
        'filename': unicode (figure is saved as `filename`)}``

    Returns
    -------
    Figure
        Current figure or None.
    """

    settings = Structure(
        **{'standalone': True,
           'filename': None})
    settings.update(kwargs)

    figure = matplotlib.pyplot.gcf()
    if settings.standalone:
        if settings.filename is not None:
            pylab.savefig(**kwargs)
        else:
            pylab.show()
        pylab.close()

        return None
    else:
        return figure


def label_rectangles(rectangles,
                     rotation='vertical',
                     text_size=10,
                     offset=None):
    """
    Add labels above given rectangles.

    Parameters
    ----------
    rectangles : object
        Rectangles to used to set the labels value and position.
    rotation : unicode, optional
        **{'horizontal', 'vertical'}**,
        Labels orientation.
    text_size : numeric, optional
        Labels text size.
    offset : array_like, optional
        Labels offset as percentages of the largest rectangle dimensions.

    Returns
    -------
    bool
        Definition success.
    """

    if offset is None:
        offset = (0.0, 0.025)

    x_m, y_m = 0, 0
    for rectangle in rectangles:
        x_m = max(x_m, rectangle.get_width())
        y_m = max(y_m, rectangle.get_height())

    for rectangle in rectangles:
        x = rectangle.get_x()
        height = rectangle.get_height()
        width = rectangle.get_width()
        ha = 'center'
        va = 'bottom'
        pylab.text(x + width / 2 + offset[0] * width,
                   height + offset[1] * y_m,
                   '{0:.1f}'.format(height),
                   ha=ha, va=va,
                   rotation=rotation,
                   fontsize=text_size,
                   clip_on=True)

    return True


def equal_axes3d(axes):
    """
    Sets equal aspect ratio to given 3d axes.

    Parameters
    ----------
    axes : object
        Axis to set the equal aspect ratio.

    Returns
    -------
    bool
        Definition success.
    """

    axes.set_aspect('equal')
    extents = np.array([getattr(axes, 'get_{}lim'.format(axis))()
                        for axis in 'xyz'])

    centers = np.mean(extents, axis=1)
    extent = np.max(np.abs(extents[..., 1] - extents[..., 0]))

    for center, axis in zip(centers, 'xyz'):
        getattr(axes, 'set_{}lim'.format(axis))(center - extent / 2,
                                                center + extent / 2)

    return True


def get_RGB_colourspace(colourspace):
    """
    Returns the *RGB* colourspace with given name.

    Parameters
    ----------
    colourspace : unicode
        *RGB* colourspace name.

    Returns
    -------
    RGB_Colourspace
        *RGB* colourspace.

    Raises
    ------
    KeyError
        If the given *RGB* colourspace is not found in the factory *RGB*
        colourspaces.
    """

    colourspace, name = RGB_COLOURSPACES.get(colourspace), colourspace
    if colourspace is None:
        raise KeyError(
            ('"{0}" colourspace not found in factory RGB colourspaces: '
             '"{1}".').format(
                name, ', '.join(sorted(RGB_COLOURSPACES.keys()))))

    return colourspace


def get_cmfs(cmfs):
    """
    Returns the colour matching functions with given name.

    Parameters
    ----------
    cmfs : unicode
        Colour matching functions name.

    Returns
    -------
    RGB_ColourMatchingFunctions or XYZ_ColourMatchingFunctions
        Colour matching functions.

    Raises
    ------
    KeyError
        If the given colour matching functions is not found in the factory
        colour matching functions.
    """

    cmfs, name = CMFS.get(cmfs), cmfs
    if cmfs is None:
        raise KeyError(
            ('"{0}" not found in factory colour matching functions: '
             '"{1}".').format(name, ', '.join(sorted(CMFS.keys()))))
    return cmfs


def get_illuminant(illuminant):
    """
    Returns the illuminant with given name.

    Parameters
    ----------
    illuminant : unicode
        Illuminant name.

    Returns
    -------
    SpectralPowerDistribution
        Illuminant.

    Raises
    ------
    KeyError
        If the given illuminant is not found in the factory illuminants.
    """

    illuminant, name = ILLUMINANTS_RELATIVE_SPDS.get(illuminant), illuminant
    if illuminant is None:
        raise KeyError(
            '"{0}" not found in factory illuminants: "{1}".'.format(
                name, ', '.join(sorted(ILLUMINANTS_RELATIVE_SPDS.keys()))))

    return illuminant


def colour_parameters_plot(colour_parameters,
                           y0_plot=True,
                           y1_plot=True,
                           **kwargs):
    """
    Plots given colour colour parameters.

    Parameters
    ----------
    colour_parameters : list
        ColourParameter sequence.
    y0_plot : bool, optional
        Plot y0 line.
    y1_plot : bool, optional
        Plot y1 line.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> cp1 = ColourParameter(
    ...     x=390, RGB=[0.03009021, 0, 0.12300545])
    >>> cp2 = ColourParameter(
    ...     x=391, RGB=[0.03434063, 0, 0.13328537], y0=0, y1=0.25)
    >>> cp3 = ColourParameter(
    ...     x=392, RGB=[0.03826312, 0, 0.14276247], y0=0, y1=0.35)
    >>> cp4 = ColourParameter(
    ...     x=393, RGB=[0.04191844, 0, 0.15158707], y0=0, y1=0.05)
    >>> cp5 = ColourParameter(
    ...     x=394, RGB=[0.04535085, 0, 0.15986838], y0=0, y1=-.25)
    >>> colour_parameters_plot(
    ...     [cp1, cp2, cp3, cp3, cp4, cp5])  # doctest: +SKIP
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

    if all([x.y0 is not None for x in colour_parameters]) and y0_plot:
        pylab.plot([x.x for x in colour_parameters],
                   [x.y0 for x in colour_parameters],
                   color='black',
                   linewidth=2)

    if all([x.y1 is not None for x in colour_parameters]) and y1_plot:
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
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> RGB = (0.32315746, 0.32983556, 0.33640183)
    >>> single_colour_plot(ColourParameter(RGB))  # doctest: +SKIP
    """

    return multi_colour_plot((colour_parameter,), **kwargs)


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
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> cp1 = ColourParameter(RGB=(0.45293517, 0.31732158, 0.26414773))
    >>> cp2 = ColourParameter(RGB=(0.77875824, 0.57726450, 0.50453169))
    >>> multi_colour_plot([cp1, cp2])  # doctest: +SKIP
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
        'x_ticker': False,
        'y_ticker': False,
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
               interpolation='nearest',
               colour_map=matplotlib.cm.Greys_r,
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
    interpolation: unicode, optional
        **{'nearest', None, 'none', 'bilinear', 'bicubic', 'spline16',
        'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
        'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'}**
        Image display interpolation.
    colour_map: unicode, optional
        Colour map used to display single channel images.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> path = os.path.join(
    ...     'resources',
    ...     ('CIE_1931_Chromaticity_Diagram'
    ...     '_CIE_1931_2_Degree_Standard_Observer.png'))
    >>> image = read_image(path)  # doctest: +SKIP
    >>> image_plot(image)  # doctest: +SKIP
    """

    image = np.asarray(image)

    pylab.imshow(np.clip(image, 0, 1),
                 interpolation=interpolation,
                 cmap=colour_map)

    height = image.shape[0]

    pylab.text(0 + label_size,
               height - label_size,
               label,
               color=label_colour if label_colour is not None else (1, 1, 1),
               alpha=label_alpha,
               fontsize=label_size)

    settings = {'x_ticker': False,
                'y_ticker': False,
                'no_axes': True,
                'bounding_box': (0, 1, 0, 1)}
    settings.update(kwargs)

    canvas(**settings)
    decorate(**settings)

    return display(**settings)
