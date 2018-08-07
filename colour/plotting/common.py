# -*- coding: utf-8 -*-
"""
Common Plotting
===============

Defines the common plotting objects:

-   :func:`colour.plotting.colour_style`
-   :func:`colour.plotting.override_style`
-   :func:`colour.plotting.colour_cycle`
-   :func:`colour.plotting.artist`
-   :func:`colour.plotting.camera`
-   :func:`colour.plotting.decorate`
-   :func:`colour.plotting.boundaries`
-   :func:`colour.plotting.display`
-   :func:`colour.plotting.render`
-   :func:`colour.plotting.label_rectangles`
-   :func:`colour.plotting.uniform_axes3d`
-   :func:`colour.plotting.single_colour_swatch_plot`
-   :func:`colour.plotting.multi_colour_swatch_plot`
-   :func:`colour.plotting.image_plot`
"""

from __future__ import division

import functools
import itertools
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import re
from collections import namedtuple
from matplotlib.colors import LinearSegmentedColormap

from colour.colorimetry import (
    CMFS, ILLUMINANTS_SPDS, LMS_ConeFundamentals, RGB_ColourMatchingFunctions,
    SpectralPowerDistribution, XYZ_ColourMatchingFunctions)
from colour.models import RGB_COLOURSPACES, RGB_Colourspace, XYZ_to_RGB
from colour.utilities import Structure

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'COLOUR_STYLE_CONSTANTS', 'colour_style', 'override_style',
    'XYZ_to_plotting_colourspace', 'ColourSwatch', 'colour_cycle', 'artist',
    'camera', 'render', 'label_rectangles', 'uniform_axes3d',
    'filter_RGB_colourspaces', 'filter_cmfs', 'filter_illuminants',
    'single_colour_swatch_plot', 'multi_colour_swatch_plot', 'image_plot'
]

COLOUR_STYLE_CONSTANTS = Structure(
    **{
        'colour':
            Structure(
                **{
                    'darkest':
                        '#111111',
                    'dark':
                        '#333333',
                    'average':
                        '#D5D5D5',
                    'bright':
                        '#F0F0F0',
                    'brightest':
                        '#F5F5F5',
                    'cycle': (
                        '#F44336',
                        '#9C27B0',
                        '#3F51B5',
                        '#03A9F4',
                        '#009688',
                        '#8BC34A',
                        '#FFEB3B',
                        '#FF9800',
                        '#795548',
                        '#607D8B',
                    ),
                    'map':
                        LinearSegmentedColormap.from_list(
                            'colour', (
                                '#F44336',
                                '#9C27B0',
                                '#3F51B5',
                                '#03A9F4',
                                '#009688',
                                '#8BC34A',
                                '#FFEB3B',
                                '#FF9800',
                                '#795548',
                                '#607D8B',
                            )),
                    'colourspace':
                        RGB_COLOURSPACES['sRGB']
                }),
        'opacity':
            Structure(**{
                'high': 0.75,
                'low': 0.25
            }),
        'hatch':
            Structure(**{'patterns': (
                '\\\\',
                'o',
                'x',
                '.',
                '*',
                '//',
            )}),
        'geometry':
            Structure(**{
                'long': 5,
                'short': 1
            })
    })
"""
Various defaults settings used across the plotting sub-package.

COLOUR_STYLE_CONSTANTS : Structure
"""


def colour_style(use_style=True):
    """
    Returns *Colour* plotting style.

    Parameters
    ----------
    use_style : bool, optional
        Whether to use the style and load it into *Matplotlib*.

    Returns
    -------
    dict
        *Colour* style.
    """

    constants = COLOUR_STYLE_CONSTANTS
    style = {
        # Figure Size Settings
        'figure.figsize': (12.80, 7.20),
        'figure.dpi': 100,
        'savefig.dpi': 100,
        'savefig.bbox': 'standard',

        # Font Settings
        'font.size': 12,
        'axes.titlesize': 'x-large',
        'axes.labelsize': 'larger',
        'legend.fontsize': 'small',
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium',

        # Text Settings
        'text.color': constants.colour.darkest,

        # Tick Settings
        'xtick.top': False,
        'xtick.bottom': True,
        'ytick.right': False,
        'ytick.left': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': constants.geometry.long * 1.25,
        'xtick.minor.size': constants.geometry.long * 0.75,
        'ytick.major.size': constants.geometry.long * 1.25,
        'ytick.minor.size': constants.geometry.long * 0.75,
        'xtick.major.width': constants.geometry.short,
        'xtick.minor.width': constants.geometry.short,
        'ytick.major.width': constants.geometry.short,
        'ytick.minor.width': constants.geometry.short,

        # Spine Settings
        'axes.linewidth': constants.geometry.short,
        'axes.edgecolor': constants.colour.dark,

        # Title Settings
        'axes.titlepad': 12 * 0.75,

        # Axes Settings
        'axes.facecolor': constants.colour.brightest,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.grid.axis': 'both',

        # Grid Settings
        'axes.axisbelow': True,
        'grid.linewidth': constants.geometry.short * 0.5,
        'grid.linestyle': '--',
        'grid.color': constants.colour.average,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': constants.opacity.high,
        'legend.fancybox': False,
        'legend.facecolor': constants.colour.bright,
        'legend.borderpad': constants.geometry.short * 0.5,

        # Lines
        'lines.linewidth': constants.geometry.short,
        'lines.markersize': constants.geometry.short * 3,

        # Cycle
        'axes.prop_cycle': matplotlib.cycler(color=constants.colour.cycle),
    }

    if use_style:
        plt.rcParams.update(style)

    return style


def override_style(**kwargs):
    """
    Decorator for overriding *Matplotlib* style.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    object

    Examples
    --------
    >>> @override_style(**{'text.color': 'red'})
    ... def f():
    ...     plt.text(0.5, 0.5, 'This is a text!')
    ...     plt.show()
    >>> f()  # doctest: +SKIP
    """

    keyword_overrides = dict(kwargs)

    def wrapper(function):
        """
        Wrapper for given function.
        """

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            """
            Wrapped function.
            """

            keywords = dict(kwargs)
            keywords.update(keyword_overrides)

            style_overrides = {
                key: value
                for key, value in keywords.items()
                if key in plt.rcParams.keys()
            }

            with plt.style.context(style_overrides):
                return function(*args, **kwargs)

        return wrapped

    return wrapper


def XYZ_to_plotting_colourspace(XYZ,
                                illuminant=RGB_COLOURSPACES['sRGB'].whitepoint,
                                chromatic_adaptation_transform='CAT02',
                                apply_encoding_cctf=True):
    """
    Converts from *CIE XYZ* tristimulus values to
    :attr:`colour.plotting.DEFAULT_PLOTTING_COLOURSPACE` colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like, optional
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    apply_encoding_cctf : bool, optional
        Apply :attr:`colour.plotting.DEFAULT_PLOTTING_COLOURSPACE` colourspace
        encoding colour component transfer function / opto-electronic transfer
        function.

    Returns
    -------
    ndarray
        :attr:`colour.plotting.DEFAULT_PLOTTING_COLOURSPACE` colourspace colour
        array.

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_plotting_colourspace(XYZ)  # doctest: +ELLIPSIS
    array([ 0.1749881...,  0.3881947...,  0.3216031...])
    """

    return XYZ_to_RGB(
        XYZ, illuminant, COLOUR_STYLE_CONSTANTS.colour.colourspace.whitepoint,
        COLOUR_STYLE_CONSTANTS.colour.colourspace.XYZ_to_RGB_matrix,
        chromatic_adaptation_transform,
        COLOUR_STYLE_CONSTANTS.colour.colourspace.encoding_cctf
        if apply_encoding_cctf else None)


class ColourSwatch(namedtuple('ColourSwatch', ('name', 'RGB'))):
    """
    Defines a data structure for a colour swatch.

    Parameters
    ----------
    name : unicode, optional
        Colour name.
    RGB : array_like, optional
        RGB Colour.
    """

    def __new__(cls, name=None, RGB=None):
        """
        Returns a new instance of the :class:`colour.plotting.ColourSwatch`
        class.
        """

        return super(ColourSwatch, cls).__new__(cls, name, RGB)


def colour_cycle(**kwargs):
    """
    Returns a colour cycle iterator using given colour map.

    Other Parameters
    ----------------
    colour_cycle_map : unicode or LinearSegmentedColormap, optional
        Matplotlib colourmap name.
    colour_cycle_count : int, optional
        Colours count to pick in the colourmap.

    Returns
    -------
    cycle
        Colour cycle iterator.
    """

    settings = Structure(
        **{
            'colour_cycle_map': COLOUR_STYLE_CONSTANTS.colour.map,
            'colour_cycle_count': len(COLOUR_STYLE_CONSTANTS.colour.cycle)
        })
    settings.update(kwargs)

    samples = np.linspace(0, 1, settings.colour_cycle_count)
    if isinstance(settings.colour_cycle_map, LinearSegmentedColormap):
        cycle = settings.colour_cycle_map(samples)
    else:
        cycle = getattr(plt.cm, settings.colour_cycle_map)(samples)

    return itertools.cycle(cycle)


def artist(**kwargs):
    """
    Returns the current figure and its axes or creates a new one.

    Other Parameters
    ----------------
    axes : Axes, optional
        Axes that will be passed through without creating a new figure.
    uniform : unicode, optional
        Whether to create the figure with an equal aspect ratio.

    Returns
    -------
    tuple
        Figure, axes.
    """

    width, height = plt.rcParams['figure.figsize']

    figure_size = (width, width) if kwargs.get('uniform') else (width, height)

    axes = kwargs.get('axes')
    if axes is None:
        figure = plt.figure(figsize=figure_size)

        return figure, figure.gca()
    else:
        return plt.gcf(), axes


def camera(**kwargs):
    """
    Sets the camera settings.

    Other Parameters
    ----------------
    azimuth : numeric, optional
        Camera azimuth.
    camera_aspect : unicode, optional
        Matplotlib axes aspect. Default is *equal*.
    elevation : numeric, optional
        Camera elevation.

    Returns
    -------
    Axes
        Current axes.
    """

    axes = kwargs.get('axes', plt.gca())

    settings = Structure(**{
        'camera_aspect': 'equal',
        'elevation': None,
        'azimuth': None
    })
    settings.update(kwargs)

    if settings.camera_aspect == 'equal':
        uniform_axes3d(axes)

    axes.view_init(elev=settings.elevation, azim=settings.azimuth)

    return axes


def render(**kwargs):
    """
    Renders the current figure while adjusting various settings such as the
    bounding box, the title or background transparency.

    Other Parameters
    ----------------
    figure : Figure, optional
        Figure to apply the render elements onto.
    axes : Axes, optional
        Axes to apply the render elements onto.
    filename : unicode, optional
        Figure will be saved using given ``filename`` argument.
    standalone : bool, optional
        Whether to show the figure and call :func:`plt.show` definition.
    aspect : unicode, optional
        Matplotlib axes aspect.
    axes_visible : bool, optional
        Whether the axes are visible. Default is *True*.
    bounding_box : array_like, optional
        Array defining current axes limits such
        `bounding_box = (x min, x max, y min, y max)`.
    legend : bool, optional
        Whether to display the legend. Default is *False*.
    legend_columns : int, optional
        Number of columns in the legend. Default is *1*.
    transparent_background : bool, optional
        Whether to turn off the background patch. Default is *False*.
    title : unicode, optional
        Figure title.
    x_label : unicode, optional
        *X* axis label.
    y_label : unicode, optional
        *Y* axis label.

    Returns
    -------
    tuple
        Current figure and axes.
    """

    figure = kwargs.get('figure')
    if figure is None:
        figure = plt.gcf()

    axes = kwargs.get('axes')
    if axes is None:
        axes = plt.gca()

    settings = Structure(
        **{
            'filename': None,
            'standalone': True,
            'aspect': None,
            'axes_visible': True,
            'bounding_box': None,
            'legend': False,
            'legend_columns': 1,
            'transparent_background': True,
            'title': None,
            'x_label': None,
            'y_label': None,
        })
    settings.update(kwargs)

    if settings.aspect:
        axes.set_aspect(settings.aspect)
    if not settings.axes_visible:
        axes.set_axis_off()
    if settings.bounding_box:
        axes.set_xlim(settings.bounding_box[0], settings.bounding_box[1])
        axes.set_ylim(settings.bounding_box[2], settings.bounding_box[3])

    if settings.title:
        axes.set_title(settings.title)
    if settings.x_label:
        axes.set_xlabel(settings.x_label)
    if settings.y_label:
        axes.set_ylabel(settings.y_label)
    if settings.legend:
        axes.legend(ncol=settings.legend_columns)

    if settings.transparent_background:
        figure.patch.set_alpha(0)
    if settings.standalone:
        if settings.filename is not None:
            figure.savefig(settings.filename)
        else:
            figure.show()

    return figure, axes


def label_rectangles(labels,
                     rectangles,
                     rotation='vertical',
                     text_size=10,
                     offset=None,
                     **kwargs):
    """
    Add labels above given rectangles.

    Parameters
    ----------
    labels : array_like
        Labels to display.
    rectangles : object
        Rectangles to used to set the labels value and position.
    rotation : unicode, optional
        **{'horizontal', 'vertical'}**,
        Labels orientation.
    text_size : numeric, optional
        Labels text size.
    offset : array_like, optional
        Labels offset as percentages of the largest rectangle dimensions.

    Other Parameters
    ----------------
    axes : Axes, optional
        Axes to use for plotting.

    Returns
    -------
    bool
        Definition success.
    """

    axes = kwargs.get('axes')
    if axes is None:
        axes = plt.gca()

    if offset is None:
        offset = (0.0, 0.025)

    x_m, y_m = 0, 0
    for rectangle in rectangles:
        x_m = max(x_m, rectangle.get_width())
        y_m = max(y_m, rectangle.get_height())

    for i, rectangle in enumerate(rectangles):
        x = rectangle.get_x()
        height = rectangle.get_height()
        width = rectangle.get_width()
        ha = 'center'
        va = 'bottom'
        axes.text(
            x + width / 2 + offset[0] * width,
            height + offset[1] * y_m,
            '{0:.1f}'.format(labels[i]),
            ha=ha,
            va=va,
            rotation=rotation,
            fontsize=text_size,
            clip_on=True)

    return True


def uniform_axes3d(axes):
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

    extents = np.array(
        [getattr(axes, 'get_{}lim'.format(axis))() for axis in 'xyz'])

    centers = np.mean(extents, axis=1)
    extent = np.max(np.abs(extents[..., 1] - extents[..., 0]))

    for center, axis in zip(centers, 'xyz'):
        getattr(axes, 'set_{}lim'.format(axis))(center - extent / 2,
                                                center + extent / 2)

    return True


def filter_RGB_colourspaces(filterer, flags=re.IGNORECASE):
    """
    Returns the *RGB* colourspaces matching given filterer.

    Parameters
    ----------
    filterer : unicode or RGB_Colourspace
        *RGB* colourspace filterer or :class:`colour.RGB_Colourspace` class
        instance which will be passed through directly.
    flags : int, optional
        Regex flags.

    Returns
    -------
    list
        Filtered *RGB* colourspaces.
    """

    if isinstance(filterer, RGB_Colourspace):
        return [filterer]
    else:
        return [
            RGB_COLOURSPACES[colourspace] for colourspace in RGB_COLOURSPACES
            if re.search(filterer, colourspace, flags)
        ]


def filter_cmfs(filterer, flags=re.IGNORECASE):
    """
    Returns the colour matching functions matching given filterer.

    Parameters
    ----------
    filterer : unicode or LMS_ConeFundamentals or RGB_ColourMatchingFunctions \
or XYZ_ColourMatchingFunctions
        Colour matching functions filterer or
        :class:`colour.LMS_ConeFundamentals`,
        :class:`colour.RGB_ColourMatchingFunctions` or
        :class:`colour.XYZ_ColourMatchingFunctions` class instance which will
        be passed through directly.
    flags : int, optional
        Regex flags.

    Returns
    -------
    list
        Filtered colour matching functions.
    """

    if isinstance(filterer, (LMS_ConeFundamentals, RGB_ColourMatchingFunctions,
                             XYZ_ColourMatchingFunctions)):
        return [filterer]
    else:
        return [
            CMFS[cmfs] for cmfs in CMFS if re.search(filterer, cmfs, flags)
        ]


def filter_illuminants(filterer, flags=re.IGNORECASE):
    """
    Returns the illuminants matching given filterer.

    Parameters
    ----------
    filterer : unicode or SpectralPowerDistribution
        Colour matching functions filterer or
        :class:`colour.SpectralPowerDistribution` class instance which will
        be passed through directly.
    flags : int, optional
        Regex flags.

    Returns
    -------
    list
        Filtered illuminants.
    """

    if isinstance(filterer, SpectralPowerDistribution):
        return [filterer]
    else:
        return [
            ILLUMINANTS_SPDS[illuminant] for illuminant in ILLUMINANTS_SPDS
            if re.search(filterer, illuminant, flags)
        ]


@override_style(
    **{
        'axes.grid': False,
        'xtick.bottom': False,
        'ytick.left': False,
        'xtick.labelbottom': False,
        'ytick.labelleft': False,
    })
def single_colour_swatch_plot(colour_swatch, **kwargs):
    """
    Plots given colour swatch.

    Parameters
    ----------
    colour_swatch : ColourSwatch
        ColourSwatch.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    width : numeric, optional
        {:func:`colour.plotting.multi_colour_swatch_plot`},
        Colour swatch width.
    height : numeric, optional
        {:func:`colour.plotting.multi_colour_swatch_plot`},
        Colour swatch height.
    spacing : numeric, optional
        {:func:`colour.plotting.multi_colour_swatch_plot`},
        Colour swatches spacing.
    columns : int, optional
        {:func:`colour.plotting.multi_colour_swatch_plot`},
        Colour swatches columns count.
    text_parameters : dict, optional
        {:func:`colour.plotting.multi_colour_swatch_plot`},
        Parameters for the :func:`plt.text` definition, ``offset`` can be
        set to define the text offset.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> RGB = ColourSwatch(RGB=(0.32315746, 0.32983556, 0.33640183))
    >>> single_colour_swatch_plot(RGB)  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_Colour_Swatch_Plot.png
        :align: center
        :alt: single_colour_swatch_plot
    """

    return multi_colour_swatch_plot((colour_swatch, ), **kwargs)


@override_style(
    **{
        'axes.grid': False,
        'xtick.bottom': False,
        'ytick.left': False,
        'xtick.labelbottom': False,
        'ytick.labelleft': False,
    })
def multi_colour_swatch_plot(colour_swatches,
                             width=1,
                             height=1,
                             spacing=0,
                             columns=None,
                             text_parameters=None,
                             background_colour=(1.0, 1.0, 1.0),
                             compare_swatches=None,
                             **kwargs):
    """
    Plots given colours swatches.

    Parameters
    ----------
    colour_swatches : list
        Colour swatch sequence.
    width : numeric, optional
        Colour swatch width.
    height : numeric, optional
        Colour swatch height.
    spacing : numeric, optional
        Colour swatches spacing.
    columns : int, optional
        Colour swatches columns count, defaults to the colour swatch count or
        half of it if comparing.
    text_parameters : dict, optional
        Parameters for the :func:`plt.text` definition, ``visible`` can be
        set to make the text visible,``offset`` can be set to define the text
        offset.
    background_colour : array_like or unicode, optional
        Background colour.
    compare_swatches : unicode, optional
        **{None, 'Stacked', 'Diagonal'}**,
        Whether to compare the swatches, in which case the colour swatch
        count must be an even number with alternating reference colour swatches
        and test colour swatches. *Stacked* will draw the test colour swatch in
        the center of the reference colour swatch, *Diagonal* will draw
        the reference colour swatch in the upper left diagonal area and the
        test colour swatch in the bottom right diagonal area.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> RGB_1 = ColourSwatch(RGB=(0.45293517, 0.31732158, 0.26414773))
    >>> RGB_2 = ColourSwatch(RGB=(0.77875824, 0.57726450, 0.50453169))
    >>> multi_colour_swatch_plot([RGB_1, RGB_2])  # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_Colour_Swatch_Plot.png
        :align: center
        :alt: multi_colour_swatch_plot
    """

    figure, axes = artist(**kwargs)

    if compare_swatches is not None:
        assert len(colour_swatches) % 2 == 0, (
            'Cannot compare an odd number of colour swatches!')

        reference_colour_swatches = colour_swatches[0::2]
        test_colour_swatches = colour_swatches[1::2]
    else:
        reference_colour_swatches = test_colour_swatches = colour_swatches

    compare_swatches = str(compare_swatches).lower()

    if columns is None:
        columns = len(reference_colour_swatches)

    text_settings = {
        'offset': 0.05,
        'visible': True,
    }
    if text_parameters is not None:
        text_settings.update(text_parameters)
    text_offset = text_settings.pop('offset')

    offset_X = offset_Y = 0
    x_min, x_max, y_min, y_max = 0, width, 0, height
    for i, colour_swatch in enumerate(reference_colour_swatches):
        if i % columns == 0 and i != 0:
            offset_X = 0
            offset_Y -= height + spacing

        x_0, x_1 = offset_X, offset_X + width
        y_0, y_1 = offset_Y, offset_Y + height

        axes.fill(
            (x_0, x_1, x_1, x_0), (y_0, y_0, y_1, y_1),
            color=reference_colour_swatches[i].RGB)

        if compare_swatches == 'stacked':
            margin_X = width * 0.25
            margin_Y = height * 0.25
            axes.fill(
                (
                    x_0 + margin_X,
                    x_1 - margin_X,
                    x_1 - margin_X,
                    x_0 + margin_X,
                ), (
                    y_0 + margin_Y,
                    y_0 + margin_Y,
                    y_1 - margin_Y,
                    y_1 - margin_Y,
                ),
                color=test_colour_swatches[i].RGB)
        else:
            axes.fill(
                (x_0, x_1, x_1), (y_0, y_0, y_1),
                color=test_colour_swatches[i].RGB)

        if colour_swatch.name is not None and text_settings['visible']:
            axes.text(
                x_0 + text_offset,
                y_0 + text_offset,
                colour_swatch.name,
                clip_on=True,
                **text_settings)

        offset_X += width + spacing

    x_max = min(len(colour_swatches), columns)
    x_max = x_max * width + x_max * spacing - spacing
    y_min = offset_Y

    axes.patch.set_facecolor(background_colour)

    bounding_box = (x_min - spacing, x_max + spacing, y_min - spacing,
                    y_max + spacing)

    settings = {
        'axes': axes,
        'bounding_box': bounding_box,
        'aspect': 'equal',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def image_plot(image,
               text_parameters=None,
               interpolation='nearest',
               colour_map=matplotlib.cm.Greys_r,
               **kwargs):
    """
    Plots given image.

    Parameters
    ----------
    image : array_like
        Image to plot.
    text_parameters : dict, optional
        Parameters for the :func:`plt.text` definition, ``offset`` can be
        set to define the text offset.
    interpolation: unicode, optional
        **{'nearest', None, 'none', 'bilinear', 'bicubic', 'spline16',
        'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
        'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'}**
        Image display interpolation.
    colour_map: unicode, optional
        Colour map used to display single channel images.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> from colour import read_image
    >>> path = os.path.join(
    ...     colour.__path__[0], '..', 'docs', '_static', 'Logo_Medium_001.png')
    >>> image_plot(read_image(path))  # doctest: +SKIP

    .. image:: ../_static/Plotting_Image_Plot.png
        :align: center
        :alt: image_plot
    """

    figure, axes = artist(**kwargs)

    text_settings = {
        'text': None,
        'offset': 0.005,
        'color': COLOUR_STYLE_CONSTANTS.colour.brightest,
        'alpha': COLOUR_STYLE_CONSTANTS.opacity.high,
    }
    if text_parameters is not None:
        text_settings.update(text_parameters)
    text_offset = text_settings.pop('offset')

    image = np.asarray(image)

    axes.imshow(
        np.clip(image, 0, 1), interpolation=interpolation, cmap=colour_map)

    width, height = image.shape[1], image.shape[0]

    if text_settings['text'] is not None:
        axes.text(
            text_offset,
            text_offset,
            text_settings['text'],
            transform=axes.transAxes,
            ha='left',
            va='bottom',
            **text_settings)

    settings = {
        'axes': axes,
        'bounding_box': (0, width, 0, height),
        'axes_visible': False,
    }
    settings.update(kwargs)

    return render(**settings)
