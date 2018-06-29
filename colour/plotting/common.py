# -*- coding: utf-8 -*-
"""
Common Plotting
===============

Defines the common plotting objects:

-   :func:`colour.plotting.colour_style`
-   :func:`colour.plotting.colour_cycle`
-   :func:`colour.plotting.canvas`
-   :func:`colour.plotting.camera`
-   :func:`colour.plotting.decorate`
-   :func:`colour.plotting.boundaries`
-   :func:`colour.plotting.display`
-   :func:`colour.plotting.render`
-   :func:`colour.plotting.label_rectangles`
-   :func:`colour.plotting.equal_axes3d`
-   :func:`colour.plotting.single_colour_swatch_plot`
-   :func:`colour.plotting.multi_colour_swatch_plot`
-   :func:`colour.plotting.image_plot`
"""

from __future__ import division

import itertools
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from collections import namedtuple
from matplotlib.colors import LinearSegmentedColormap

from colour.colorimetry import CMFS, ILLUMINANTS_SPDS
from colour.models import RGB_COLOURSPACES, XYZ_to_RGB
from colour.utilities import Structure

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'COLOUR_STYLE_CONSTANTS', 'colour_style', 'XYZ_to_plotting_colourspace',
    'ColourSwatch', 'colour_cycle', 'canvas', 'camera', 'boundaries',
    'decorate', 'display', 'render', 'label_rectangles', 'equal_axes3d',
    'get_RGB_colourspace', 'get_cmfs', 'get_illuminant',
    'single_colour_swatch_plot', 'multi_colour_swatch_plot', 'image_plot'
]

COLOUR_STYLE_CONSTANTS = Structure(
    **{
        'figure_width':
            12.80,
        'figure_height':
            7.20,
        'figure_size': (
            12.80,
            7.20,
        ),
        'figure_dpi':
            100,
        'font_size':
            12,
        'colour_cycle': (
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
        'hatch_patterns': (
            '\\\\',
            'o',
            'x',
            '.',
            '*',
            '//',
        ),
        'darkest_colour':
            '#111111',
        'dark_colour':
            '#333333',
        'average_colour':
            '#D5D5D5',
        'light_colour':
            '#F0F0F0',
        'lightest_colour':
            '#F5F5F5',
        'high_opacity':
            0.75,
        'low_opacity':
            0.25,
        'size':
            10,
        'width':
            1,
        'colourspace':
            RGB_COLOURSPACES['sRGB']
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
        'figure.figsize': constants.figure_size,
        'figure.dpi': constants.figure_dpi,
        'savefig.dpi': constants.figure_dpi,

        # Font Settings
        'font.size': constants.font_size,
        'axes.titlesize': constants.font_size * 1.2,
        'axes.labelsize': constants.font_size * 1.1,
        'legend.fontsize': constants.font_size * 0.9,
        'xtick.labelsize': constants.font_size,
        'ytick.labelsize': constants.font_size,

        # Tick Settings
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': constants.size * 1.5,
        'xtick.minor.size': constants.size * 0.75,
        'ytick.major.size': constants.size * 1.5,
        'ytick.minor.size': constants.size * 0.75,
        'xtick.major.width': constants.width,
        'xtick.minor.width': constants.width,
        'ytick.major.width': constants.width,
        'ytick.minor.width': constants.width,

        # Spine Settings
        'axes.linewidth': constants.width,
        'axes.edgecolor': constants.dark_colour,

        # Title Settings
        'axes.titlepad': constants.font_size * 0.75,

        # Axes Settings
        'axes.facecolor': constants.lightest_colour,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.grid.axis': 'both',

        # Grid Settings
        'axes.axisbelow': True,
        'grid.linewidth': constants.width * 0.5,
        'grid.linestyle': '--',
        'grid.color': constants.average_colour,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': constants.high_opacity,
        'legend.fancybox': False,
        'legend.facecolor': constants.light_colour,
        'legend.borderpad': constants.width * 0.5,

        # Lines
        'lines.linewidth': constants.width,

        # Cycle
        'axes.prop_cycle': matplotlib.cycler(color=constants.colour_cycle),
    }

    if use_style:
        plt.rcParams.update(style)

    return style


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

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are normalised to domain [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_plotting_colourspace(XYZ)  # doctest: +ELLIPSIS
    array([ 0.1749881...,  0.3881947...,  0.3216031...])
    """

    return XYZ_to_RGB(XYZ, illuminant,
                      COLOUR_STYLE_CONSTANTS.colourspace.whitepoint,
                      COLOUR_STYLE_CONSTANTS.colourspace.XYZ_to_RGB_matrix,
                      chromatic_adaptation_transform,
                      COLOUR_STYLE_CONSTANTS.colourspace.encoding_cctf
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
            'colour_cycle_map': 'hsv',
            'colour_cycle_count': len(COLOUR_STYLE_CONSTANTS.colour_cycle)
        })
    settings.update(kwargs)

    if settings.colour_cycle_map is None:
        cycle = COLOUR_STYLE_CONSTANTS.colour_cycle
    else:
        samples = np.linspace(0, 1, settings.colour_cycle_count)
        if isinstance(settings.colour_cycle_map, LinearSegmentedColormap):
            cycle = settings.colour_cycle_map(samples)
        else:
            cycle = getattr(plt.cm,
                            settings.colour_cycle_map)(samples)

    return itertools.cycle(cycle)


def canvas(**kwargs):
    """
    Sets the figure size.

    Other Parameters
    ----------------
    figure_size : array_like, optional
        Array defining figure *width* and *height* such as
        *figure_size = (width, height)*.

    Returns
    -------
    Figure
        Current figure.
    """

    settings = Structure(**{'figure_size': COLOUR_STYLE_CONSTANTS.figure_size})
    settings.update(kwargs)

    figure = plt.gcf()
    if figure is None:
        figure = plt.figure(figsize=settings.figure_size)
    else:
        figure.set_size_inches(settings.figure_size)

    return figure


def camera(**kwargs):
    """
    Sets the camera settings.

    Other Parameters
    ----------------
    camera_aspect : unicode, optional
        Matplotlib axes aspect. Default is *equal*.
    elevation : numeric, optional
        Camera elevation.
    azimuth : numeric, optional
        Camera azimuth.

    Returns
    -------
    Axes
        Current axes.
    """

    settings = Structure(**{
        'camera_aspect': 'equal',
        'elevation': None,
        'azimuth': None
    })
    settings.update(kwargs)

    axes = plt.gca()
    if settings.camera_aspect == 'equal':
        equal_axes3d(axes)

    axes.view_init(elev=settings.elevation, azim=settings.azimuth)

    return axes


def boundaries(**kwargs):
    """
    Sets the plot boundaries.

    Other Parameters
    ----------------
    bounding_box : array_like, optional
        Array defining current axes limits such
        `bounding_box = (x min, x max, y min, y max)`.
    x_tighten : bool, optional
        Whether to tighten the *X* axis limit. Default is *False*.
    y_tighten : bool, optional
        Whether to tighten the *Y* axis limit. Default is *False*.
    limits : array_like, optional
        Array defining current axes limits such as
        *limits = (x limit min, x limit max, y limit min, y limit max)*.
        ``limits`` argument values are added to the ``margins`` argument values
        to define the final bounding box for the current axes.
    margins : array_like, optional
        Array defining current axes margins such as
        *margins = (x margin min, x margin max, y margin min, y margin max)*.
        ``margins`` argument values are added to the ``limits`` argument values
        to define the final bounding box for the current axes.

    Returns
    -------
    Axes
        Current axes.
    """

    settings = Structure(**{'bounding_box': None})
    settings.update(kwargs)

    axes = plt.gca()
    if settings.bounding_box:
        plt.xlim(settings.bounding_box[0], settings.bounding_box[1])
        plt.ylim(settings.bounding_box[2], settings.bounding_box[3])

    return axes


def decorate(**kwargs):
    """
    Sets the figure decorations.

    Other Parameters
    ----------------
    title : unicode, optional
        Figure title.
    x_label : unicode, optional
        *X* axis label.
    y_label : unicode, optional
        *Y* axis label.
    legend : bool, optional
        Whether to display the legend. Default is *False*.
    legend_columns : int, optional
        Number of columns in the legend. Default is *1*.
    legend_location : unicode, optional
        Matplotlib legend location. Default is *upper right*.
    x_ticker : bool, optional
        Whether to display the *X* axis ticker. Default is *True*.
    y_ticker : bool, optional
        Whether to display the *Y* axis ticker. Default is *True*.
    x_ticker_major_locator : Locator, optional
        Locator type for the *X* axis major ticker.
    y_ticker_major_locator : Locator, optional
        Locator type for the *Y* axis major ticker.
    x_ticker_minor_locator : Locator, optional
        Locator type for the *X* axis minor ticker.
    y_ticker_minor_locator : Locator, optional
        Locator type for the *Y* axis minor ticker.
    x_axis_line : bool, optional
        Whether to draw the *X* axis line. Default is *False*.
    y_axis_line : bool, optional
        Whether to draw the *Y* axis line. Default is *False*.
    aspect : unicode, optional
        Matplotlib axes aspect.
    no_axes : bool, optional
        Whether to turn off the axes. Default is *False*.


    Returns
    -------
    Axes
        Current axes.
    """

    settings = Structure(
        **{
            'title': None,
            'x_label': None,
            'y_label': None,
            'legend': False,
            'legend_columns': 1,
            'legend_location': 'upper right',
            'x_ticker': True,
            'y_ticker': True,
            'x_ticker_major_locator': None,
            'y_ticker_major_locator': None,
            'x_ticker_minor_locator': None,
            'y_ticker_minor_locator': None,
            'x_axis_line': False,
            'y_axis_line': False,
            'aspect': None,
            'no_axes': False
        })
    settings.update(kwargs)

    axes = plt.gca()
    if settings.title:
        plt.title(settings.title)
    if settings.x_label:
        plt.xlabel(settings.x_label)
    if settings.y_label:
        plt.ylabel(settings.y_label)
    if settings.legend:
        plt.legend(
            loc=settings.legend_location, ncol=settings.legend_columns)
    if settings.x_ticker:
        if settings.x_ticker_major_locator is not None:
            axes.xaxis.set_major_locator(settings.x_ticker_major_locator)
        if settings.x_ticker_minor_locator is not None:
            axes.xaxis.set_minor_locator(settings.x_ticker_minor_locator)
    else:
        axes.set_xticks([])
    if settings.y_ticker:
        if settings.y_ticker_major_locator is not None:
            axes.yaxis.set_major_locator(settings.y_ticker_major_locator)
        if settings.y_ticker_minor_locator is not None:
            axes.yaxis.set_minor_locator(settings.y_ticker_minor_locator)
    else:
        axes.set_yticks([])
    if settings.x_axis_line:
        plt.axvline(color=COLOUR_STYLE_CONSTANTS.dark_colour, linestyle='--')
    if settings.y_axis_line:
        plt.axhline(color=COLOUR_STYLE_CONSTANTS.dark_colour, linestyle='--')
    if settings.aspect:
        plt.axes().set_aspect(settings.aspect)
    if settings.no_axes:
        axes.set_axis_off()

    return axes


def display(**kwargs):
    """
    Sets the figure display.

    Other Parameters
    ----------------
    transparent_background : bool, optional
        Whether to turn off the background patch. Default is *False*.
    tight_layout : bool, optional
        Whether to use tight layout. Default is *False*.
    standalone : bool, optional
        Whether to show the figure.
    filename : unicode, optional
        Figure will be saved using given ``filename`` argument.

    Returns
    -------
    Figure
        Current figure or None.
    """

    settings = Structure(
        **{
            'transparent_background': False,
            'tight_layout': False,
            'standalone': True,
            'filename': None
        })
    settings.update(kwargs)

    figure = plt.gcf()

    if settings.transparent_background:
        figure.patch.set_visible(False)
    if settings.tight_layout:
        figure.tight_layout()
    if settings.standalone:
        if settings.filename is not None:
            plt.savefig(settings.filename)
        else:
            plt.show()
        plt.close()

        return None
    else:
        return figure


def render(with_boundaries=True, with_decorate=True, **kwargs):
    """
    Convenient wrapper definition combining :func:`colour.plotting.decorate`,
    :func:`colour.plotting.boundaries` and :func:`colour.plotting.display`
    definitions.

    Parameters
    ----------
    with_boundaries : bool, optional
        Whether to call :func:`colour.plotting.boundaries` definition.
    with_decorate : bool, optional
        Whether to call :func:`colour.plotting.decorate` definition.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.
    """

    if with_boundaries:
        boundaries(**kwargs)

    if with_decorate:
        decorate(**kwargs)

    return display(**kwargs)


def label_rectangles(labels,
                     rectangles,
                     rotation='vertical',
                     text_size=10,
                     offset=None):
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

    for i, rectangle in enumerate(rectangles):
        x = rectangle.get_x()
        height = rectangle.get_height()
        width = rectangle.get_width()
        ha = 'center'
        va = 'bottom'
        plt.text(
            x + width / 2 + offset[0] * width,
            height + offset[1] * y_m,
            '{0:.1f}'.format(labels[i]),
            ha=ha,
            va=va,
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
    extents = np.array(
        [getattr(axes, 'get_{}lim'.format(axis))() for axis in 'xyz'])

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
             '"{1}".').format(name,
                              ', '.join(sorted(RGB_COLOURSPACES.keys()))))

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

    illuminant, name = ILLUMINANTS_SPDS.get(illuminant), illuminant
    if illuminant is None:
        raise KeyError('"{0}" not found in factory illuminants: "{1}".'.format(
            name, ', '.join(sorted(ILLUMINANTS_SPDS.keys()))))

    return illuminant


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.
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
    Figure
        Current figure or None.

    Examples
    --------
    >>> RGB = ColourSwatch(RGB=(0.32315746, 0.32983556, 0.33640183))
    >>> single_colour_swatch_plot(RGB)  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_Colour_Swatch_Plot.png
        :align: center
        :alt: single_colour_swatch_plot
    """

    return multi_colour_swatch_plot((colour_swatch, ), **kwargs)


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> RGB_1 = ColourSwatch(RGB=(0.45293517, 0.31732158, 0.26414773))
    >>> RGB_2 = ColourSwatch(RGB=(0.77875824, 0.57726450, 0.50453169))
    >>> multi_colour_swatch_plot([RGB_1, RGB_2])  # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_Colour_Swatch_Plot.png
        :align: center
        :alt: multi_colour_swatch_plot
    """

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
        'visible': True,
        'offset': 0.05,
    }
    if text_parameters is not None:
        text_settings.update(text_parameters)
    text_offset = text_settings.pop('offset')

    canvas(**kwargs)

    offset_X = offset_Y = 0
    x_min, x_max, y_min, y_max = 0, width, 0, height
    for i, colour_swatch in enumerate(reference_colour_swatches):
        if i % columns == 0 and i != 0:
            offset_X = 0
            offset_Y -= height + spacing

        x_0, x_1 = offset_X, offset_X + width
        y_0, y_1 = offset_Y, offset_Y + height

        plt.fill(
            (x_0, x_1, x_1, x_0), (y_0, y_0, y_1, y_1),
            color=reference_colour_swatches[i].RGB)

        if compare_swatches == 'stacked':
            margin_X = width * 0.25
            margin_Y = height * 0.25
            plt.fill(
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
            plt.fill(
                (x_0, x_1, x_1), (y_0, y_0, y_1),
                color=test_colour_swatches[i].RGB)

        if colour_swatch.name is not None and text_settings['visible']:
            plt.text(
                x_0 + text_offset,
                y_0 + text_offset,
                colour_swatch.name,
                clip_on=True,
                **text_settings)

        offset_X += width + spacing

    x_max = min(len(colour_swatches), columns)
    x_max = x_max * width + x_max * spacing - spacing
    y_min = offset_Y

    plt.gca().patch.set_facecolor(background_colour)

    settings = {
        'x_ticker':
            False,
        'y_ticker':
            False,
        'bounding_box': (x_min - spacing, x_max + spacing, y_min - spacing,
                         y_max + spacing),
        'aspect':
            'equal'
    }
    settings.update(kwargs)

    return render(**settings)


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

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

    text_settings = {
        'text': None,
        'offset': 5,
        'color': COLOUR_STYLE_CONSTANTS.lightest_colour,
        'alpha': COLOUR_STYLE_CONSTANTS.high_opacity
    }
    if text_parameters is not None:
        text_settings.update(text_parameters)
    text_offset = text_settings.pop('offset')

    image = np.asarray(image)

    plt.imshow(
        np.clip(image, 0, 1), interpolation=interpolation, cmap=colour_map)

    height = image.shape[0]

    if text_settings['text'] is not None:
        plt.text(text_offset / 2, height - text_offset,
                 text_settings['text'], **text_settings)

    settings = {
        'x_ticker': False,
        'y_ticker': False,
        'no_axes': True,
        'bounding_box': (0, 1, 0, 1)
    }
    settings.update(kwargs)

    canvas(**settings)

    return render(with_boundaries=False, **settings)
