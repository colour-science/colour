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
-   :func:`colour.plotting.plot_single_colour_swatch`
-   :func:`colour.plotting.plot_multi_colour_swatches`
-   :func:`colour.plotting.plot_single_function`
-   :func:`colour.plotting.plot_multi_functions`
-   :func:`colour.plotting.plot_image`
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
import textwrap
from collections import OrderedDict, namedtuple
from functools import partial
from matplotlib.colors import LinearSegmentedColormap

from colour.characterisation import COLOURCHECKERS
from colour.colorimetry import CMFS, ILLUMINANTS_SDS, LIGHT_SOURCES_SDS
from colour.models import RGB_COLOURSPACES, XYZ_to_RGB
from colour.utilities import (CaseInsensitiveMapping, Structure,
                              as_float_array, is_sibling, is_string,
                              filter_mapping, runtime_warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'COLOUR_STYLE_CONSTANTS', 'COLOUR_ARROW_STYLE', 'colour_style',
    'override_style', 'XYZ_to_plotting_colourspace', 'ColourSwatch',
    'colour_cycle', 'artist', 'camera', 'render', 'wrap_label',
    'label_rectangles', 'uniform_axes3d', 'filter_passthrough',
    'filter_RGB_colourspaces', 'filter_cmfs', 'filter_illuminants',
    'filter_colour_checkers', 'plot_single_colour_swatch',
    'plot_multi_colour_swatches', 'plot_single_function',
    'plot_multi_functions', 'plot_image'
]

COLOUR_STYLE_CONSTANTS = Structure(
    **{
        'colour':
            Structure(
                **{
                    'darkest':
                        '#111111',
                    'darker':
                        '#222222',
                    'dark':
                        '#333333',
                    'dim':
                        '#505050',
                    'average':
                        '#808080',
                    'light':
                        '#D5D5D5',
                    'bright':
                        '#EEEEEE',
                    'brighter':
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

COLOUR_ARROW_STYLE = Structure(
    **{
        'color': COLOUR_STYLE_CONSTANTS.colour.dark,
        'headwidth': COLOUR_STYLE_CONSTANTS.geometry.short * 4,
        'headlength': COLOUR_STYLE_CONSTANTS.geometry.long,
        'width': COLOUR_STYLE_CONSTANTS.geometry.short * 0.5,
        'shrink': COLOUR_STYLE_CONSTANTS.geometry.short * 0.1,
        'connectionstyle': 'arc3,rad=-0.2',
    })
"""
Annotation arrow settings used across the plotting sub-package.

COLOUR_ARROW_STYLE : Structure
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
        # 'font.size': 12,
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
        'axes.titlepad': plt.rcParams['font.size'] * 0.75,

        # Axes Settings
        'axes.facecolor': constants.colour.brightest,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.grid.axis': 'both',

        # Grid Settings
        'axes.axisbelow': True,
        'grid.linewidth': constants.geometry.short * 0.5,
        'grid.linestyle': '--',
        'grid.color': constants.colour.light,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': constants.opacity.high,
        'legend.fancybox': False,
        'legend.facecolor': constants.colour.brighter,
        'legend.borderpad': constants.geometry.short * 0.5,

        # Lines
        'lines.linewidth': constants.geometry.short,
        'lines.markersize': constants.geometry.short * 3,
        'lines.markeredgewidth': constants.geometry.short * 0.75,

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
    \\**kwargs : dict, optional
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
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_plotting_colourspace(XYZ)  # doctest: +ELLIPSIS
    array([ 0.7057393...,  0.1924826...,  0.2235416...])
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
    tight_layout : bool, optional
        Whether to invoke the :func:`plt.tight_layout` definition.
    legend : bool, optional
        Whether to display the legend. Default is *False*.
    legend_columns : int, optional
        Number of columns in the legend. Default is *1*.
    transparent_background : bool, optional
        Whether to turn off the background patch. Default is *False*.
    title : unicode, optional
        Figure title.
    wrap_title : unicode, optional
        Whether to wrap the figure title, the default is to wrap at a number
        of characters equal to the width of the figure multiplied by 10.
    x_label : unicode, optional
        *X* axis label.
    y_label : unicode, optional
        *Y* axis label.
    x_ticker : bool, optional
        Whether to display the *X* axis ticker. Default is *True*.
    y_ticker : bool, optional
        Whether to display the *Y* axis ticker. Default is *True*.

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
            'tight_layout': True,
            'legend': False,
            'legend_columns': 1,
            'transparent_background': True,
            'title': None,
            'wrap_title': True,
            'x_label': None,
            'y_label': None,
            'x_ticker': True,
            'y_ticker': True,
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
        title = settings.title
        if settings.wrap_title:
            title = wrap_label(settings.title,
                               int(plt.rcParams['figure.figsize'][0] * 10))
        axes.set_title(title)
    if settings.x_label:
        axes.set_xlabel(settings.x_label)
    if settings.y_label:
        axes.set_ylabel(settings.y_label)
    if not settings.x_ticker:
        axes.set_xticks([])
    if not settings.y_ticker:
        axes.set_yticks([])
    if settings.legend:
        axes.legend(ncol=settings.legend_columns)

    if settings.tight_layout:
        figure.tight_layout()

    if settings.transparent_background:
        figure.patch.set_alpha(0)
    if settings.standalone:
        if settings.filename is not None:
            figure.savefig(settings.filename)
        else:
            plt.show()

    return figure, axes


def wrap_label(label, wrap_length=60):
    """
    Wraps given label at given length.

    The intent of this definition is to wrap long titles so that they don't
    overflow the figure.

    Parameters
    ----------
    label : unicode
        Label to wrap.
    wrap_length : int, optional
        Length at which wrapping should occur.

    Returns
    -------
    unicode
        Wrapped label.

    Examples
    --------
    >>> wrap_label(  # doctest: +SKIP
    ...     'This is a very long figure title that would overflow the figure '
    ...     'container if it was not wrapped.')
    'This is a very long figure title that would overflow the figure \
container if it\\nwas not wrapped.'
    """

    if wrap_length is not None:
        return '\n'.join(textwrap.wrap(label, wrap_length))
    else:
        return label


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


def filter_passthrough(mapping,
                       filterers,
                       anchors=True,
                       allow_non_siblings=True,
                       flags=re.IGNORECASE):
    """
    Returns mapping objects matching given filterers while passing through
    class instances whose type is one of the mapping element types.

    This definition allows passing custom but compatible objects to the various
    plotting definitions that by default expect the key from a dataset element.
    For example, a typical call to :func:`colour.plotting.\
plot_multi_illuminant_sds` definition is as follows:

    >>> import colour
    >>> import colour.plotting
    >>> colour.plotting.plot_multi_illuminant_sds(['A'])
    ... # doctest: +SKIP

    But it is also possible to pass a custom spectral distribution as follows:

    >>> data = {
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360
    ... }
    >>> colour.plotting.plot_multi_illuminant_sds(
    ...     ['A', colour.SpectralDistribution(data)])
    ... # doctest: +SKIP

    Similarly, a typical call to :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931` definition is as follows:

    >>> colour.plotting.plot_planckian_locus_in_chromaticity_diagram_CIE1931(
    ...     ['A'])
    ... # doctest: +SKIP

    But it is also possible to pass a custom whitepoint as follows:

    >>> colour.plotting.plot_planckian_locus_in_chromaticity_diagram_CIE1931(
    ...     ['A', {'Custom': np.array([1 / 3 + 0.05, 1 / 3 + 0.05])}])
    ... # doctest: +SKIP

    Parameters
    ----------
    mapping : dict_like
        Mapping to filter.
    filterers : unicode or object or array_like
        Filterer or object class instance (which is passed through directly if
        its type is one of the mapping element types) or list
        of filterers.
    anchors : bool, optional
        Whether to use Regex line anchors, i.e. *^* and *$* are added,
        surrounding the filterers patterns.
    allow_non_siblings : bool, optional
        Whether to allow non-siblings to be also passed through.
    flags : int, optional
        Regex flags.

    Returns
    -------
    dict_like
        Filtered mapping.
    """

    if is_string(filterers):
        filterers = [filterers]
    elif not isinstance(filterers, (list, tuple)):
        filterers = [filterers]

    string_filterers = [
        filterer for filterer in filterers if is_string(filterer)
    ]

    object_filterers = [
        filterer for filterer in filterers if is_sibling(filterer, mapping)
    ]

    if allow_non_siblings:
        non_siblings = [
            filterer for filterer in filterers
            if filterer not in string_filterers and
            filterer not in object_filterers
        ]

        if non_siblings:
            runtime_warning(
                'Non-sibling elements are passed-through: "{0}"'.format(
                    non_siblings))

            object_filterers.extend(non_siblings)

    filtered_mapping = filter_mapping(mapping, string_filterers, anchors,
                                      flags)

    for filterer in object_filterers:
        if isinstance(filterer, (dict, OrderedDict, CaseInsensitiveMapping)):
            for key, value in filterer.items():
                filtered_mapping[key] = value
        else:
            try:
                name = filterer.name
            except AttributeError:
                try:
                    name = filterer.__name__
                except AttributeError:
                    name = str(id(filterer))

            filtered_mapping[name] = filterer

    return filtered_mapping


def filter_RGB_colourspaces(filterers,
                            anchors=True,
                            allow_non_siblings=True,
                            flags=re.IGNORECASE):
    """
    Returns the *RGB* colourspaces matching given filterers.

    Parameters
    ----------
    filterers : unicode or RGB_Colourspace or array_like
        Filterer or :class:`colour.RGB_Colourspace` class instance (which is
        passed through directly if its type is one of the mapping element
        types) or list of filterers.
    anchors : bool, optional
        Whether to use Regex line anchors, i.e. *^* and *$* are added,
        surrounding the filterers patterns.
    allow_non_siblings : bool, optional
        Whether to allow non-siblings to be also passed through.
    flags : int, optional
        Regex flags.

    Returns
    -------
    dict_like
        Filtered *RGB* colourspaces.
    """

    return filter_passthrough(RGB_COLOURSPACES, filterers, anchors,
                              allow_non_siblings, flags)


def filter_cmfs(filterers,
                anchors=True,
                allow_non_siblings=True,
                flags=re.IGNORECASE):
    """
    Returns the colour matching functions matching given filterers.

    Parameters
    ----------
    filterers : unicode or LMS_ConeFundamentals or \
RGB_ColourMatchingFunctions or XYZ_ColourMatchingFunctions or array_like
        Filterer or :class:`colour.LMS_ConeFundamentals`,
        :class:`colour.RGB_ColourMatchingFunctions` or
        :class:`colour.XYZ_ColourMatchingFunctions` class instance (which is
        passed through directly if its type is one of the mapping element
        types) or list of filterers.
    anchors : bool, optional
        Whether to use Regex line anchors, i.e. *^* and *$* are added,
        surrounding the filterers patterns.
    allow_non_siblings : bool, optional
        Whether to allow non-siblings to be also passed through.
    flags : int, optional
        Regex flags.

    Returns
    -------
    dict_like
        Filtered colour matching functions.
    """

    return filter_passthrough(CMFS, filterers, anchors, allow_non_siblings,
                              flags)


def filter_illuminants(filterers,
                       anchors=True,
                       allow_non_siblings=True,
                       flags=re.IGNORECASE):
    """
    Returns the illuminants matching given filterers.

    Parameters
    ----------
    filterers : unicode or SpectralDistribution or array_like
        Filterer or :class:`colour.SpectralDistribution` class instance
        (which is passed through directly if its type is one of the mapping
        element types) or list of filterers.
    anchors : bool, optional
        Whether to use Regex line anchors, i.e. *^* and *$* are added,
        surrounding the filterers patterns.
    allow_non_siblings : bool, optional
        Whether to allow non-siblings to be also passed through.
    flags : int, optional
        Regex flags.

    Returns
    -------
    dict_like
        Filtered illuminants.
    """

    illuminants = OrderedDict()

    illuminants.update(
        filter_passthrough(ILLUMINANTS_SDS, filterers, anchors,
                           allow_non_siblings, flags))

    illuminants.update(
        filter_passthrough(LIGHT_SOURCES_SDS, filterers, anchors,
                           allow_non_siblings, flags))

    return illuminants


def filter_colour_checkers(filterers,
                           anchors=True,
                           allow_non_siblings=True,
                           flags=re.IGNORECASE):
    """
    Returns the colour checkers matching given filterers.

    Parameters
    ----------
    filterers : unicode or ColourChecker or array_like
        Filterer or :class:`colour.characterisation.ColourChecker` class
        instance (which is passed through directly if its type is one of the
        mapping element types) or list of filterers.
    anchors : bool, optional
        Whether to use Regex line anchors, i.e. *^* and *$* are added,
        surrounding the filterers patterns.
    allow_non_siblings : bool, optional
        Whether to allow non-siblings to be also passed through.
    flags : int, optional
        Regex flags.

    Returns
    -------
    dict_like
        Filtered colour checkers.
    """

    return filter_passthrough(COLOURCHECKERS, filterers, anchors,
                              allow_non_siblings, flags)


@override_style(
    **{
        'axes.grid': False,
        'xtick.bottom': False,
        'ytick.left': False,
        'xtick.labelbottom': False,
        'ytick.labelleft': False,
    })
def plot_single_colour_swatch(colour_swatch, **kwargs):
    """
    Plots given colour swatch.

    Parameters
    ----------
    colour_swatch : ColourSwatch
        ColourSwatch.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_colour_swatches`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    width : numeric, optional
        {:func:`colour.plotting.plot_multi_colour_swatches`},
        Colour swatch width.
    height : numeric, optional
        {:func:`colour.plotting.plot_multi_colour_swatches`},
        Colour swatch height.
    spacing : numeric, optional
        {:func:`colour.plotting.plot_multi_colour_swatches`},
        Colour swatches spacing.
    columns : int, optional
        {:func:`colour.plotting.plot_multi_colour_swatches`},
        Colour swatches columns count.
    text_parameters : dict, optional
        {:func:`colour.plotting.plot_multi_colour_swatches`},
        Parameters for the :func:`plt.text` definition, ``offset`` can be
        set to define the text offset.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> RGB = ColourSwatch(RGB=(0.45620519, 0.03081071, 0.04091952))
    >>> plot_single_colour_swatch(RGB)  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Single_Colour_Swatch.png
        :align: center
        :alt: plot_single_colour_swatch
    """

    return plot_multi_colour_swatches((colour_swatch, ), **kwargs)


@override_style(
    **{
        'axes.grid': False,
        'xtick.bottom': False,
        'ytick.left': False,
        'xtick.labelbottom': False,
        'ytick.labelleft': False,
    })
def plot_multi_colour_swatches(colour_swatches,
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
    \\**kwargs : dict, optional
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
    >>> plot_multi_colour_swatches([RGB_1, RGB_2])  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Multi_Colour_Swatches.png
        :align: center
        :alt: plot_multi_colour_swatches
    """

    _figure, axes = artist(**kwargs)

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
def plot_single_function(function,
                         samples=None,
                         log_x=None,
                         log_y=None,
                         **kwargs):
    """
    Plots given function.

    Parameters
    ----------
    function : callable, optional
        Function to plot.
    samples : array_like, optional,
        Samples to evaluate the functions with.
    log_x : int, optional
        Log base to use for the *x* axis scale, if *None*, the *x* axis scale
        will be linear.
    log_y : int, optional
        Log base to use for the *y* axis scale, if *None*, the *y* axis scale
        will be linear.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_single_function(partial(gamma_function, exponent=1 / 2.2))
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Single_Function.png
        :align: center
        :alt: plot_single_function
    """

    try:
        name = function.__name__
    except AttributeError:
        name = 'Unnamed'

    settings = {
        'title': '{0} - Function'.format(name),
        'legend': False,
    }
    settings.update(kwargs)

    return plot_multi_functions({
        name: function
    }, samples, log_x, log_y, **settings)


@override_style()
def plot_multi_functions(functions,
                         samples=None,
                         log_x=None,
                         log_y=None,
                         **kwargs):
    """
    Plots given functions.

    Parameters
    ----------
    functions : dict
        Functions to plot.
    samples : array_like, optional,
        Samples to evaluate the functions with.
    log_x : int, optional
        Log base to use for the *x* axis scale, if *None*, the *x* axis scale
        will be linear.
    log_y : int, optional
        Log base to use for the *y* axis scale, if *None*, the *y* axis scale
        will be linear.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> functions = {
    ...     'Gamma 2.2' : lambda x: x ** (1 / 2.2),
    ...     'Gamma 2.4' : lambda x: x ** (1 / 2.4),
    ...     'Gamma 2.6' : lambda x: x ** (1 / 2.6),
    ... }
    >>> plot_multi_functions(functions)
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Multi_Functions.png
        :align: center
        :alt: plot_multi_functions
    """

    settings = {}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if log_x is not None and log_y is not None:
        assert log_x >= 2 and log_y >= 2, (
            'Log base must be equal or greater than 2.')

        plotting_function = partial(plt.loglog, basex=log_x, basey=log_y)
    elif log_x is not None:
        assert log_x >= 2, 'Log base must be equal or greater than 2.'

        plotting_function = partial(plt.semilogx, basex=log_x)
    elif log_y is not None:
        assert log_y >= 2, 'Log base must be equal or greater than 2.'

        plotting_function = partial(plt.semilogy, basey=log_y)
    else:
        plotting_function = plt.plot

    if samples is None:
        samples = np.linspace(0, 1, 1000)

    for name, function in functions.items():
        plotting_function(samples, function(samples), label='{0}'.format(name))

    x_label = ('x - Log Base {0} Scale'.format(log_x)
               if log_x is not None else 'x - Linear Scale')
    y_label = ('y - Log Base {0} Scale'.format(log_y)
               if log_y is not None else 'y - Linear Scale')
    settings = {
        'axes': axes,
        'legend': True,
        'title': '{0} - Functions'.format(', '.join(functions)),
        'x_label': x_label,
        'y_label': y_label
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_image(image,
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
    \\**kwargs : dict, optional
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
    >>> plot_image(read_image(path))  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Image.png
        :align: center
        :alt: plot_image
    """

    _figure, axes = artist(**kwargs)

    text_settings = {
        'text': None,
        'offset': 0.005,
        'color': COLOUR_STYLE_CONSTANTS.colour.brightest,
        'alpha': COLOUR_STYLE_CONSTANTS.opacity.high,
    }
    if text_parameters is not None:
        text_settings.update(text_parameters)
    text_offset = text_settings.pop('offset')

    image = as_float_array(image)

    axes.imshow(
        np.clip(image, 0, 1), interpolation=interpolation, cmap=colour_map)

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
        'axes_visible': False,
    }
    settings.update(kwargs)

    return render(**settings)
