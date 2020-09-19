# -*- coding: utf-8 -*-
"""
CIE Chromaticity Diagrams Plotting
==================================

Defines the *CIE* chromaticity diagrams plotting objects:

-   :func:`colour.plotting.plot_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.plot_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.plot_chromaticity_diagram_CIE1976UCS`
-   :func:`colour.plotting.plot_sds_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.plot_sds_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.plot_sds_in_chromaticity_diagram_CIE1976UCS`
"""

from __future__ import division

import bisect
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from colour.algebra import normalise_vector
from colour.colorimetry import SDS_ILLUMINANTS, sd_to_XYZ, sds_and_msds_to_sds
from colour.models import (Luv_to_uv, Luv_uv_to_xy, UCS_to_uv, UCS_uv_to_xy,
                           XYZ_to_Luv, XYZ_to_UCS, XYZ_to_xy, xy_to_XYZ)
from colour.plotting import (CONSTANTS_COLOUR_STYLE, CONSTANTS_ARROW_STYLE,
                             XYZ_to_plotting_colourspace, artist, filter_cmfs,
                             filter_illuminants, override_style, render,
                             update_settings_collection)
from colour.utilities import (domain_range_scale, first_item, is_string,
                              normalise_maximum, tstack, suppress_warnings)
from colour.utilities.deprecation import handle_arguments_deprecation

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'plot_spectral_locus', 'plot_chromaticity_diagram_colours',
    'plot_chromaticity_diagram', 'plot_chromaticity_diagram_CIE1931',
    'plot_chromaticity_diagram_CIE1960UCS',
    'plot_chromaticity_diagram_CIE1976UCS', 'plot_sds_in_chromaticity_diagram',
    'plot_sds_in_chromaticity_diagram_CIE1931',
    'plot_sds_in_chromaticity_diagram_CIE1960UCS',
    'plot_sds_in_chromaticity_diagram_CIE1976UCS'
]


@override_style()
def plot_spectral_locus(cmfs='CIE 1931 2 Degree Standard Observer',
                        spectral_locus_colours=None,
                        spectral_locus_labels=None,
                        method='CIE 1931',
                        **kwargs):
    """
    Plots the *Spectral Locus* according to given method.

    Parameters
    ----------
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    spectral_locus_colours : array_like or unicode, optional
        *Spectral Locus* colours, if ``spectral_locus_colours`` is set to
        *RGB*, the colours will be computed according to the corresponding
        chromaticity coordinates.
    spectral_locus_labels : array_like, optional
        Array of wavelength labels used to customise which labels will be drawn
        around the spectral locus. Passing an empty array will result in no
        wavelength labels being drawn.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

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
    >>> plot_spectral_locus(spectral_locus_colours='RGB')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Spectral_Locus.png
        :align: center
        :alt: plot_spectral_locus
    """

    if spectral_locus_colours is None:
        spectral_locus_colours = CONSTANTS_COLOUR_STYLE.colour.dark

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    cmfs = first_item(filter_cmfs(cmfs).values())

    illuminant = CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    if method == 'CIE 1931':
        ij = XYZ_to_xy(cmfs.values, illuminant)
        labels = ((390, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600,
                   620, 700)
                  if spectral_locus_labels is None else spectral_locus_labels)
    elif method == 'CIE 1960 UCS':
        ij = UCS_to_uv(XYZ_to_UCS(cmfs.values))
        labels = ((420, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
                   550, 560, 570, 580, 590, 600, 610, 620, 630, 645, 680)
                  if spectral_locus_labels is None else spectral_locus_labels)
    elif method == 'CIE 1976 UCS':
        ij = Luv_to_uv(XYZ_to_Luv(cmfs.values, illuminant), illuminant)
        labels = ((420, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
                   550, 560, 570, 580, 590, 600, 610, 620, 630, 645, 680)
                  if spectral_locus_labels is None else spectral_locus_labels)
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '[\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\']'.format(
                method))

    pl_ij = tstack([
        np.linspace(ij[0][0], ij[-1][0], 20),
        np.linspace(ij[0][1], ij[-1][1], 20)
    ]).reshape(-1, 1, 2)
    sl_ij = np.copy(ij).reshape(-1, 1, 2)

    if spectral_locus_colours.upper() == 'RGB':
        spectral_locus_colours = normalise_maximum(
            XYZ_to_plotting_colourspace(cmfs.values), axis=-1)

        if method == 'CIE 1931':
            XYZ = xy_to_XYZ(pl_ij)
        elif method == 'CIE 1960 UCS':
            XYZ = xy_to_XYZ(UCS_uv_to_xy(pl_ij))
        elif method == 'CIE 1976 UCS':
            XYZ = xy_to_XYZ(Luv_uv_to_xy(pl_ij))
        purple_line_colours = normalise_maximum(
            XYZ_to_plotting_colourspace(XYZ.reshape(-1, 3)), axis=-1)
    else:
        purple_line_colours = spectral_locus_colours

    for slp_ij, slp_colours in ((pl_ij, purple_line_colours),
                                (sl_ij, spectral_locus_colours)):
        line_collection = LineCollection(
            np.concatenate([slp_ij[:-1], slp_ij[1:]], axis=1),
            colors=slp_colours)
        axes.add_collection(line_collection)

    wl_ij = dict(tuple(zip(wavelengths, ij)))
    for label in labels:
        ij = wl_ij.get(label)

        if ij is None:
            continue

        i, j = ij
        ij = np.array([ij])

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else wavelengths[-1])

        dx = wl_ij[right][0] - wl_ij[left][0]
        dy = wl_ij[right][1] - wl_ij[left][1]

        direction = np.array([-dy, dx])

        normal = (np.array([-dy, dx]) if np.dot(
            normalise_vector(ij - equal_energy), normalise_vector(direction)) >
                  0 else np.array([dy, -dx]))
        normal = normalise_vector(normal) / 30

        label_colour = (spectral_locus_colours
                        if is_string(spectral_locus_colours) else
                        spectral_locus_colours[index])
        axes.plot(
            (i, i + normal[0] * 0.75), (j, j + normal[1] * 0.75),
            color=label_colour)

        axes.plot(i, j, 'o', color=label_colour)

        axes.text(
            i + normal[0],
            j + normal[1],
            label,
            clip_on=True,
            ha='left' if normal[0] >= 0 else 'right',
            va='center',
            fontdict={'size': 'small'})

    settings = {'axes': axes}
    settings.update(kwargs)

    return render(**kwargs)


@override_style()
def plot_chromaticity_diagram_colours(
        samples=256,
        diagram_opacity=1.0,
        diagram_clipping_path=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        method='CIE 1931',
        **kwargs):
    """
    Plots the *Chromaticity Diagram* colours according to given method.

    Parameters
    ----------
    samples : numeric, optional
        Samples count on one axis.
    diagram_opacity : numeric, optional
        Opacity of the *Chromaticity Diagram* colours.
    diagram_clipping_path : array_like, optional
        Path of points used to clip the *Chromaticity Diagram* colours.
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

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
    >>> plot_chromaticity_diagram_colours()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram_Colours.png
        :align: center
        :alt: plot_chromaticity_diagram_colours
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    cmfs = first_item(filter_cmfs(cmfs).values())

    illuminant = CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint

    ii, jj = np.meshgrid(
        np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    ij = tstack([ii, jj])

    # NOTE: Various values in the grid have potential to generate
    # zero-divisions, they could be avoided by perturbing the grid, e.g. adding
    # a small epsilon. It was decided instead to disable warnings.
    with suppress_warnings(python_warnings=True):
        if method == 'CIE 1931':
            XYZ = xy_to_XYZ(ij)
            spectral_locus = XYZ_to_xy(cmfs.values, illuminant)
        elif method == 'CIE 1960 UCS':
            XYZ = xy_to_XYZ(UCS_uv_to_xy(ij))
            spectral_locus = UCS_to_uv(XYZ_to_UCS(cmfs.values))
        elif method == 'CIE 1976 UCS':
            XYZ = xy_to_XYZ(Luv_uv_to_xy(ij))
            spectral_locus = Luv_to_uv(
                XYZ_to_Luv(cmfs.values, illuminant), illuminant)
        else:
            raise ValueError(
                'Invalid method: "{0}", must be one of '
                '[\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\']'.format(
                    method))

    RGB = normalise_maximum(
        XYZ_to_plotting_colourspace(XYZ, illuminant), axis=-1)

    polygon = Polygon(
        spectral_locus
        if diagram_clipping_path is None else diagram_clipping_path,
        facecolor='none',
        edgecolor='none')
    axes.add_patch(polygon)
    # Preventing bounding box related issues as per
    # https://github.com/matplotlib/matplotlib/issues/10529
    image = axes.imshow(
        RGB,
        interpolation='bilinear',
        extent=(0, 1, 0, 1),
        clip_path=None,
        alpha=diagram_opacity)
    image.set_clip_path(polygon)

    settings = {'axes': axes}
    settings.update(kwargs)

    return render(**kwargs)


@override_style()
def plot_chromaticity_diagram(cmfs='CIE 1931 2 Degree Standard Observer',
                              show_diagram_colours=True,
                              show_spectral_locus=True,
                              method='CIE 1931',
                              **kwargs):
    """
    Plots the *Chromaticity Diagram* according to given method.

    Parameters
    ----------
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    show_diagram_colours : bool, optional
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus : bool, optional
        Whether to display the *Spectral Locus*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_spectral_locus`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram_colours`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_chromaticity_diagram()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram.png
        :align: center
        :alt: plot_chromaticity_diagram
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    cmfs = first_item(filter_cmfs(cmfs).values())

    if show_diagram_colours:
        settings = {'axes': axes, 'method': method}
        settings.update(kwargs)
        settings['standalone'] = False
        settings['cmfs'] = cmfs

        plot_chromaticity_diagram_colours(**settings)

    if show_spectral_locus:
        settings = {'axes': axes, 'method': method}
        settings.update(kwargs)
        settings['standalone'] = False
        settings['cmfs'] = cmfs

        plot_spectral_locus(**settings)

    if method == 'CIE 1931':
        x_label, y_label = 'CIE x', 'CIE y'
    elif method == 'CIE 1960 UCS':
        x_label, y_label = 'CIE u', 'CIE v'
    elif method == 'CIE 1976 UCS':
        x_label, y_label = 'CIE u\'', 'CIE v\'',
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '[\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\']'.format(
                method))

    title = '{0} Chromaticity Diagram - {1}'.format(method, cmfs.strict_name)

    settings.update({
        'axes': axes,
        'standalone': True,
        'bounding_box': (0, 1, 0, 1),
        'title': title,
        'x_label': x_label,
        'y_label': y_label,
    })
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_chromaticity_diagram_CIE1931(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        show_spectral_locus=True,
        **kwargs):
    """
    Plots the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    show_diagram_colours : bool, optional
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus : bool, optional
        Whether to display the *Spectral Locus*.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_chromaticity_diagram_CIE1931()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return plot_chromaticity_diagram(cmfs, show_diagram_colours,
                                     show_spectral_locus, **settings)


@override_style()
def plot_chromaticity_diagram_CIE1960UCS(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        show_spectral_locus=True,
        **kwargs):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    show_diagram_colours : bool, optional
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus : bool, optional
        Whether to display the *Spectral Locus*.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_chromaticity_diagram_CIE1960UCS()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return plot_chromaticity_diagram(cmfs, show_diagram_colours,
                                     show_spectral_locus, **settings)


@override_style()
def plot_chromaticity_diagram_CIE1976UCS(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        show_spectral_locus=True,
        **kwargs):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    show_diagram_colours : bool, optional
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus : bool, optional
        Whether to display the *Spectral Locus*.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_chromaticity_diagram_CIE1976UCS()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return plot_chromaticity_diagram(cmfs, show_diagram_colours,
                                     show_spectral_locus, **settings)


@override_style()
def plot_sds_in_chromaticity_diagram(
        sds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable=plot_chromaticity_diagram,
        method='CIE 1931',
        annotate_kwargs=None,
        plot_kwargs=None,
        **kwargs):
    """
    Plots given spectral distribution chromaticity coordinates into the
    *Chromaticity Diagram* using given method.

    Parameters
    ----------
    sds : array_like or MultiSpectralDistributions
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single
        :class:`colour.MultiSpectralDistributions` class instance, a list
        of :class:`colour.MultiSpectralDistributions` class instances or a
        list of :class:`colour.SpectralDistribution` class instances.
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    chromaticity_diagram_callable : callable, optional
        Callable responsible for drawing the *Chromaticity Diagram*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.
    annotate_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.annotate` definition, used to
        annotate the resulting chromaticity coordinates with their respective
        spectral distribution names. ``annotate_kwargs`` can be either a single
        dictionary applied to all the arrows with same settings or a sequence
        of dictionaries with different settings for each spectral distribution.
        The following special keyword arguments can also be used:

        -   *annotate* : bool, whether to annotate the spectral distributions.
    plot_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.plot` definition, used to control
        the style of the plotted spectral distributions. ``plot_kwargs`` can be
        either a single dictionary applied to all the plotted spectral
        distributions with same settings or a sequence of dictionaries with
        different settings for each plotted spectral distributions.
        The following special keyword arguments can also be used:

        -   *illuminant* : unicode or :class:`colour.SpectralDistribution`, the
            illuminant used to compute the spectral distributions colours. The
            default is the illuminant associated with the whitepoint of the
            default plotting colourspace. ``illuminant`` can be of any type or
            form supported by the :func:`colour.plotting.filter_cmfs`
            definition.
        -   *cmfs* : unicode, the standard observer colour matching functions
            used for computing the spectral distributions colours. ``cmfs`` can
            be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   *normalise_sd_colours* : bool, whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   *use_sd_colours* : bool, whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the :func:`plt.plot`
            definition ``color`` argument with pre-computed values. The default
            is *True*.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
        Also handles keywords arguments for deprecation management.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> A = SDS_ILLUMINANTS['A']
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> annotate_kwargs = [
    ...     {'xytext': (-25, 15), 'arrowprops':{'arrowstyle':'-'}},
    ...     {}
    ... ]
    >>> plot_kwargs = [
    ...     {
    ...         'illuminant': SDS_ILLUMINANTS['E'],
    ...         'markersize' : 15,
    ...         'normalise_sd_colours': True,
    ...         'use_sd_colours': True
    ...     },
    ...     {'illuminant': SDS_ILLUMINANTS['E']},
    ... ]
    >>> plot_sds_in_chromaticity_diagram(
    ...     [A, D65], annotate_kwargs=annotate_kwargs, plot_kwargs=plot_kwargs)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_SDS_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_sds_in_chromaticity_diagram
    """

    annotate_kwargs = handle_arguments_deprecation({
        'ArgumentRenamed': [['annotate_parameters', 'annotate_kwargs']],
    }, **kwargs).get('annotate_kwargs', annotate_kwargs)

    sds = sds_and_msds_to_sds(sds)

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    settings.update({
        'axes': axes,
        'standalone': False,
        'method': method,
        'cmfs': cmfs,
    })

    chromaticity_diagram_callable(**settings)

    if method == 'CIE 1931':

        def XYZ_to_ij(XYZ):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return XYZ_to_xy(XYZ)

        bounding_box = (-0.1, 0.9, -0.1, 0.9)
    elif method == 'CIE 1960 UCS':

        def XYZ_to_ij(XYZ):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(XYZ))

        bounding_box = (-0.1, 0.7, -0.2, 0.6)

    elif method == 'CIE 1976 UCS':

        def XYZ_to_ij(XYZ):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return Luv_to_uv(XYZ_to_Luv(XYZ))

        bounding_box = (-0.1, 0.7, -0.1, 0.7)
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '[\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\']'.format(
                method))

    annotate_settings_collection = [{
        'annotate': True,
        'xytext': (-50, 30),
        'textcoords': 'offset points',
        'arrowprops': CONSTANTS_ARROW_STYLE,
    } for _ in range(len(sds))]

    if annotate_kwargs is not None:
        update_settings_collection(annotate_settings_collection,
                                   annotate_kwargs, len(sds))

    plot_settings_collection = [{
        'color':
            CONSTANTS_COLOUR_STYLE.colour.brightest,
        'label':
            '{0}'.format(sd.strict_name),
        'marker':
            'o',
        'markeredgecolor':
            CONSTANTS_COLOUR_STYLE.colour.dark,
        'markeredgewidth':
            CONSTANTS_COLOUR_STYLE.geometry.short * 0.75,
        'markersize': (CONSTANTS_COLOUR_STYLE.geometry.short * 6 +
                       CONSTANTS_COLOUR_STYLE.geometry.short * 0.75),
        'cmfs':
            cmfs,
        'illuminant':
            SDS_ILLUMINANTS[
                CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint_name
            ],
        'use_sd_colours':
            False,
        'normalise_sd_colours':
            False,
    } for sd in sds]

    if plot_kwargs is not None:
        update_settings_collection(plot_settings_collection, plot_kwargs,
                                   len(sds))

    for i, sd in enumerate(sds):
        plot_settings = plot_settings_collection[i]

        cmfs = first_item(filter_cmfs(plot_settings.pop('cmfs')).values())
        illuminant = first_item(
            filter_illuminants(plot_settings.pop('illuminant')).values())
        normalise_sd_colours = plot_settings.pop('normalise_sd_colours')
        use_sd_colours = plot_settings.pop('use_sd_colours')

        with domain_range_scale('1'):
            XYZ = sd_to_XYZ(sd, cmfs, illuminant)

        if use_sd_colours:
            if normalise_sd_colours:
                XYZ /= XYZ[..., 1]

            plot_settings['color'] = np.clip(
                XYZ_to_plotting_colourspace(XYZ), 0, 1)

        ij = XYZ_to_ij(XYZ)

        axes.plot(ij[0], ij[1], **plot_settings)

        if (sd.name is not None and
                annotate_settings_collection[i]['annotate']):
            annotate_settings = annotate_settings_collection[i]
            annotate_settings.pop('annotate')

            axes.annotate(sd.name, xy=ij, **annotate_settings)

    settings.update({'standalone': True, 'bounding_box': bounding_box})
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_sds_in_chromaticity_diagram_CIE1931(
        sds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1931=(
            plot_chromaticity_diagram_CIE1931),
        annotate_kwargs=None,
        plot_kwargs=None,
        **kwargs):
    """
    Plots given spectral distribution chromaticity coordinates into the
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    sds : array_like or MultiSpectralDistributions
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single :class:`colour.MultiSpectralDistributions`
        class instance, a list of :class:`colour.MultiSpectralDistributions`
        class instances or a list of :class:`colour.SpectralDistribution` class
        instances.
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    annotate_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.annotate` definition, used to
        annotate the resulting chromaticity coordinates with their respective
        spectral distribution names. ``annotate_kwargs`` can be either a single
        dictionary applied to all the arrows with same settings or a sequence
        of dictionaries with different settings for each spectral distribution.
        The following special keyword arguments can also be used:

        -   *annotate* : bool, whether to annotate the spectral distributions.
    plot_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.plot` definition, used to control
        the style of the plotted spectral distributions. ``plot_kwargs`` can be
        either a single dictionary applied to all the plotted spectral
        distributions with same settings or a sequence of dictionaries with
        different settings for each plotted spectral distributions.
        The following special keyword arguments can also be used:

        -   *illuminant* : unicode or :class:`colour.SpectralDistribution`, the
            illuminant used to compute the spectral distributions colours. The
            default is the illuminant associated with the whitepoint of the
            default plotting colourspace. ``illuminant`` can be of any type or
            form supported by the :func:`colour.plotting.filter_cmfs`
            definition.
        -   *cmfs* : unicode, the standard observer colour matching functions
            used for computing the spectral distributions colours. ``cmfs`` can
            be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   *normalise_sd_colours* : bool, whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   *use_sd_colours* : bool, whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the :func:`plt.plot`
            definition ``color`` argument with pre-computed values. The default
            is *True*.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
        Also handles keywords arguments for deprecation management.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> A = SDS_ILLUMINANTS['A']
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> plot_sds_in_chromaticity_diagram_CIE1931([A, D65])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_SDS_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_sds_in_chromaticity_diagram_CIE1931
    """

    annotate_kwargs = handle_arguments_deprecation({
        'ArgumentRenamed': [['annotate_parameters', 'annotate_kwargs']],
    }, **kwargs).get('annotate_kwargs', annotate_kwargs)

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return plot_sds_in_chromaticity_diagram(
        sds,
        cmfs,
        chromaticity_diagram_callable_CIE1931,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings)


@override_style()
def plot_sds_in_chromaticity_diagram_CIE1960UCS(
        sds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1960UCS=(
            plot_chromaticity_diagram_CIE1960UCS),
        annotate_kwargs=None,
        plot_kwargs=None,
        **kwargs):
    """
    Plots given spectral distribution chromaticity coordinates into the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    sds : array_like or MultiSpectralDistributions
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single :class:`colour.MultiSpectralDistributions`
        class instance, a list of :class:`colour.MultiSpectralDistributions`
        class instances or a list of :class:`colour.SpectralDistribution` class
        instances.
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.
    annotate_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.annotate` definition, used to
        annotate the resulting chromaticity coordinates with their respective
        spectral distribution names. ``annotate_kwargs`` can be either a single
        dictionary applied to all the arrows with same settings or a sequence
        of dictionaries with different settings for each spectral distribution.
        The following special keyword arguments can also be used:

        -   *annotate* : bool, whether to annotate the spectral distributions.
    plot_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.plot` definition, used to control
        the style of the plotted spectral distributions. ``plot_kwargs`` can be
        either a single dictionary applied to all the plotted spectral
        distributions with same settings or a sequence of dictionaries with
        different settings for each plotted spectral distributions.
        The following special keyword arguments can also be used:

        -   *illuminant* : unicode or :class:`colour.SpectralDistribution`, the
            illuminant used to compute the spectral distributions colours. The
            default is the illuminant associated with the whitepoint of the
            default plotting colourspace. ``illuminant`` can be of any type or
            form supported by the :func:`colour.plotting.filter_cmfs`
            definition.
        -   *cmfs* : unicode, the standard observer colour matching functions
            used for computing the spectral distributions colours. ``cmfs`` can
            be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   *normalise_sd_colours* : bool, whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   *use_sd_colours* : bool, whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the :func:`plt.plot`
            definition ``color`` argument with pre-computed values. The default
            is *True*.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
        Also handles keywords arguments for deprecation management.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> A = SDS_ILLUMINANTS['A']
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> plot_sds_in_chromaticity_diagram_CIE1960UCS([A, D65])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_SDS_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_sds_in_chromaticity_diagram_CIE1960UCS
    """

    annotate_kwargs = handle_arguments_deprecation({
        'ArgumentRenamed': [['annotate_parameters', 'annotate_kwargs']],
    }, **kwargs).get('annotate_kwargs', annotate_kwargs)

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return plot_sds_in_chromaticity_diagram(
        sds,
        cmfs,
        chromaticity_diagram_callable_CIE1960UCS,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings)


@override_style()
def plot_sds_in_chromaticity_diagram_CIE1976UCS(
        sds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1976UCS=(
            plot_chromaticity_diagram_CIE1976UCS),
        annotate_kwargs=None,
        plot_kwargs=None,
        **kwargs):
    """
    Plots given spectral distribution chromaticity coordinates into the
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    sds : array_like or MultiSpectralDistributions
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single :class:`colour.MultiSpectralDistributions`
        class instance, a list of :class:`colour.MultiSpectralDistributions`
        class instances or a list of :class:`colour.SpectralDistribution` class
        instances.
    cmfs : unicode or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    chromaticity_diagram_callable_CIE1976UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.
    annotate_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.annotate` definition, used to
        annotate the resulting chromaticity coordinates with their respective
        spectral distribution names. ``annotate_kwargs`` can be either a single
        dictionary applied to all the arrows with same settings or a sequence
        of dictionaries with different settings for each spectral distribution.
        The following special keyword arguments can also be used:

        -   *annotate* : bool, whether to annotate the spectral distributions.
    plot_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.plot` definition, used to control
        the style of the plotted spectral distributions. ``plot_kwargs`` can be
        either a single dictionary applied to all the plotted spectral
        distributions with same settings or a sequence of dictionaries with
        different settings for each plotted spectral distributions.
        The following special keyword arguments can also be used:

        -   *illuminant* : unicode or :class:`colour.SpectralDistribution`, the
            illuminant used to compute the spectral distributions colours. The
            default is the illuminant associated with the whitepoint of the
            default plotting colourspace. ``illuminant`` can be of any type or
            form supported by the :func:`colour.plotting.filter_cmfs`
            definition.
        -   *cmfs* : unicode, the standard observer colour matching functions
            used for computing the spectral distributions colours. ``cmfs`` can
            be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   *normalise_sd_colours* : bool, whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   *use_sd_colours* : bool, whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the :func:`plt.plot`
            definition ``color`` argument with pre-computed values. The default
            is *True*.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
        Also handles keywords arguments for deprecation management.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> A = SDS_ILLUMINANTS['A']
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> plot_sds_in_chromaticity_diagram_CIE1976UCS([A, D65])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_SDS_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_sds_in_chromaticity_diagram_CIE1976UCS
    """

    annotate_kwargs = handle_arguments_deprecation({
        'ArgumentRenamed': [['annotate_parameters', 'annotate_kwargs']],
    }, **kwargs).get('annotate_kwargs', annotate_kwargs)

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return plot_sds_in_chromaticity_diagram(
        sds,
        cmfs,
        chromaticity_diagram_callable_CIE1976UCS,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings)
