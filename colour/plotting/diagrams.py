# -*- coding: utf-8 -*-
"""
CIE Chromaticity Diagrams Plotting
==================================

Defines the *CIE* chromaticity diagrams plotting objects:

-   :func:`colour.plotting.chromaticity_diagram_plot_CIE1931`
-   :func:`colour.plotting.chromaticity_diagram_plot_CIE1960UCS`
-   :func:`colour.plotting.chromaticity_diagram_plot_CIE1976UCS`
-   :func:`colour.plotting.spds_chromaticity_diagram_plot_CIE1931`
-   :func:`colour.plotting.spds_chromaticity_diagram_plot_CIE1960UCS`
-   :func:`colour.plotting.spds_chromaticity_diagram_plot_CIE1976UCS`
"""

from __future__ import division

import bisect
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from colour.algebra import normalise_vector
from colour.colorimetry import spectral_to_XYZ
from colour.models import (Luv_to_uv, Luv_uv_to_xy, UCS_to_uv, UCS_uv_to_xy,
                           XYZ_to_Luv, XYZ_to_UCS, XYZ_to_xy, xy_to_XYZ)
from colour.plotting import (COLOUR_STYLE_CONSTANTS,
                             XYZ_to_plotting_colourspace, artist, filter_cmfs,
                             override_style, render)
from colour.utilities import (domain_range_scale, first_item, is_string,
                              normalise_maximum, suppress_warnings, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'spectral_locus_plot', 'chromaticity_diagram_colours_plot',
    'chromaticity_diagram_plot', 'chromaticity_diagram_plot_CIE1931',
    'chromaticity_diagram_plot_CIE1960UCS',
    'chromaticity_diagram_plot_CIE1976UCS', 'spds_chromaticity_diagram_plot',
    'spds_chromaticity_diagram_plot_CIE1931',
    'spds_chromaticity_diagram_plot_CIE1960UCS',
    'spds_chromaticity_diagram_plot_CIE1976UCS'
]


@override_style()
def spectral_locus_plot(cmfs='CIE 1931 2 Degree Standard Observer',
                        spectral_locus_colours=None,
                        spectral_locus_labels=None,
                        method='CIE 1931',
                        **kwargs):
    """
    Plots the *Spectral Locus* according to given method.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions defining the
        *Spectral Locus*.
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
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> spectral_locus_plot(spectral_locus_colours='RGB')  # doctest: +SKIP

    .. image:: ../_static/Plotting_Spectral_Locus_Plot.png
        :align: center
        :alt: spectral_locus_plot
    """

    if spectral_locus_colours is None:
        spectral_locus_colours = COLOUR_STYLE_CONSTANTS.colour.dark

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    method = method.upper()

    cmfs = first_item(filter_cmfs(cmfs))

    illuminant = COLOUR_STYLE_CONSTANTS.colour.colourspace.whitepoint

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
            '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
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
        i, j = wl_ij[label]

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else wavelengths[-1])

        dx = wl_ij[right][0] - wl_ij[left][0]
        dy = wl_ij[right][1] - wl_ij[left][1]

        ij = np.array([i, j])
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
def chromaticity_diagram_colours_plot(
        samples=256,
        diagram_opacity=1.0,
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
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

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
    >>> chromaticity_diagram_colours_plot()  # doctest: +SKIP

    .. image:: ../_static/Plotting_Chromaticity_Diagram_Colours_Plot.png
        :align: center
        :alt: chromaticity_diagram_colours_plot
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    method = method.upper()

    cmfs = first_item(filter_cmfs(cmfs))

    illuminant = COLOUR_STYLE_CONSTANTS.colour.colourspace.whitepoint

    ii, jj = np.meshgrid(
        np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    ij = tstack((ii, jj))

    with suppress_warnings(False):
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
                '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
                    method))

        RGB = normalise_maximum(
            XYZ_to_plotting_colourspace(XYZ, illuminant), axis=-1)

    polygon = Polygon(spectral_locus, facecolor='none', edgecolor='none')
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
def chromaticity_diagram_plot(cmfs='CIE 1931 2 Degree Standard Observer',
                              show_diagram_colours=True,
                              show_spectral_locus=True,
                              method='CIE 1931',
                              **kwargs):
    """
    Plots the *Chromaticity Diagram* according to given method.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    show_diagram_colours : bool, optional
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus : bool, optional
        Whether to display the *Spectral Locus*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.spectral_locus_plot`,
        :func:`colour.plotting.diagrams.chromaticity_diagram_colours_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> chromaticity_diagram_plot()  # doctest: +SKIP

    .. image:: ../_static/Plotting_Chromaticity_Diagram_Plot.png
        :align: center
        :alt: chromaticity_diagram_plot
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    method = method.upper()

    cmfs = first_item(filter_cmfs(cmfs))

    if show_diagram_colours:
        settings = {'axes': axes, 'method': method}
        settings.update(kwargs)
        settings['standalone'] = False

        chromaticity_diagram_colours_plot(**settings)

    if show_spectral_locus:
        settings = {'axes': axes, 'method': method}
        settings.update(kwargs)
        settings['standalone'] = False

        spectral_locus_plot(**settings)

    if method == 'CIE 1931':
        x_label, y_label = 'CIE x', 'CIE y'
    elif method == 'CIE 1960 UCS':
        x_label, y_label = 'CIE u', 'CIE v'
    elif method == 'CIE 1976 UCS':
        x_label, y_label = 'CIE u\'', 'CIE v\'',
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
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
def chromaticity_diagram_plot_CIE1931(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        show_spectral_locus=True,
        **kwargs):
    """
    Plots the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    show_diagram_colours : bool, optional
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus : bool, optional
        Whether to display the *Spectral Locus*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1931()  # doctest: +SKIP

    .. image:: ../_static/Plotting_Chromaticity_Diagram_Plot_CIE1931.png
        :align: center
        :alt: chromaticity_diagram_plot_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return chromaticity_diagram_plot(cmfs, show_diagram_colours,
                                     show_spectral_locus, **settings)


@override_style()
def chromaticity_diagram_plot_CIE1960UCS(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        show_spectral_locus=True,
        **kwargs):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    show_diagram_colours : bool, optional
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus : bool, optional
        Whether to display the *Spectral Locus*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1960UCS()  # doctest: +SKIP

    .. image:: ../_static/Plotting_Chromaticity_Diagram_Plot_CIE1960UCS.png
        :align: center
        :alt: chromaticity_diagram_plot_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return chromaticity_diagram_plot(cmfs, show_diagram_colours,
                                     show_spectral_locus, **settings)


@override_style()
def chromaticity_diagram_plot_CIE1976UCS(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        show_spectral_locus=True,
        **kwargs):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    show_diagram_colours : bool, optional
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus : bool, optional
        Whether to display the *Spectral Locus*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1976UCS()  # doctest: +SKIP

    .. image:: ../_static/Plotting_Chromaticity_Diagram_Plot_CIE1976UCS.png
        :align: center
        :alt: chromaticity_diagram_plot_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return chromaticity_diagram_plot(cmfs, show_diagram_colours,
                                     show_spectral_locus, **settings)


@override_style()
def spds_chromaticity_diagram_plot(
        spds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        annotate_parameters=None,
        chromaticity_diagram_callable=chromaticity_diagram_plot,
        method='CIE 1931',
        **kwargs):
    """
    Plots given spectral power distribution chromaticity coordinates into the
    *Chromaticity Diagram* using given method.

    Parameters
    ----------
    spds : array_like, optional
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    annotate_parameters : dict or array_like, optional
        Parameters for the :func:`plt.annotate` definition, used to annotate
        the resulting chromaticity coordinates with their respective spectral
        power distribution names if ``annotate`` is set to *True*.
        ``annotate_parameters`` can be either a single dictionary applied to
        all the arrows with same settings or a sequence of dictionaries with
        different settings for each spectral power distribution.
    chromaticity_diagram_callable : callable, optional
        Callable responsible for drawing the *Chromaticity Diagram*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import ILLUMINANTS_SPDS
    >>> A = ILLUMINANTS_SPDS['A']
    >>> D65 = ILLUMINANTS_SPDS['D65']
    >>> spds_chromaticity_diagram_plot([A, D65])  # doctest: +SKIP

    .. image:: ../_static/Plotting_SPDS_Chromaticity_Diagram_Plot.png
        :align: center
        :alt: spds_chromaticity_diagram_plot
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

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
            '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
                method))

    annotate_settings_collection = [{
        'annotate': True,
        'xytext': (-50, 30),
        'textcoords': 'offset points',
        'arrowprops': {
            'arrowstyle': '->',
            'connectionstyle': 'arc3, rad=-0.2'
        }
    } for _ in range(len(spds))]

    if annotate_parameters is not None:
        if not isinstance(annotate_parameters, dict):
            assert len(annotate_parameters) == len(spds), (
                'Multiple annotate parameters defined, but they do not match '
                'the spectral power distributions count!')

        for i, annotate_settings in enumerate(annotate_settings_collection):
            if isinstance(annotate_parameters, dict):
                annotate_settings.update(annotate_parameters)
            else:
                annotate_settings.update(annotate_parameters[i])

    for i, spd in enumerate(spds):
        with domain_range_scale('1'):
            XYZ = spectral_to_XYZ(spd)

        ij = XYZ_to_ij(XYZ)

        axes.plot(
            ij[0], ij[1], 'o', color=COLOUR_STYLE_CONSTANTS.colour.brightest)

        if (spd.name is not None and
                annotate_settings_collection[i]['annotate']):
            annotate_settings = annotate_settings_collection[i]
            annotate_settings.pop('annotate')

            axes.annotate(spd.name, xy=ij, **annotate_settings)

    settings.update({'standalone': True, 'bounding_box': bounding_box})
    settings.update(kwargs)

    return render(**settings)


@override_style()
def spds_chromaticity_diagram_plot_CIE1931(
        spds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        annotate_parameters=None,
        chromaticity_diagram_callable_CIE1931=(
            chromaticity_diagram_plot_CIE1931),
        **kwargs):
    """
    Plots given spectral power distribution chromaticity coordinates into the
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    spds : array_like, optional
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    annotate_parameters : dict or array_like, optional
        Parameters for the :func:`plt.annotate` definition, used to annotate
        the resulting chromaticity coordinates with their respective spectral
        power distribution names if ``annotate`` is set to *True*.
        ``annotate_parameters`` can be either a single dictionary applied to
        all the arrows with same settings or a sequence of dictionaries with
        different settings for each spectral power distribution.
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import ILLUMINANTS_SPDS
    >>> A = ILLUMINANTS_SPDS['A']
    >>> D65 = ILLUMINANTS_SPDS['D65']
    >>> spds_chromaticity_diagram_plot_CIE1931([A, D65])  # doctest: +SKIP

    .. image:: ../_static/Plotting_SPDS_Chromaticity_Diagram_Plot_CIE1931.png
        :align: center
        :alt: spds_chromaticity_diagram_plot_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return spds_chromaticity_diagram_plot(
        spds, cmfs, annotate_parameters, chromaticity_diagram_callable_CIE1931,
        **settings)


@override_style()
def spds_chromaticity_diagram_plot_CIE1960UCS(
        spds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        annotate_parameters=None,
        chromaticity_diagram_callable_CIE1960UCS=(
            chromaticity_diagram_plot_CIE1960UCS),
        **kwargs):
    """
    Plots given spectral power distribution chromaticity coordinates into the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    spds : array_like, optional
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    annotate_parameters : dict or array_like, optional
        Parameters for the :func:`plt.annotate` definition, used to annotate
        the resulting chromaticity coordinates with their respective spectral
        power distribution names if ``annotate`` is set to *True*.
        ``annotate_parameters`` can be either a single dictionary applied to
        all the arrows with same settings or a sequence of dictionaries with
        different settings for each spectral power distribution.
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import ILLUMINANTS_SPDS
    >>> A = ILLUMINANTS_SPDS['A']
    >>> D65 = ILLUMINANTS_SPDS['D65']
    >>> spds_chromaticity_diagram_plot_CIE1960UCS([A, D65])  # doctest: +SKIP

    .. image:: ../_static/Plotting_\
SPDS_Chromaticity_Diagram_Plot_CIE1960UCS.png
        :align: center
        :alt: spds_chromaticity_diagram_plot_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return spds_chromaticity_diagram_plot(
        spds, cmfs, annotate_parameters,
        chromaticity_diagram_callable_CIE1960UCS, **settings)


@override_style()
def spds_chromaticity_diagram_plot_CIE1976UCS(
        spds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        annotate_parameters=None,
        chromaticity_diagram_callable_CIE1976UCS=(
            chromaticity_diagram_plot_CIE1976UCS),
        **kwargs):
    """
    Plots given spectral power distribution chromaticity coordinates into the
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    spds : array_like, optional
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    annotate_parameters : dict or array_like, optional
        Parameters for the :func:`plt.annotate` definition, used to annotate
        the resulting chromaticity coordinates with their respective spectral
        power distribution names if ``annotate`` is set to *True*.
        ``annotate_parameters`` can be either a single dictionary applied to
        all the arrows with same settings or a sequence of dictionaries with
        different settings for each spectral power distribution.
    chromaticity_diagram_callable_CIE1976UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import ILLUMINANTS_SPDS
    >>> A = ILLUMINANTS_SPDS['A']
    >>> D65 = ILLUMINANTS_SPDS['D65']
    >>> spds_chromaticity_diagram_plot_CIE1976UCS([A, D65])  # doctest: +SKIP

    .. image:: ../_static/Plotting_\
SPDS_Chromaticity_Diagram_Plot_CIE1976UCS.png
        :align: center
        :alt: spds_chromaticity_diagram_plot_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return spds_chromaticity_diagram_plot(
        spds, cmfs, annotate_parameters,
        chromaticity_diagram_callable_CIE1976UCS, **settings)
