# -*- coding: utf-8 -*-
"""
Colour Temperature & Correlated Colour Temperature Plotting
===========================================================

Defines the colour temperature and correlated colour temperature plotting
objects:

-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS`
"""

from __future__ import division

import numpy as np

from colour.colorimetry import CMFS, ILLUMINANTS
from colour.models import (UCS_uv_to_xy, XYZ_to_UCS, UCS_to_uv, xy_to_XYZ)
from colour.temperature import CCT_to_uv
from colour.plotting import (COLOUR_STYLE_CONSTANTS, COLOUR_ARROW_STYLE,
                             artist, plot_chromaticity_diagram_CIE1931,
                             plot_chromaticity_diagram_CIE1960UCS,
                             filter_passthrough, override_style, render)
from colour.plotting.diagrams import plot_chromaticity_diagram
from colour.utilities import tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'plot_planckian_locus', 'plot_planckian_locus_CIE1931',
    'plot_planckian_locus_CIE1960UCS',
    'plot_planckian_locus_in_chromaticity_diagram',
    'plot_planckian_locus_in_chromaticity_diagram_CIE1931',
    'plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS'
]


@override_style()
def plot_planckian_locus(planckian_locus_colours=None,
                         method='CIE 1931',
                         **kwargs):
    """
    Plots the *Planckian Locus* according to given method.

    Parameters
    ----------
    planckian_locus_colours : array_like or unicode, optional
        *Planckian Locus* colours.
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
    >>> plot_planckian_locus()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Planckian_Locus.png
        :align: center
        :alt: plot_planckian_locus
    """

    if planckian_locus_colours is None:
        planckian_locus_colours = COLOUR_STYLE_CONSTANTS.colour.dark

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if method == 'CIE 1931':

        def uv_to_ij(uv):
            """
            Converts given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return UCS_uv_to_xy(uv)

        D_uv = 0.025
    elif method == 'CIE 1960 UCS':

        def uv_to_ij(uv):
            """
            Converts given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return uv

        D_uv = 0.025
    else:
        raise ValueError('Invalid method: "{0}", must be one of '
                         '{{\'CIE 1931\', \'CIE 1960 UCS\'}}'.format(method))

    start, end = 1667, 100000
    CCT = np.arange(start, end + 250, 250)
    CCT_D_uv = tstack([CCT, np.zeros(CCT.shape)])
    ij = uv_to_ij(CCT_to_uv(CCT_D_uv, 'Robertson 1968'))

    axes.plot(ij[..., 0], ij[..., 1], color=planckian_locus_colours)

    for i in (1667, 2000, 2500, 3000, 4000, 6000, 10000):
        i0, j0 = uv_to_ij(CCT_to_uv(np.array([i, -D_uv]), 'Robertson 1968'))
        i1, j1 = uv_to_ij(CCT_to_uv(np.array([i, D_uv]), 'Robertson 1968'))
        axes.plot((i0, i1), (j0, j1), color=planckian_locus_colours)
        axes.annotate(
            '{0}K'.format(i),
            xy=(i0, j0),
            xytext=(0, -10),
            textcoords='offset points',
            size='x-small')

    settings = {'axes': axes}
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_planckian_locus_CIE1931(planckian_locus_colours=None, **kwargs):
    """
    Plots the *Planckian Locus* according to *CIE 1931* method.

    Parameters
    ----------
    planckian_locus_colours : array_like or unicode, optional
        *Planckian Locus* colours.

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
    >>> plot_planckian_locus_CIE1931()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Planckian_Locus_CIE1931.png
        :align: center
        :alt: plot_planckian_locus_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return plot_planckian_locus(planckian_locus_colours, **settings)


@override_style()
def plot_planckian_locus_CIE1960UCS(planckian_locus_colours=None, **kwargs):
    """
    Plots the *Planckian Locus* according to *CIE 1960 UCS* method.

    Parameters
    ----------
    planckian_locus_colours : array_like or unicode, optional
        *Planckian Locus* colours.

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
    >>> plot_planckian_locus_CIE1960UCS()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Planckian_Locus_CIE1960UCS.png
        :align: center
        :alt: plot_planckian_locus_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return plot_planckian_locus(planckian_locus_colours, **settings)


@override_style()
def plot_planckian_locus_in_chromaticity_diagram(
        illuminants=None,
        annotate_parameters=None,
        chromaticity_diagram_callable=plot_chromaticity_diagram,
        planckian_locus_callable=plot_planckian_locus,
        method='CIE 1931',
        **kwargs):
    """
    Plots the *Planckian Locus* and given illuminants in the
    *Chromaticity Diagram* according to given method.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.
    annotate_parameters : dict or array_like, optional
        Parameters for the :func:`plt.annotate` definition, used to annotate
        the resulting chromaticity coordinates with their respective illuminant
        names if ``annotate`` is set to *True*. ``annotate_parameters`` can be
        either a single dictionary applied to all the arrows with same settings
        or a sequence of dictionaries with different settings for each
        illuminant.
    chromaticity_diagram_callable : callable, optional
        Callable responsible for drawing the *Chromaticity Diagram*.
    planckian_locus_callable : callable, optional
        Callable responsible for drawing the *Planckian Locus*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.temperature.plot_planckian_locus`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_planckian_locus_in_chromaticity_diagram(['A', 'B', 'C'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_Planckian_Locus_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_planckian_locus_in_chromaticity_diagram
    """

    cmfs = CMFS['CIE 1931 2 Degree Standard Observer']

    if illuminants is None:
        illuminants = ('A', 'B', 'C')

    illuminants = filter_passthrough(ILLUMINANTS.get(cmfs.name), illuminants)

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    settings = {'axes': axes, 'method': method}
    settings.update(kwargs)
    settings['standalone'] = False

    chromaticity_diagram_callable(**settings)

    planckian_locus_callable(**settings)

    if method == 'CIE 1931':

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy

        bounding_box = (-0.1, 0.9, -0.1, 0.9)
    elif method == 'CIE 1960 UCS':

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(xy)))

        bounding_box = (-0.1, 0.7, -0.2, 0.6)
    else:
        raise ValueError('Invalid method: "{0}", must be one of '
                         '{{\'CIE 1931\', \'CIE 1960 UCS\'}}'.format(method))

    annotate_settings_collection = [{
        'annotate': True,
        'xytext': (-50, 30),
        'textcoords': 'offset points',
        'arrowprops': COLOUR_ARROW_STYLE,
    } for _ in range(len(illuminants))]

    if annotate_parameters is not None:
        if not isinstance(annotate_parameters, dict):
            assert len(annotate_parameters) == len(illuminants), (
                'Multiple annotate parameters defined, but they do not match '
                'the illuminants count!')

        for i, annotate_settings in enumerate(annotate_settings_collection):
            if isinstance(annotate_parameters, dict):
                annotate_settings.update(annotate_parameters)
            else:
                annotate_settings.update(annotate_parameters[i])

    for i, (illuminant, xy) in enumerate(illuminants.items()):
        ij = xy_to_ij(xy)

        axes.plot(
            ij[0],
            ij[1],
            'o',
            color=COLOUR_STYLE_CONSTANTS.colour.brightest,
            markeredgecolor=COLOUR_STYLE_CONSTANTS.colour.dark,
            markersize=(COLOUR_STYLE_CONSTANTS.geometry.short * 6 +
                        COLOUR_STYLE_CONSTANTS.geometry.short * 0.75),
            markeredgewidth=COLOUR_STYLE_CONSTANTS.geometry.short * 0.75,
            label=illuminant)

        if annotate_settings_collection[i]['annotate']:
            annotate_settings = annotate_settings_collection[i]
            annotate_settings.pop('annotate')

            axes.annotate(illuminant, xy=ij, **annotate_settings)

    title = (('{0} Illuminants - Planckian Locus\n'
              '{1} Chromaticity Diagram - '
              'CIE 1931 2 Degree Standard Observer').format(
                  ', '.join(illuminants), method) if illuminants else
             ('Planckian Locus\n{0} Chromaticity Diagram - '
              'CIE 1931 2 Degree Standard Observer'.format(method)))

    settings.update({
        'axes': axes,
        'standalone': True,
        'bounding_box': bounding_box,
        'title': title,
    })
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_planckian_locus_in_chromaticity_diagram_CIE1931(
        illuminants=None,
        annotate_parameters=None,
        chromaticity_diagram_callable_CIE1931=(
            plot_chromaticity_diagram_CIE1931),
        planckian_locus_callable_CIE1931=plot_planckian_locus_CIE1931,
        **kwargs):
    """
    Plots the *Planckian Locus* and given illuminants in
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.
    annotate_parameters : dict or array_like, optional
        Parameters for the :func:`plt.annotate` definition, used to annotate
        the resulting chromaticity coordinates with their respective illuminant
        names if ``annotate`` is set to *True*. ``annotate_parameters`` can be
        either a single dictionary applied to all the arrows with same settings
        or a sequence of dictionaries with different settings for each
        illuminant.
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    planckian_locus_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *Planckian Locus* according to
        *CIE 1931* method.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.temperature.plot_planckian_locus`,
        :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_planckian_locus_in_chromaticity_diagram_CIE1931(['A', 'B', 'C'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_Planckian_Locus_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_planckian_locus_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return plot_planckian_locus_in_chromaticity_diagram(
        illuminants, annotate_parameters,
        chromaticity_diagram_callable_CIE1931,
        planckian_locus_callable_CIE1931, **settings)


@override_style()
def plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
        illuminants=None,
        annotate_parameters=None,
        chromaticity_diagram_callable_CIE1960UCS=(
            plot_chromaticity_diagram_CIE1960UCS),
        planckian_locus_callable_CIE1960UCS=plot_planckian_locus_CIE1960UCS,
        **kwargs):
    """
    Plots the *Planckian Locus* and given illuminants in
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.
    annotate_parameters : dict or array_like, optional
        Parameters for the :func:`plt.annotate` definition, used to annotate
        the resulting chromaticity coordinates with their respective illuminant
        names if ``annotate`` is set to *True*. ``annotate_parameters`` can be
        either a single dictionary applied to all the arrows with same settings
        or a sequence of dictionaries with different settings for each
        illuminant.
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.
    planckian_locus_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the *Planckian Locus* according to
        *CIE 1960 UCS* method.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.temperature.plot_planckian_locus`,
        :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
    ...     ['A', 'C', 'E'])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_Planckian_Locus_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return plot_planckian_locus_in_chromaticity_diagram(
        illuminants, annotate_parameters,
        chromaticity_diagram_callable_CIE1960UCS,
        planckian_locus_callable_CIE1960UCS, **settings)
