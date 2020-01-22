# -*- coding: utf-8 -*-
"""
Characterisation Plotting
=========================

Defines the characterisation plotting objects:

-   :func:`colour.plotting.plot_single_colour_checker`
-   :func:`colour.plotting.plot_multi_colour_checkers`
"""

from __future__ import division

import numpy as np

from colour.models import xyY_to_XYZ
from colour.plotting import (
    COLOUR_STYLE_CONSTANTS, ColourSwatch, XYZ_to_plotting_colourspace, artist,
    filter_colour_checkers, plot_multi_colour_swatches, override_style, render)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['plot_single_colour_checker', 'plot_multi_colour_checkers']


@override_style(
    **{
        'axes.grid': False,
        'xtick.bottom': False,
        'ytick.left': False,
        'xtick.labelbottom': False,
        'ytick.labelleft': False,
    })
def plot_single_colour_checker(colour_checker='ColorChecker 2005', **kwargs):
    """
    Plots given colour checker.

    Parameters
    ----------
    colour_checker : unicode, optional
        Color checker name.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_colour_swatches`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_single_colour_checker('ColorChecker 2005')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Single_Colour_Checker.png
        :align: center
        :alt: plot_single_colour_checker
    """

    return plot_multi_colour_checkers([colour_checker], **kwargs)


@override_style(
    **{
        'axes.grid': False,
        'xtick.bottom': False,
        'ytick.left': False,
        'xtick.labelbottom': False,
        'ytick.labelleft': False,
    })
def plot_multi_colour_checkers(colour_checkers=None, **kwargs):
    """
    Plots and compares given colour checkers.

    Parameters
    ----------
    colour_checkers : array_like, optional
        Color checker names, must be less than or equal to 2 names.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_colour_swatches`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_colour_checkers(['ColorChecker 1976', 'ColorChecker 2005'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Multi_Colour_Checkers.png
        :align: center
        :alt: plot_multi_colour_checkers
    """

    if colour_checkers is None:
        colour_checkers = ['ColorChecker 1976', 'ColorChecker 2005']
    else:
        assert len(colour_checkers) <= 2, (
            'Only two colour checkers can be compared at a time!')

    colour_checkers = filter_colour_checkers(colour_checkers).values()

    _figure, axes = artist(**kwargs)

    compare_swatches = len(colour_checkers) == 2

    colour_swatches = []
    colour_checker_names = []
    for colour_checker in colour_checkers:
        colour_checker_names.append(colour_checker.name)
        for label, xyY in colour_checker.data.items():
            XYZ = xyY_to_XYZ(xyY)
            RGB = XYZ_to_plotting_colourspace(XYZ, colour_checker.illuminant)
            colour_swatches.append(
                ColourSwatch(label.title(), np.clip(np.ravel(RGB), 0, 1)))

    if compare_swatches:
        colour_swatches = [
            swatch
            for pairs in zip(colour_swatches[0:len(colour_swatches) // 2],
                             colour_swatches[len(colour_swatches) // 2:])
            for swatch in pairs
        ]

    background_colour = '0.1'
    width = height = 1.0
    spacing = 0.25
    columns = 6

    settings = {
        'axes': axes,
        'width': width,
        'height': height,
        'spacing': spacing,
        'columns': columns,
        'text_parameters': {
            'size': 8
        },
        'background_colour': background_colour,
        'compare_swatches': 'Stacked' if compare_swatches else None,
    }
    settings.update(kwargs)
    settings['standalone'] = False

    plot_multi_colour_swatches(colour_swatches, **settings)

    axes.text(
        0.5,
        0.005,
        '{0} - {1} - Colour Rendition Chart'.format(
            ', '.join(colour_checker_names),
            COLOUR_STYLE_CONSTANTS.colour.colourspace.name),
        transform=axes.transAxes,
        color=COLOUR_STYLE_CONSTANTS.colour.bright,
        ha='center',
        va='bottom')

    settings.update({
        'axes': axes,
        'standalone': True,
        'title': ', '.join(colour_checker_names),
    })

    return render(**settings)
