# -*- coding: utf-8 -*-
"""
Characterisation Plotting
=========================

Defines the characterisation plotting objects:

-   :func:`colour.plotting.single_colour_checker_plot`
-   :func:`colour.plotting.multi_colour_checker_plot`
"""

from __future__ import division

import numpy as np

from colour.characterisation import COLOURCHECKERS
from colour.models import xyY_to_XYZ
from colour.plotting import (ColourSwatch, COLOUR_STYLE_CONSTANTS,
                             XYZ_to_plotting_colourspace,
                             multi_colour_swatch_plot, render)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['single_colour_checker_plot', 'multi_colour_checker_plot']


def single_colour_checker_plot(colour_checker='ColorChecker 2005', **kwargs):
    """
    Plots given colour checker.

    Parameters
    ----------
    colour_checker : unicode, optional
        Color checker name.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.multi_colour_swatch_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Raises
    ------
    KeyError
        If the given colour rendition chart is not found in the factory colour
        rendition charts.

    Examples
    --------
    >>> colour_checker_plot('ColorChecker 2005')  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_Colour_Checker_Plot.png
        :align: center
        :alt: colour_checker_plot
    """

    return multi_colour_checker_plot([colour_checker], **kwargs)


def multi_colour_checker_plot(colour_checkers=None, **kwargs):
    """
    Plots and compares given colour checkers.

    Parameters
    ----------
    colour_checkers : array_like, optional
        Color checker names, must be less than or equal to 2 names.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.multi_colour_swatch_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Raises
    ------
    KeyError
        If the given colour rendition chart is not found in the factory colour
        rendition charts.

    Examples
    --------
    >>> multi_colour_checker_plot(['ColorChecker 1976', 'ColorChecker 2005'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_Colour_Checker_Plot.png
        :align: center
        :alt: colour_checker_plot
    """

    if colour_checkers is None:
        colour_checkers = ['ColorChecker 1976', 'ColorChecker 2005']
    else:
        assert len(colour_checkers) <= 2, (
            'Only two colour checkers can be compared at a time!')

    compare_swatches = len(colour_checkers) == 2

    colour_swatches = []
    colour_checker_names = []
    for colour_checker in colour_checkers:
        colour_checker = COLOURCHECKERS.get(colour_checker)
        colour_checker_names.append(colour_checker.name)
        if colour_checker is None:
            raise KeyError(('Colour checker "{0}" not found in '
                            'factory colour checkers: "{1}".').format(
                                colour_checker.name,
                                sorted(COLOURCHECKERS.keys())))

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
        'standalone': False,
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

    figure = multi_colour_swatch_plot(colour_swatches, **settings)
    axes = figure.gca()

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
        'title': ', '.join(colour_checker_names),
        'facecolor': background_colour,
        'edgecolor': None,
        'standalone': True
    })

    return render(**settings)
