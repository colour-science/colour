#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Models Plotting
======================

Defines the colour models plotting objects:

-   :func:`colourspaces_CIE_1931_chromaticity_diagram_plot`
-   :func:`single_transfer_function_plot`
-   :func:`multi_transfer_function_plot`
"""

from __future__ import division

import numpy as np
import pylab

from colour.models import (
    POINTER_GAMUT_DATA,
    POINTER_GAMUT_ILLUMINANT,
    POINTER_GAMUT_BOUNDARIES,
    RGB_COLOURSPACES,
    LCHab_to_Lab,
    Lab_to_XYZ,
    XYZ_to_xy)
from colour.plotting import (
    CIE_1931_chromaticity_diagram_plot,
    DEFAULT_FIGURE_WIDTH,
    canvas,
    decorate,
    boundaries,
    display,
    colour_cycle,
    get_cmfs)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['get_RGB_colourspace',
           'colourspaces_CIE_1931_chromaticity_diagram_plot',
           'single_transfer_function_plot',
           'multi_transfer_function_plot']


def get_RGB_colourspace(colourspace):
    """
    Returns the *RGB* colourspace with given name.

    Parameters
    ----------
    colourspace : unicode
        *RGB* Colourspace name.

    Returns
    -------
    RGB_Colourspace
        *RGB* Colourspace.

    Raises
    ------
    KeyError
        If the given colourspace is not found in the factory colourspaces.
    """

    colourspace, name = RGB_COLOURSPACES.get(colourspace), colourspace
    if colourspace is None:
        raise KeyError(
            ('"{0}" colourspace not found in factory colourspaces: '
             '"{1}".').format(name, ', '.join(
                sorted(RGB_COLOURSPACES.keys()))))

    return colourspace


def colourspaces_CIE_1931_chromaticity_diagram_plot(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots given colourspaces in *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces : list, optional
        Colourspaces to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> csps = ['sRGB', 'ACES RGB']
    >>> colourspaces_CIE_1931_chromaticity_diagram_plot(csps)  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if colourspaces is None:
        colourspaces = ('sRGB', 'ACES RGB', 'Pointer Gamut')

    cmfs, name = get_cmfs(cmfs), cmfs

    settings = {
        'title': '{0} - {1}'.format(', '.join(colourspaces), name),
        'standalone': False}
    settings.update(kwargs)

    if not CIE_1931_chromaticity_diagram_plot(**settings):
        return

    x_limit_min, x_limit_max = [-0.1], [0.9]
    y_limit_min, y_limit_max = [-0.1], [0.9]

    cycle = colour_cycle('rainbow')
    for colourspace in colourspaces:
        if colourspace == 'Pointer Gamut':
            xy = np.array(POINTER_GAMUT_BOUNDARIES)
            alpha_p, colour_p = 0.85, '0.95'
            pylab.plot(xy[:, 0],
                       xy[:, 1],
                       label='Pointer\'s Gamut',
                       color=colour_p,
                       alpha=alpha_p,
                       linewidth=2)
            pylab.plot([xy[-1][0],
                        xy[0][0]],
                       [xy[-1][1],
                        xy[0][1]],
                       color=colour_p,
                       alpha=alpha_p,
                       linewidth=2)

            xy = []
            for LCHab in POINTER_GAMUT_DATA:
                XYZ = Lab_to_XYZ(LCHab_to_Lab(LCHab), POINTER_GAMUT_ILLUMINANT)
                xy.append(XYZ_to_xy(XYZ, POINTER_GAMUT_ILLUMINANT))
            xy = np.array(xy)
            pylab.scatter(xy[:, 0],
                          xy[:, 1],
                          alpha=alpha_p / 2,
                          color=colour_p,
                          marker='+')

        else:
            colourspace, name = get_RGB_colourspace(
                colourspace), colourspace

            r, g, b, a = next(cycle)

            primaries = colourspace.primaries
            whitepoint = colourspace.whitepoint

            pylab.plot([whitepoint[0], whitepoint[0]],
                       [whitepoint[1], whitepoint[1]],
                       color=(r, g, b),
                       label=colourspace.name,
                       linewidth=2)
            pylab.plot([whitepoint[0], whitepoint[0]],
                       [whitepoint[1], whitepoint[1]],
                       'o',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot([primaries[0, 0], primaries[1, 0]],
                       [primaries[0, 1], primaries[1, 1]],
                       'o-',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot([primaries[1, 0], primaries[2, 0]],
                       [primaries[1, 1], primaries[2, 1]],
                       'o-',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot([primaries[2, 0], primaries[0, 0]],
                       [primaries[2, 1], primaries[0, 1]],
                       'o-',
                       color=(r, g, b),
                       linewidth=2)

            x_limit_min.append(np.amin(primaries[:, 0]))
            y_limit_min.append(np.amin(primaries[:, 1]))
            x_limit_max.append(np.amax(primaries[:, 0]))
            y_limit_max.append(np.amax(primaries[:, 1]))

    settings.update({
        'legend': True,
        'legend_location': 'upper right',
        'x_tighten': True,
        'y_tighten': True,
        'limits': [min(x_limit_min), max(x_limit_max),
                   min(y_limit_min), max(y_limit_max)],
        'margins': [-0.05, 0.05, -0.05, 0.05],
        'standalone': True})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def single_transfer_function_plot(colourspace='sRGB', **kwargs):
    """
    Plots given colourspace transfer function.

    Parameters
    ----------
    colourspace : unicode, optional
        *RGB* Colourspace transfer function to plot.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> single_transfer_function_plot()  # doctest: +SKIP
    True
    """

    settings = {'title': '{0} - Transfer Function'.format(colourspace)}
    settings.update(kwargs)

    return multi_transfer_function_plot([colourspace], **settings)


def multi_transfer_function_plot(colourspaces=None,
                                 inverse=False, **kwargs):
    """
    Plots given colourspaces transfer functions.

    Parameters
    ----------
    colourspaces : list, optional
        Colourspaces transfer functions to plot.
    inverse : bool
        Plot inverse transfer functions.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> multi_transfer_function_plot(['sRGB', 'Rec. 709'])  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if colourspaces is None:
        colourspaces = ['sRGB', 'Rec. 709']

    samples = np.linspace(0, 1, 1000)
    for i, colourspace in enumerate(colourspaces):
        colourspace = get_RGB_colourspace(colourspace)

        RGBs = np.array([colourspace.inverse_transfer_function(x)
                         if inverse else
                         colourspace.transfer_function(x)
                         for x in samples])
        pylab.plot(samples,
                   RGBs,
                   label=u'{0}'.format(colourspace.name),
                   linewidth=2)

    settings.update({
        'title': '{0} - Transfer Functions'.format(
            ', '.join(colourspaces)),
        'x_tighten': True,
        'legend': True,
        'legend_location': 'upper left',
        'x_ticker': True,
        'y_ticker': True,
        'grid': True,
        'limits': [0, 1, 0, 1],
        'aspect': 'equal'})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)
