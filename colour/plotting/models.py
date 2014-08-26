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

import random
import numpy as np
import pylab

from colour.models import POINTER_GAMUT_DATA, RGB_COLOURSPACES
from colour.plotting import (
    CIE_1931_chromaticity_diagram_plot,
    aspect,
    bounding_box,
    display,
    figure_size,
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
    colourspace : Unicode
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
             '"{1}".').format(name, sorted(RGB_COLOURSPACES.keys())))

    return colourspace


@figure_size((8, 8))
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

    if colourspaces is None:
        colourspaces = ('sRGB', 'ACES RGB', 'Pointer Gamut')

    cmfs, name = get_cmfs(cmfs), cmfs

    settings = {'title': '{0} - {1}'.format(', '.join(colourspaces), name),
                'standalone': False}
    settings.update(kwargs)

    if not CIE_1931_chromaticity_diagram_plot(**settings):
        return

    x_limit_min, x_limit_max = [-0.1], [0.9]
    y_limit_min, y_limit_max = [-0.1], [0.9]
    for colourspace in colourspaces:
        if colourspace == 'Pointer Gamut':
            x, y = tuple(zip(*POINTER_GAMUT_DATA))
            pylab.plot(x,
                       y,
                       label='Pointer Gamut',
                       color='0.95',
                       linewidth=2)
            pylab.plot([x[-1],
                        x[0]],
                       [y[-1],
                        y[0]],
                       color='0.95',
                       linewidth=2)
        else:
            colourspace, name = get_RGB_colourspace(
                colourspace), colourspace

            random_colour = lambda: float(random.randint(64, 224)) / 255
            r, g, b = random_colour(), random_colour(), random_colour()

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

    settings.update({'legend': True,
                     'legend_location': 'upper right',
                     'x_tighten': True,
                     'y_tighten': True,
                     'limits': [min(x_limit_min), max(x_limit_max),
                                min(y_limit_min), max(y_limit_max)],
                     'margins': [-0.05, 0.05, -0.05, 0.05],
                     'standalone': True})

    bounding_box(**settings)
    aspect(**settings)

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


@figure_size((8, 8))
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

    if colourspaces is None:
        colourspaces = ['sRGB', 'Rec. 709']

    samples = np.linspace(0, 1, 1000)
    for i, colourspace in enumerate(colourspaces):
        colourspace, name = get_RGB_colourspace(colourspace), colourspace

        RGBs = np.array([colourspace.inverse_transfer_function(x)
                         if inverse else
                         colourspace.transfer_function(x)
                         for x in samples])
        pylab.plot(samples,
                   RGBs,
                   label=u'{0}'.format(colourspace.name),
                   linewidth=2)

    settings = {
        'title': '{0} - Transfer Functions'.format(
            ', '.join(colourspaces)),
        'x_tighten': True,
        'legend': True,
        'legend_location': 'upper left',
        'x_ticker': True,
        'y_ticker': True,
        'grid': True,
        'limits': [0, 1, 0, 1]}

    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)

    return display(**settings)
