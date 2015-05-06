#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Models Plotting
======================

Defines the colour models plotting objects:

-   :func:`RGB_colourspaces_CIE_1931_chromaticity_diagram_plot`
-   :func:`RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot`
-   :func:`RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot`
-   :func:`RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot`
-   :func:`RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot`
-   :func:`RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot`
-   :func:`single_transfer_function_plot`
-   :func:`multi_transfer_function_plot`
"""

from __future__ import division

import numpy as np
import pylab

from colour.constants import EPSILON
from colour.models import (
    LCHab_to_Lab,
    Lab_to_XYZ,
    Luv_to_uv,
    POINTER_GAMUT_BOUNDARIES,
    POINTER_GAMUT_DATA,
    POINTER_GAMUT_ILLUMINANT,
    RGB_COLOURSPACES,
    RGB_to_XYZ,
    UCS_to_uv,
    XYZ_to_Luv,
    XYZ_to_UCS,
    XYZ_to_xy,
    xy_to_XYZ)
from colour.plotting import (
    CHROMATICITY_DIAGRAM_DEFAULT_ILLUMINANT,
    CIE_1931_chromaticity_diagram_plot,
    CIE_1960_UCS_chromaticity_diagram_plot,
    CIE_1976_UCS_chromaticity_diagram_plot,
    DEFAULT_FIGURE_WIDTH,
    boundaries,
    canvas,
    colour_cycle,
    decorate,
    display,
    get_cmfs)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'get_RGB_colourspace',
    'RGB_colourspaces_CIE_1931_chromaticity_diagram_plot',
    'RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot',
    'RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot',
    'single_transfer_function_plot',
    'multi_transfer_function_plot']


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
            ('"{0}" colourspace not found in factory colourspaces: '
             '"{1}".').format(name, ', '.join(
                sorted(RGB_COLOURSPACES.keys()))))

    return colourspace


def RGB_colourspaces_CIE_1931_chromaticity_diagram_plot(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots given *RGB* colourspaces in *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces : list, optional
        *RGB* colourspaces to plot.
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
    >>> csps = ['sRGB', 'ACES2065-1']
    >>> RGB_colourspaces_CIE_1931_chromaticity_diagram_plot(csps)  # noqa  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if colourspaces is None:
        colourspaces = ('sRGB', 'ACES2065-1', 'Pointer Gamut')

    cmfs, name = get_cmfs(cmfs), cmfs

    settings = {
        'title': '{0} - {1} - CIE 1931 Chromaticity Diagram'.format(
            ', '.join(colourspaces), name),
        'standalone': False}
    settings.update(kwargs)

    if not CIE_1931_chromaticity_diagram_plot(**settings):
        return

    x_limit_min, x_limit_max = [-0.1], [0.9]
    y_limit_min, y_limit_max = [-0.1], [0.9]

    settings = {'colour_cycle_map': 'rainbow',
                'colour_cycle_count': len(colourspaces)}
    settings.update(kwargs)

    cycle = colour_cycle(**settings)

    for colourspace in colourspaces:
        if colourspace == 'Pointer Gamut':
            xy = np.asarray(POINTER_GAMUT_BOUNDARIES)
            alpha_p, colour_p = 0.85, '0.95'
            pylab.plot(xy[..., 0],
                       xy[..., 1],
                       label='Pointer\'s Gamut',
                       color=colour_p,
                       alpha=alpha_p,
                       linewidth=2)
            pylab.plot((xy[-1][0], xy[0][0]),
                       (xy[-1][1], xy[0][1]),
                       color=colour_p,
                       alpha=alpha_p,
                       linewidth=2)

            XYZ = Lab_to_XYZ(LCHab_to_Lab(POINTER_GAMUT_DATA),
                             POINTER_GAMUT_ILLUMINANT)
            xy = XYZ_to_xy(XYZ, POINTER_GAMUT_ILLUMINANT)
            pylab.scatter(xy[..., 0],
                          xy[..., 1],
                          alpha=alpha_p / 2,
                          color=colour_p,
                          marker='+')

        else:
            colourspace, name = get_RGB_colourspace(
                colourspace), colourspace

            r, g, b, _a = next(cycle)

            primaries = colourspace.primaries
            whitepoint = colourspace.whitepoint

            pylab.plot((whitepoint[0], whitepoint[0]),
                       (whitepoint[1], whitepoint[1]),
                       color=(r, g, b),
                       label=colourspace.name,
                       linewidth=2)
            pylab.plot((whitepoint[0], whitepoint[0]),
                       (whitepoint[1], whitepoint[1]),
                       'o',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[0, 0], primaries[1, 0]),
                       (primaries[0, 1], primaries[1, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[1, 0], primaries[2, 0]),
                       (primaries[1, 1], primaries[2, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[2, 0], primaries[0, 0]),
                       (primaries[2, 1], primaries[0, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)

            x_limit_min.append(np.amin(primaries[..., 0]) - 0.1)
            y_limit_min.append(np.amin(primaries[..., 1]) - 0.1)
            x_limit_max.append(np.amax(primaries[..., 0]) + 0.1)
            y_limit_max.append(np.amax(primaries[..., 1]) + 0.1)

    settings.update({
        'legend': True,
        'legend_location': 'upper right',
        'x_tighten': True,
        'y_tighten': True,
        'limits': (min(x_limit_min), max(x_limit_max),
                   min(y_limit_min), max(y_limit_max)),
        'standalone': True})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots given *RGB* colourspaces in *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces : list, optional
        *RGB* colourspaces to plot.
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
    >>> csps = ['sRGB', 'ACES2065-1']
    >>> RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot(csps)  # noqa  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if colourspaces is None:
        colourspaces = ('sRGB', 'ACES2065-1', 'Pointer Gamut')

    cmfs, name = get_cmfs(cmfs), cmfs

    settings = {
        'title': '{0} - {1} - CIE 1960 UCS Chromaticity Diagram'.format(
            ', '.join(colourspaces), name),
        'standalone': False}
    settings.update(kwargs)

    if not CIE_1960_UCS_chromaticity_diagram_plot(**settings):
        return

    x_limit_min, x_limit_max = [-0.1], [0.7]
    y_limit_min, y_limit_max = [-0.2], [0.6]

    settings = {'colour_cycle_map': 'rainbow',
                'colour_cycle_count': len(colourspaces)}
    settings.update(kwargs)

    cycle = colour_cycle(**settings)

    for colourspace in colourspaces:
        if colourspace == 'Pointer Gamut':
            uv = UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(POINTER_GAMUT_BOUNDARIES)))
            alpha_p, colour_p = 0.85, '0.95'
            pylab.plot(uv[..., 0],
                       uv[..., 1],
                       label='Pointer\'s Gamut',
                       color=colour_p,
                       alpha=alpha_p,
                       linewidth=2)
            pylab.plot((uv[-1][0], uv[0][0]),
                       (uv[-1][1], uv[0][1]),
                       color=colour_p,
                       alpha=alpha_p,
                       linewidth=2)

            XYZ = Lab_to_XYZ(LCHab_to_Lab(POINTER_GAMUT_DATA),
                             POINTER_GAMUT_ILLUMINANT)
            uv = UCS_to_uv(XYZ_to_UCS(XYZ))
            pylab.scatter(uv[..., 0],
                          uv[..., 1],
                          alpha=alpha_p / 2,
                          color=colour_p,
                          marker='+')

        else:
            colourspace, name = get_RGB_colourspace(
                colourspace), colourspace

            r, g, b, _a = next(cycle)

            # RGB colourspaces such as *ACES2065-1* have primaries with
            # chromaticity coordinates set to 0 thus we prevent nan from being
            # yield by zero division in later colour transformations.
            primaries = np.where(colourspace.primaries == 0,
                                 EPSILON,
                                 colourspace.primaries)

            primaries = UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(primaries)))
            whitepoint = UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(
                colourspace.whitepoint)))

            pylab.plot((whitepoint[0], whitepoint[0]),
                       (whitepoint[1], whitepoint[1]),
                       color=(r, g, b),
                       label=colourspace.name,
                       linewidth=2)
            pylab.plot((whitepoint[0], whitepoint[0]),
                       (whitepoint[1], whitepoint[1]),
                       'o',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[0, 0], primaries[1, 0]),
                       (primaries[0, 1], primaries[1, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[1, 0], primaries[2, 0]),
                       (primaries[1, 1], primaries[2, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[2, 0], primaries[0, 0]),
                       (primaries[2, 1], primaries[0, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)

            x_limit_min.append(np.amin(primaries[..., 0]) - 0.1)
            y_limit_min.append(np.amin(primaries[..., 1]) - 0.1)
            x_limit_max.append(np.amax(primaries[..., 0]) + 0.1)
            y_limit_max.append(np.amax(primaries[..., 1]) + 0.1)

    settings.update({
        'legend': True,
        'legend_location': 'upper right',
        'x_tighten': True,
        'y_tighten': True,
        'limits': (min(x_limit_min), max(x_limit_max),
                   min(y_limit_min), max(y_limit_max)),
        'standalone': True})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots given *RGB* colourspaces in *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces : list, optional
        *RGB* colourspaces to plot.
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
    >>> csps = ['sRGB', 'ACES2065-1']
    >>> RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot(csps)  # noqa  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if colourspaces is None:
        colourspaces = ('sRGB', 'ACES2065-1', 'Pointer Gamut')

    cmfs, name = get_cmfs(cmfs), cmfs

    illuminant = CHROMATICITY_DIAGRAM_DEFAULT_ILLUMINANT

    settings = {
        'title': '{0} - {1} - CIE 1976 UCS Chromaticity Diagram'.format(
            ', '.join(colourspaces), name),
        'standalone': False}
    settings.update(kwargs)

    if not CIE_1976_UCS_chromaticity_diagram_plot(**settings):
        return

    x_limit_min, x_limit_max = [-0.1], [0.7]
    y_limit_min, y_limit_max = [-0.1], [0.7]

    settings = {'colour_cycle_map': 'rainbow',
                'colour_cycle_count': len(colourspaces)}
    settings.update(kwargs)

    cycle = colour_cycle(**settings)

    for colourspace in colourspaces:
        if colourspace == 'Pointer Gamut':
            uv = Luv_to_uv(XYZ_to_Luv(xy_to_XYZ(
                POINTER_GAMUT_BOUNDARIES), illuminant), illuminant)
            alpha_p, colour_p = 0.85, '0.95'
            pylab.plot(uv[..., 0],
                       uv[..., 1],
                       label='Pointer\'s Gamut',
                       color=colour_p,
                       alpha=alpha_p,
                       linewidth=2)
            pylab.plot((uv[-1][0], uv[0][0]),
                       (uv[-1][1], uv[0][1]),
                       color=colour_p,
                       alpha=alpha_p,
                       linewidth=2)

            XYZ = Lab_to_XYZ(LCHab_to_Lab(POINTER_GAMUT_DATA),
                             POINTER_GAMUT_ILLUMINANT)
            uv = Luv_to_uv(XYZ_to_Luv(XYZ, illuminant), illuminant)
            pylab.scatter(uv[..., 0],
                          uv[..., 1],
                          alpha=alpha_p / 2,
                          color=colour_p,
                          marker='+')

        else:
            colourspace, name = get_RGB_colourspace(
                colourspace), colourspace

            r, g, b, _a = next(cycle)

            # RGB colourspaces such as *ACES2065-1* have primaries with
            # chromaticity coordinates set to 0 thus we prevent nan from being
            # yield by zero division in later colour transformations.
            primaries = np.where(colourspace.primaries == 0,
                                 EPSILON,
                                 colourspace.primaries)

            primaries = Luv_to_uv(XYZ_to_Luv(xy_to_XYZ(
                primaries), illuminant), illuminant)
            whitepoint = Luv_to_uv(XYZ_to_Luv(xy_to_XYZ(
                colourspace.whitepoint), illuminant), illuminant)

            pylab.plot((whitepoint[0], whitepoint[0]),
                       (whitepoint[1], whitepoint[1]),
                       color=(r, g, b),
                       label=colourspace.name,
                       linewidth=2)
            pylab.plot((whitepoint[0], whitepoint[0]),
                       (whitepoint[1], whitepoint[1]),
                       'o',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[0, 0], primaries[1, 0]),
                       (primaries[0, 1], primaries[1, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[1, 0], primaries[2, 0]),
                       (primaries[1, 1], primaries[2, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)
            pylab.plot((primaries[2, 0], primaries[0, 0]),
                       (primaries[2, 1], primaries[0, 1]),
                       'o-',
                       color=(r, g, b),
                       linewidth=2)

            x_limit_min.append(np.amin(primaries[..., 0]) - 0.1)
            y_limit_min.append(np.amin(primaries[..., 1]) - 0.1)
            x_limit_max.append(np.amax(primaries[..., 0]) + 0.1)
            y_limit_max.append(np.amax(primaries[..., 1]) + 0.1)

    settings.update({
        'legend': True,
        'legend_location': 'upper right',
        'x_tighten': True,
        'y_tighten': True,
        'limits': (min(x_limit_min), max(x_limit_max),
                   min(y_limit_min), max(y_limit_max)),
        'standalone': True})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot(
        RGB,
        colourspace,
        **kwargs):
    """
    Plots given *RGB* colourspace array in *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    colourspace : RGB_Colourspace
        *RGB* colourspace of the *RGB* array.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> RGB = np.random.random((10, 10, 3))
    >>> RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot(RGB)  # noqa  # doctest: +SKIP
    True
    """

    settings = {}
    settings.update(kwargs)

    settings['colourspaces'] = (
        [colourspace.name] + settings.get('colourspaces', []))
    RGB_colourspaces_CIE_1931_chromaticity_diagram_plot(
        standalone=False, **settings)

    alpha_p, colour_p = 0.85, 'black'

    xy = XYZ_to_xy(RGB_to_XYZ(RGB,
                              colourspace.whitepoint,
                              colourspace.whitepoint,
                              colourspace.RGB_to_XYZ_matrix),
                   colourspace.whitepoint)

    pylab.scatter(xy[..., 0],
                  xy[..., 1],
                  alpha=alpha_p / 2,
                  color=colour_p,
                  marker='+')

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot(
        RGB,
        colourspace,
        **kwargs):
    """
    Plots given *RGB* colourspace array in *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    colourspace : RGB_Colourspace
        *RGB* colourspace of the *RGB* array.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> RGB = np.random.random((10, 10, 3))
    >>> RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot(RGB)  # noqa  # doctest: +SKIP
    True
    """

    settings = {}
    settings.update(kwargs)

    settings['colourspaces'] = (
        [colourspace.name] + settings.get('colourspaces', []))
    RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot(
        standalone=False, **settings)

    alpha_p, colour_p = 0.85, 'black'

    uv = UCS_to_uv(XYZ_to_UCS(RGB_to_XYZ(RGB,
                                         colourspace.whitepoint,
                                         colourspace.whitepoint,
                                         colourspace.RGB_to_XYZ_matrix)))

    pylab.scatter(uv[..., 0],
                  uv[..., 1],
                  alpha=alpha_p / 2,
                  color=colour_p,
                  marker='+')

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot(
        RGB,
        colourspace,
        **kwargs):
    """
    Plots given *RGB* colourspace array in *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    colourspace : RGB_Colourspace
        *RGB* colourspace of the *RGB* array.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> RGB = np.random.random((10, 10, 3))
    >>> RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot(RGB)  # noqa  # doctest: +SKIP
    True
    """

    settings = {}
    settings.update(kwargs)

    settings['colourspaces'] = (
        [colourspace.name] + settings.get('colourspaces', []))
    RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot(
        standalone=False, **settings)

    alpha_p, colour_p = 0.85, 'black'

    uv = Luv_to_uv(XYZ_to_Luv(RGB_to_XYZ(RGB,
                                         colourspace.whitepoint,
                                         colourspace.whitepoint,
                                         colourspace.RGB_to_XYZ_matrix),
                              colourspace.whitepoint),
                   colourspace.whitepoint)

    pylab.scatter(uv[..., 0],
                  uv[..., 1],
                  alpha=alpha_p / 2,
                  color=colour_p,
                  marker='+')

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
    for colourspace in colourspaces:
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
        'limits': (0, 1, 0, 1),
        'aspect': 'equal'})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)
