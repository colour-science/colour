# -*- coding: utf-8 -*-
"""
Colour Models Plotting
======================

Defines the colour models plotting objects:

-   :func:`colour.plotting.\
RGB_colourspaces_chromaticity_diagram_plot_CIE1931`
-   :func:`colour.plotting.\
RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS`
-   :func:`colour.plotting.\
RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS`
-   :func:`colour.plotting.\
RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931`
-   :func:`colour.plotting.\
RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS`
-   :func:`colour.plotting.\
RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS`
-   :func:`colour.plotting.single_cctf_plot`
-   :func:`colour.plotting.multi_cctf_plot`
"""

from __future__ import division

import numpy as np
import pylab

from colour.constants import EPSILON
from colour.models import (
    LCHab_to_Lab, Lab_to_XYZ, Luv_to_uv, POINTER_GAMUT_BOUNDARIES,
    POINTER_GAMUT_DATA, POINTER_GAMUT_ILLUMINANT, RGB_to_RGB, RGB_to_XYZ,
    UCS_to_uv, XYZ_to_Luv, XYZ_to_UCS, XYZ_to_xy, xy_to_XYZ)
from colour.plotting import (
    DEFAULT_FIGURE_WIDTH, DEFAULT_PLOTTING_COLOURSPACE,
    chromaticity_diagram_plot_CIE1931, chromaticity_diagram_plot_CIE1960UCS,
    chromaticity_diagram_plot_CIE1976UCS, canvas, colour_cycle,
    get_RGB_colourspace, get_cmfs, render)
from colour.plotting.diagrams import chromaticity_diagram_plot

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RGB_colourspaces_chromaticity_diagram_plot',
    'RGB_colourspaces_chromaticity_diagram_plot_CIE1931',
    'RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS',
    'RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS',
    'RGB_chromaticity_coordinates_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931',
    'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS',
    'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS',
    'single_cctf_plot', 'multi_cctf_plot'
]


def RGB_colourspaces_chromaticity_diagram_plot(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable=chromaticity_diagram_plot,
        method='CIE 1931',
        **kwargs):
    """
    Plots given *RGB* colourspaces in the *Chromaticity Diagram* accordingly
    to given method.

    Parameters
    ----------
    colourspaces : array_like, optional
        *RGB* colourspaces to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    chromaticity_diagram_callable : callable, optional
        Callable responsible for drawing the *Chromaticity Diagram*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> c = ['ITU-R Rec. 709', 'ACEScg', 'S-Gamut']
    >>> RGB_colourspaces_chromaticity_diagram_plot(c)
    ... # doctest: +SKIP
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if colourspaces is None:
        colourspaces = ['ITU-R BT.709', 'ACEScg', 'S-Gamut', 'Pointer Gamut']

    cmfs, name = get_cmfs(cmfs), cmfs

    illuminant = DEFAULT_PLOTTING_COLOURSPACE.whitepoint

    method = method.upper()
    settings = {
        'method':
            method,
        'title':
            '{0}\n{1} - {2} Chromaticity Diagram'.format(
                ', '.join(colourspaces), name, method),
        'standalone':
            False
    }
    settings.update(kwargs)

    chromaticity_diagram_callable(**settings)

    method = method.upper()
    if method == 'CIE 1931':

        def XYZ_to_ij(XYZ, *args):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return XYZ_to_xy(XYZ, *args)

        def xy_to_ij(xy, *args):
            """
            Converts given *xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return xy

        x_limit_min, x_limit_max = [-0.1], [0.9]
        y_limit_min, y_limit_max = [-0.1], [0.9]
    elif method == 'CIE 1960 UCS':

        def XYZ_to_ij(XYZ, *args):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(XYZ))

        def xy_to_ij(xy, *args):
            """
            Converts given *xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return XYZ_to_ij(xy_to_XYZ(xy), *args)

        x_limit_min, x_limit_max = [-0.1], [0.7]
        y_limit_min, y_limit_max = [-0.2], [0.6]

    elif method == 'CIE 1976 UCS':

        def XYZ_to_ij(XYZ, *args):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return Luv_to_uv(XYZ_to_Luv(XYZ, *args), *args)

        def xy_to_ij(xy, *args):
            """
            Converts given *xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return XYZ_to_ij(xy_to_XYZ(xy), *args)

        x_limit_min, x_limit_max = [-0.1], [0.7]
        y_limit_min, y_limit_max = [-0.1], [0.7]
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
                method))

    settings = {
        'colour_cycle_map': 'rainbow',
        'colour_cycle_count': len(colourspaces)
    }
    settings.update(kwargs)

    cycle = colour_cycle(**settings)

    if 'Pointer Gamut' in colourspaces:
        colourspaces.remove('Pointer Gamut')

        ij = xy_to_ij(np.asarray(POINTER_GAMUT_BOUNDARIES), illuminant)
        alpha_p, colour_p = 0.85, '0.95'
        pylab.plot(
            ij[..., 0],
            ij[..., 1],
            label='Pointer\'s Gamut',
            color=colour_p,
            alpha=alpha_p,
            linewidth=1)
        pylab.plot(
            (ij[-1][0], ij[0][0]), (ij[-1][1], ij[0][1]),
            color=colour_p,
            alpha=alpha_p,
            linewidth=1)

        XYZ = Lab_to_XYZ(
            LCHab_to_Lab(POINTER_GAMUT_DATA), POINTER_GAMUT_ILLUMINANT)
        ij = XYZ_to_ij(XYZ, POINTER_GAMUT_ILLUMINANT)
        pylab.scatter(
            ij[..., 0],
            ij[..., 1],
            alpha=alpha_p / 2,
            color=colour_p,
            marker='+')

    for colourspace in colourspaces:
        colourspace, name = get_RGB_colourspace(colourspace), colourspace

        R, G, B, _A = next(cycle)

        # RGB colourspaces such as *ACES2065-1* have primaries with
        # chromaticity coordinates set to 0 thus we prevent nan from being
        # yield by zero division in later colour transformations.
        P = np.where(colourspace.primaries == 0, EPSILON,
                     colourspace.primaries)
        P = xy_to_ij(P, illuminant)
        W = xy_to_ij(colourspace.whitepoint, illuminant)

        pylab.plot(
            (W[0], W[0]), (W[1], W[1]),
            color=(R, G, B),
            label=colourspace.name,
            linewidth=1)
        pylab.plot(
            (W[0], W[0]), (W[1], W[1]), 'o', color=(R, G, B), linewidth=1)
        pylab.plot(
            (P[0, 0], P[1, 0]), (P[0, 1], P[1, 1]),
            'o-',
            color=(R, G, B),
            linewidth=1)
        pylab.plot(
            (P[1, 0], P[2, 0]), (P[1, 1], P[2, 1]),
            'o-',
            color=(R, G, B),
            linewidth=1)
        pylab.plot(
            (P[2, 0], P[0, 0]), (P[2, 1], P[0, 1]),
            'o-',
            color=(R, G, B),
            linewidth=1)

        x_limit_min.append(np.amin(P[..., 0]) - 0.1)
        y_limit_min.append(np.amin(P[..., 1]) - 0.1)
        x_limit_max.append(np.amax(P[..., 0]) + 0.1)
        y_limit_max.append(np.amax(P[..., 1]) + 0.1)

    settings.update({
        'legend':
            True,
        'legend_location':
            'upper right',
        'x_tighten':
            True,
        'y_tighten':
            True,
        'limits': [
            min(x_limit_min),
            max(x_limit_max),
            min(y_limit_min),
            max(y_limit_max),
        ],
        'standalone':
            True
    })
    settings.update(kwargs)

    return render(**settings)


def RGB_colourspaces_chromaticity_diagram_plot_CIE1931(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1931=(
            chromaticity_diagram_plot_CIE1931),
        **kwargs):
    """
    Plots given *RGB* colourspaces in the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces : array_like, optional
        *RGB* colourspaces to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> c = ['ITU-R Rec. 709', 'ACEScg', 'S-Gamut']
    >>> RGB_colourspaces_chromaticity_diagram_plot_CIE1931(c)
    ... # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return RGB_colourspaces_chromaticity_diagram_plot(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1931, **settings)


def RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1960UCS=(
            chromaticity_diagram_plot_CIE1960UCS),
        **kwargs):
    """
    Plots given *RGB* colourspaces in the *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces : array_like, optional
        *RGB* colourspaces to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> c = ['ITU-R Rec. 709', 'ACEScg', 'S-Gamut']
    >>> RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS(c)
    ... # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return RGB_colourspaces_chromaticity_diagram_plot(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1960UCS,
        **settings)


def RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1976UCS=(
            chromaticity_diagram_plot_CIE1976UCS),
        **kwargs):
    """
    Plots given *RGB* colourspaces in the *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces : array_like, optional
        *RGB* colourspaces to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    chromaticity_diagram_callable_CIE1976UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> c = ['ITU-R Rec. 709', 'ACEScg', 'S-Gamut']
    >>> RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS(c)
    ... # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return RGB_colourspaces_chromaticity_diagram_plot(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1976UCS,
        **settings)


def RGB_chromaticity_coordinates_chromaticity_diagram_plot(
        RGB,
        colourspace='sRGB',
        chromaticity_diagram_callable=(
            RGB_colourspaces_chromaticity_diagram_plot),
        method='CIE 1931',
        scatter_parameters=None,
        **kwargs):
    """
    Plots given *RGB* colourspace array in the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    colourspace : optional, unicode
        *RGB* colourspace of the *RGB* array.
    chromaticity_diagram_callable : callable, optional
        Callable responsible for drawing the *Chromaticity Diagram*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.
    scatter_parameters : dict, optional
        Parameters for the :func:`pylab.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> RGB = np.random.random((10, 10, 3))
    >>> c = 'ITU-R Rec. 709'
    >>> RGB_chromaticity_coordinates_chromaticity_diagram_plot(RGB, c)
    ... # doctest: +SKIP
    """

    scatter_settings = {
        's': 40,
        'c': 'RGB',
        'marker': 'o',
        'alpha': 0.85,
        'linewidths': 0.1,
        'edgecolors': 'black',
    }
    if scatter_parameters is not None:
        scatter_settings.update(scatter_parameters)

    settings = dict(kwargs)
    settings.update({'standalone': False})

    colourspace, name = get_RGB_colourspace(colourspace), colourspace
    settings['colourspaces'] = [name] + settings.get('colourspaces', [])

    chromaticity_diagram_callable(**settings)

    use_RGB_colours = scatter_settings['c'].upper() == 'RGB'
    if use_RGB_colours:
        RGB = RGB[RGB[:, 1].argsort()]
        scatter_settings['c'] = np.clip(
            RGB_to_RGB(
                RGB,
                colourspace,
                DEFAULT_PLOTTING_COLOURSPACE,
                apply_encoding_cctf=True).reshape(-1, 3), 0, 1)

    XYZ = RGB_to_XYZ(RGB, colourspace.whitepoint, colourspace.whitepoint,
                     colourspace.RGB_to_XYZ_matrix)

    method = method.upper()
    if method == 'CIE 1931':
        ij = XYZ_to_xy(XYZ, colourspace.whitepoint)
    elif method == 'CIE 1960 UCS':
        ij = UCS_to_uv(XYZ_to_UCS(XYZ))

    elif method == 'CIE 1976 UCS':
        ij = Luv_to_uv(
            XYZ_to_Luv(XYZ, colourspace.whitepoint), colourspace.whitepoint)

    pylab.scatter(ij[..., 0], ij[..., 1], **scatter_settings)

    settings.update({'standalone': True})
    settings.update(kwargs)

    return render(**settings)


def RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931(
        RGB,
        colourspace='sRGB',
        chromaticity_diagram_callable_CIE1931=(
            RGB_colourspaces_chromaticity_diagram_plot_CIE1931),
        scatter_parameters=None,
        **kwargs):
    """
    Plots given *RGB* colourspace array in the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    colourspace : optional, unicode
        *RGB* colourspace of the *RGB* array.
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    scatter_parameters : dict, optional
        Parameters for the :func:`pylab.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> RGB = np.random.random((10, 10, 3))
    >>> c = 'ITU-R Rec. 709'
    >>> RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931(RGB, c)
    ... # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return RGB_chromaticity_coordinates_chromaticity_diagram_plot(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1931,
        scatter_parameters=scatter_parameters,
        **settings)


def RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS(
        RGB,
        colourspace='sRGB',
        chromaticity_diagram_callable_CIE1960UCS=(
            RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS),
        scatter_parameters=None,
        **kwargs):
    """
    Plots given *RGB* colourspace array in the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    colourspace : optional, unicode
        *RGB* colourspace of the *RGB* array.
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.
    scatter_parameters : dict, optional
        Parameters for the :func:`pylab.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> RGB = np.random.random((10, 10, 3))
    >>> c = 'ITU-R BT.709'
    >>> RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS(
    ...     RGB, c)  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return RGB_chromaticity_coordinates_chromaticity_diagram_plot(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1960UCS,
        scatter_parameters=scatter_parameters,
        **settings)


def RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS(
        RGB,
        colourspace='sRGB',
        chromaticity_diagram_callable_CIE1976UCS=(
            RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS),
        scatter_parameters=None,
        **kwargs):
    """
    Plots given *RGB* colourspace array in the
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    colourspace : optional, unicode
        *RGB* colourspace of the *RGB* array.
    chromaticity_diagram_callable_CIE1976UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.
    scatter_parameters : dict, optional
        Parameters for the :func:`pylab.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> RGB = np.random.random((10, 10, 3))
    >>> c = 'ITU-R BT.709'
    >>> RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS(
    ...     RGB, c)  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return RGB_chromaticity_coordinates_chromaticity_diagram_plot(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1976UCS,
        scatter_parameters=scatter_parameters,
        **settings)


def single_cctf_plot(colourspace='ITU-R BT.709', decoding_cctf=False,
                     **kwargs):
    """
    Plots given colourspace colour component transfer function.

    Parameters
    ----------
    colourspace : unicode, optional
        *RGB* Colourspace colour component transfer function to plot.
    decoding_cctf : bool
        Plot decoding colour component transfer function instead.

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
    >>> single_cctf_plot()  # doctest: +SKIP
    """

    settings = {
        'title':
            '{0} - {1} CCTF'.format(colourspace, 'Decoding'
                                    if decoding_cctf else 'Encoding')
    }
    settings.update(kwargs)

    return multi_cctf_plot([colourspace], decoding_cctf, **settings)


def multi_cctf_plot(colourspaces=None, decoding_cctf=False, **kwargs):
    """
    Plots given colourspaces colour component transfer functions.

    Parameters
    ----------
    colourspaces : array_like, optional
        Colourspaces colour component transfer function to plot.
    decoding_cctf : bool
        Plot decoding colour component transfer function instead.

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
    >>> multi_cctf_plot(['ITU-R BT.709', 'sRGB'])  # doctest: +SKIP
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if colourspaces is None:
        colourspaces = ('ITU-R BT.709', 'sRGB')

    samples = np.linspace(0, 1, 1000)
    for colourspace in colourspaces:
        colourspace = get_RGB_colourspace(colourspace)

        RGBs = (colourspace.decoding_cctf(samples)
                if decoding_cctf else colourspace.encoding_cctf(samples))

        pylab.plot(
            samples, RGBs, label=u'{0}'.format(colourspace.name), linewidth=1)

    mode = 'Decoding' if decoding_cctf else 'Encoding'
    settings.update({
        'title': '{0} - {1} CCTFs'.format(', '.join(colourspaces), mode),
        'x_tighten': True,
        'x_label': 'Signal Value' if decoding_cctf else 'Tristimulus Value',
        'y_label': 'Tristimulus Value' if decoding_cctf else 'Signal Value',
        'legend': True,
        'legend_location': 'upper left',
        'grid': True,
        'limits': (0, 1, 0, 1),
        'aspect': 'equal'
    })
    settings.update(kwargs)

    return render(**settings)
