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

from colour.constants import EPSILON
from colour.models import (
    LCHab_to_Lab, Lab_to_XYZ, Luv_to_uv, POINTER_GAMUT_BOUNDARIES,
    POINTER_GAMUT_DATA, POINTER_GAMUT_ILLUMINANT, RGB_to_RGB, RGB_to_XYZ,
    UCS_to_uv, XYZ_to_Luv, XYZ_to_UCS, XYZ_to_xy, xy_to_Luv_uv, xy_to_UCS_uv)
from colour.plotting import (
    COLOUR_STYLE_CONSTANTS, chromaticity_diagram_plot_CIE1931,
    chromaticity_diagram_plot_CIE1960UCS, chromaticity_diagram_plot_CIE1976UCS,
    artist, colour_cycle, get_RGB_colourspace, get_cmfs, override_style,
    render)
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


@override_style()
def RGB_colourspaces_chromaticity_diagram_plot(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable=chromaticity_diagram_plot,
        method='CIE 1931',
        show_whitepoints=True,
        **kwargs):
    """
    Plots given *RGB* colourspaces in the *Chromaticity Diagram* according
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
    show_whitepoints : bool, optional
        Whether to display the *RGB* colourspaces whitepoints.

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
    >>> RGB_colourspaces_chromaticity_diagram_plot(
    ...     ['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
RGB_Colourspaces_Chromaticity_Diagram_Plot.png
        :align: center
        :alt: RGB_colourspaces_chromaticity_diagram_plot
    """

    if colourspaces is None:
        colourspaces = ['ITU-R BT.709', 'ACEScg', 'S-Gamut', 'Pointer Gamut']

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    method = method.upper()

    cmfs, name = get_cmfs(cmfs), cmfs

    title = '{0}\n{1} - {2} Chromaticity Diagram'.format(
        ', '.join(colourspaces), name, method)

    settings = {
        'axes': axes,
        'standalone': False,
        'title': title,
        'method': method,
    }
    settings.update(kwargs)

    chromaticity_diagram_callable(**settings)

    if method == 'CIE 1931':

        def XYZ_to_ij(XYZ, *args):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return XYZ_to_xy(XYZ, *args)

        def xy_to_ij(xy):
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

        def xy_to_ij(xy):
            """
            Converts given *xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return xy_to_UCS_uv(xy)

        x_limit_min, x_limit_max = [-0.1], [0.7]
        y_limit_min, y_limit_max = [-0.2], [0.6]

    elif method == 'CIE 1976 UCS':

        def XYZ_to_ij(XYZ, *args):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return Luv_to_uv(XYZ_to_Luv(XYZ, *args), *args)

        def xy_to_ij(xy):
            """
            Converts given *xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return xy_to_Luv_uv(xy)

        x_limit_min, x_limit_max = [-0.1], [0.7]
        y_limit_min, y_limit_max = [-0.1], [0.7]
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
                method))

    settings = {
        'colour_cycle_count': len(colourspaces)
    }
    settings.update(kwargs)

    cycle = colour_cycle(**settings)

    if 'Pointer Gamut' in colourspaces:
        colourspaces.remove('Pointer Gamut')

        ij = xy_to_ij(np.asarray(POINTER_GAMUT_BOUNDARIES))
        alpha_p = COLOUR_STYLE_CONSTANTS.opacity.high
        colour_p = COLOUR_STYLE_CONSTANTS.colour.brightest
        axes.plot(
            ij[..., 0],
            ij[..., 1],
            label='Pointer\'s Gamut',
            color=colour_p,
            alpha=alpha_p)
        axes.plot(
            (ij[-1][0], ij[0][0]), (ij[-1][1], ij[0][1]),
            color=colour_p,
            alpha=alpha_p)

        XYZ = Lab_to_XYZ(
            LCHab_to_Lab(POINTER_GAMUT_DATA), POINTER_GAMUT_ILLUMINANT)
        ij = XYZ_to_ij(XYZ, POINTER_GAMUT_ILLUMINANT)
        axes.scatter(
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
        P = xy_to_ij(P)
        W = xy_to_ij(colourspace.whitepoint)

        axes.plot(
            (W[0], W[0]), (W[1], W[1]),
            color=(R, G, B),
            label=colourspace.name)

        if show_whitepoints:
            axes.plot((W[0], W[0]), (W[1], W[1]), 'o', color=(R, G, B))

        axes.plot(
            (P[0, 0], P[1, 0]), (P[0, 1], P[1, 1]), 'o-', color=(R, G, B))
        axes.plot(
            (P[1, 0], P[2, 0]), (P[1, 1], P[2, 1]), 'o-', color=(R, G, B))
        axes.plot(
            (P[2, 0], P[0, 0]), (P[2, 1], P[0, 1]), 'o-', color=(R, G, B))

        x_limit_min.append(np.amin(P[..., 0]) - 0.1)
        y_limit_min.append(np.amin(P[..., 1]) - 0.1)
        x_limit_max.append(np.amax(P[..., 0]) + 0.1)
        y_limit_max.append(np.amax(P[..., 1]) + 0.1)

    bounding_box = (
        min(x_limit_min),
        max(x_limit_max),
        min(y_limit_min),
        max(y_limit_max),
    )

    settings.update({
        'standalone': True,
        'legend': True,
        'bounding_box': bounding_box,
    })
    settings.update(kwargs)

    return render(**settings)


@override_style()
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
    >>> RGB_colourspaces_chromaticity_diagram_plot_CIE1931(
    ...     ['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
RGB_Colourspaces_Chromaticity_Diagram_Plot_CIE1931.png
        :align: center
        :alt: RGB_colourspaces_chromaticity_diagram_plot_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return RGB_colourspaces_chromaticity_diagram_plot(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1931, **settings)


@override_style()
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
    >>> RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS((
    ...     ['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
RGB_Colourspaces_Chromaticity_Diagram_Plot_CIE1960UCS.png
        :align: center
        :alt: RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return RGB_colourspaces_chromaticity_diagram_plot(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1960UCS,
        **settings)


@override_style()
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
    >>> RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS((
    ...     ['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
RGB_Colourspaces_Chromaticity_Diagram_Plot_CIE1976UCS.png
        :align: center
        :alt: RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return RGB_colourspaces_chromaticity_diagram_plot(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1976UCS,
        **settings)


@override_style()
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
        Parameters for the :func:`plt.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

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
    >>> RGB = np.random.random((128, 128, 3))
    >>> RGB_chromaticity_coordinates_chromaticity_diagram_plot(
    ...     RGB, 'ITU-R BT.709')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
RGB_Chromaticity_Coordinates_Chromaticity_Diagram_Plot.png
        :align: center
        :alt: RGB_chromaticity_coordinates_chromaticity_diagram_plot
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    method = method.upper()

    scatter_settings = {
        's': 40,
        'c': 'RGB',
        'marker': 'o',
        'alpha': 0.85,
    }
    if scatter_parameters is not None:
        scatter_settings.update(scatter_parameters)

    settings = dict(kwargs)
    settings.update({'axes': axes, 'standalone': False})

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
                COLOUR_STYLE_CONSTANTS.colour.colourspace,
                apply_encoding_cctf=True).reshape(-1, 3), 0, 1)

    XYZ = RGB_to_XYZ(RGB, colourspace.whitepoint, colourspace.whitepoint,
                     colourspace.RGB_to_XYZ_matrix)

    if method == 'CIE 1931':
        ij = XYZ_to_xy(XYZ, colourspace.whitepoint)
    elif method == 'CIE 1960 UCS':
        ij = UCS_to_uv(XYZ_to_UCS(XYZ))

    elif method == 'CIE 1976 UCS':
        ij = Luv_to_uv(
            XYZ_to_Luv(XYZ, colourspace.whitepoint), colourspace.whitepoint)

    axes.scatter(ij[..., 0], ij[..., 1], **scatter_settings)

    settings.update({'standalone': True})
    settings.update(kwargs)

    return render(**settings)


@override_style()
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
        Parameters for the :func:`plt.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

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
    >>> RGB = np.random.random((128, 128, 3))
    >>> RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931(
    ...     RGB, 'ITU-R BT.709')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
RGB_Chromaticity_Coordinates_Chromaticity_Diagram_Plot_CIE1931.png
        :align: center
        :alt: RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return RGB_chromaticity_coordinates_chromaticity_diagram_plot(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1931,
        scatter_parameters=scatter_parameters,
        **settings)


@override_style()
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
        Parameters for the :func:`plt.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

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
    >>> RGB = np.random.random((128, 128, 3))
    >>> RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS(
    ...     RGB, 'ITU-R BT.709')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
RGB_Chromaticity_Coordinates_Chromaticity_Diagram_Plot_CIE1960UCS.png
        :align: center
        :alt: RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return RGB_chromaticity_coordinates_chromaticity_diagram_plot(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1960UCS,
        scatter_parameters=scatter_parameters,
        **settings)


@override_style()
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
        Parameters for the :func:`plt.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

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
    >>> RGB = np.random.random((128, 128, 3))
    >>> RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS(
    ...     RGB, 'ITU-R BT.709')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
RGB_Chromaticity_Coordinates_Chromaticity_Diagram_Plot_CIE1976UCS.png
        :align: center
        :alt: RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return RGB_chromaticity_coordinates_chromaticity_diagram_plot(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1976UCS,
        scatter_parameters=scatter_parameters,
        **settings)


@override_style()
def single_cctf_plot(colourspace='ITU-R BT.709', decoding_cctf=False,
                     **kwargs):
    """
    Plots given colourspace colour component transfer function.

    Parameters
    ----------
    colourspace : unicode, optional
        *RGB* Colourspace colour component transfer function to plot.
    decoding_cctf : bool
        Plot the decoding colour component transfer function instead.

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
    >>> single_cctf_plot('ITU-R BT.709')  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_CCTF_Plot.png
        :align: center
        :alt: single_cctf_plot
    """

    settings = {
        'title':
            '{0} - {1} CCTF'.format(colourspace, 'Decoding'
                                    if decoding_cctf else 'Encoding')
    }
    settings.update(kwargs)

    return multi_cctf_plot([colourspace], decoding_cctf, **settings)


@override_style()
def multi_cctf_plot(colourspaces=None, decoding_cctf=False, **kwargs):
    """
    Plots given colourspaces colour component transfer functions.

    Parameters
    ----------
    colourspaces : array_like, optional
        Colourspaces colour component transfer function to plot.
    decoding_cctf : bool
        Plot the decoding colour component transfer function instead.

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
    >>> multi_cctf_plot(['ITU-R BT.709', 'sRGB'])  # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_CCTF_Plot.png
        :align: center
        :alt: multi_cctf_plot
    """

    if colourspaces is None:
        colourspaces = ('ITU-R BT.709', 'sRGB')

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    samples = np.linspace(0, 1, 1000)
    for colourspace in colourspaces:
        colourspace = get_RGB_colourspace(colourspace)

        RGBs = (colourspace.decoding_cctf(samples)
                if decoding_cctf else colourspace.encoding_cctf(samples))

        axes.plot(samples, RGBs, label=u'{0}'.format(colourspace.name))

    mode = 'Decoding' if decoding_cctf else 'Encoding'
    settings = {
        'axes': axes,
        'aspect': 'equal',
        'bounding_box': (0, 1, 0, 1),
        'legend': True,
        'title': '{0} - {1} CCTFs'.format(', '.join(colourspaces), mode),
        'x_label': 'Signal Value' if decoding_cctf else 'Tristimulus Value',
        'y_label': 'Tristimulus Value' if decoding_cctf else 'Signal Value',
    }
    settings.update(kwargs)

    return render(**settings)
