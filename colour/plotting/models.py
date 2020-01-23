# -*- coding: utf-8 -*-
"""
Colour Models Plotting
======================

Defines the colour models plotting objects:

-   :func:`colour.plotting.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS`
-   :func:`colour.plotting.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS`
-   :func:`colour.plotting.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS`
-   :func:`colour.plotting.plot_single_cctf`
-   :func:`colour.plotting.plot_multi_cctfs`
-   :func:`colour.plotting.plot_constant_hue_loci`

References
----------
-   :cite:`Ebner1998` : Ebner, F., & Fairchild, M. D. (1998). Finding constant
    hue surfaces in color space. In G. B. Beretta & R. Eschbach (Eds.), Proc.
    SPIE 3300, Color Imaging: Device-Independent Color, Color Hardcopy, and
    Graphic Arts III, (2 January 1998) (pp. 107-117). doi:10.1117/12.298269
-   :cite:`Hung1995` : Hung, P.-C., & Berns, R. S. (1995). Determination of
    constant Hue Loci for a CRT gamut and their predictions using color
    appearance spaces. Color Research & Application, 20(5), 285-295.
    doi:10.1002/col.5080200506
-   :cite:`Mansencal2019` : Mansencal, T. (2019). Colour - Datasets.
    doi:10.5281/zenodo.3362520
"""

from __future__ import division

import numpy as np
import scipy.optimize
try:  # pragma: no cover
    from collections import Mapping
except ImportError:  # pragma: no cover
    from collections.abc import Mapping

from matplotlib.patches import Ellipse
from matplotlib.path import Path

from colour.constants import EPSILON
from colour.algebra import (point_at_angle_on_ellipse,
                            ellipse_coefficients_canonical_form,
                            ellipse_fitting)
from colour.graph import convert
from colour.models import (
    COLOURSPACE_MODELS_AXIS_LABELS, CCTF_ENCODINGS, CCTF_DECODINGS,
    LCHab_to_Lab, Lab_to_XYZ, Luv_to_uv, MACADAM_1942_ELLIPSES_DATA,
    POINTER_GAMUT_BOUNDARIES, POINTER_GAMUT_DATA, POINTER_GAMUT_ILLUMINANT,
    RGB_to_RGB, RGB_to_XYZ, UCS_to_uv, XYZ_to_Luv, XYZ_to_RGB, XYZ_to_UCS,
    XYZ_to_xy, xy_to_Luv_uv, xy_to_UCS_uv)
from colour.plotting import (
    COLOUR_STYLE_CONSTANTS, plot_chromaticity_diagram_CIE1931, artist,
    plot_chromaticity_diagram_CIE1960UCS, plot_chromaticity_diagram_CIE1976UCS,
    colour_cycle, colour_style, filter_passthrough, filter_RGB_colourspaces,
    filter_cmfs, plot_multi_functions, override_style, render)
from colour.plotting.diagrams import plot_chromaticity_diagram
from colour.utilities import (as_float_array, as_int_array, domain_range_scale,
                              first_item, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'common_colourspace_model_axis_reorder', 'plot_pointer_gamut',
    'plot_RGB_colourspaces_in_chromaticity_diagram',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS',
    'plot_RGB_chromaticities_in_chromaticity_diagram',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS',
    'ellipses_MacAdam1942',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS',
    'plot_single_cctf', 'plot_multi_cctfs', 'plot_constant_hue_loci'
]


def common_colourspace_model_axis_reorder(a, model=None):
    """
    Reorder the axes of given colourspace model :math:`a` array according to
    the most common volume plotting axes order.

    Parameters
    ----------
    a : array_like
        Colourspace model :math:`a` array.
    model : unicode, optional
        **{'CIE XYZ', 'CIE xyY', 'CIE xy', 'CIE Lab', 'CIE LCHab', 'CIE Luv',
        'CIE Luv uv', 'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW',
        'DIN 99', 'Hunter Lab', 'Hunter Rdab', 'IPT', 'JzAzBz', 'OSA UCS',
        'hdr-CIELAB', 'hdr-IPT'}**,
        Colourspace model.

    Returns
    -------
    ndarray
        Reordered colourspace model :math:`a` array.

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> common_colourspace_model_axis_reorder(a)
    array([0, 1, 2])
    >>> common_colourspace_model_axis_reorder(a, 'CIE Lab')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'CIE LCHab')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'CIE Luv')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'CIE LCHab')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'DIN 99')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'Hunter Lab')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'Hunter Rdab')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'IPT')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'JzAzBz')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'OSA UCS')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'hdr-CIELAB')
    array([ 1.,  2.,  0.])
    >>> common_colourspace_model_axis_reorder(a, 'hdr-IPT')
    array([ 1.,  2.,  0.])
    """

    if model in ('CIE Lab', 'CIE LCHab', 'CIE Luv', 'CIE LCHuv', 'DIN 99',
                 'Hunter Lab', 'Hunter Rdab', 'IPT', 'JzAzBz', 'OSA UCS',
                 'hdr-CIELAB', 'hdr-IPT'):
        i, j, k = tsplit(a)
        a = tstack([j, k, i])

    return a


@override_style()
def plot_pointer_gamut(method='CIE 1931', **kwargs):
    """
    Plots *Pointer's Gamut* according to given method.

    Parameters
    ----------
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        Plotting method.

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
    >>> plot_pointer_gamut()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Pointer_Gamut.png
        :align: center
        :alt: plot_pointer_gamut
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    if method == 'CIE 1931':

        def XYZ_to_ij(XYZ, *args):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return XYZ_to_xy(XYZ, *args)

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy

    elif method == 'CIE 1960 UCS':

        def XYZ_to_ij(XYZ, *args):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(XYZ))

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_UCS_uv(xy)

    elif method == 'CIE 1976 UCS':

        def XYZ_to_ij(XYZ, *args):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return Luv_to_uv(XYZ_to_Luv(XYZ, *args), *args)

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_Luv_uv(xy)

    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}}'.format(
                method))

    ij = xy_to_ij(as_float_array(POINTER_GAMUT_BOUNDARIES))
    alpha_p = COLOUR_STYLE_CONSTANTS.opacity.high
    colour_p = COLOUR_STYLE_CONSTANTS.colour.darkest
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
        ij[..., 0], ij[..., 1], alpha=alpha_p / 2, color=colour_p, marker='+')

    settings.update({'axes': axes})
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_RGB_colourspaces_in_chromaticity_diagram(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable=plot_chromaticity_diagram,
        method='CIE 1931',
        show_whitepoints=True,
        show_pointer_gamut=False,
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
    show_pointer_gamut : bool, optional
        Whether to display the *Pointer's Gamut*.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.plot_pointer_gamut`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_RGB_colourspaces_in_chromaticity_diagram(
    ...     ['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Colourspaces_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_RGB_colourspaces_in_chromaticity_diagram
    """

    if colourspaces is None:
        colourspaces = ['ITU-R BT.709', 'ACEScg', 'S-Gamut']

    colourspaces = filter_RGB_colourspaces(colourspaces).values()

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    cmfs = first_item(filter_cmfs(cmfs).values())

    title = '{0}\n{1} - {2} Chromaticity Diagram'.format(
        ', '.join([colourspace.name for colourspace in colourspaces]),
        cmfs.name, method)

    settings = {'axes': axes, 'title': title, 'method': method}
    settings.update(kwargs)
    settings['standalone'] = False

    chromaticity_diagram_callable(**settings)

    if show_pointer_gamut:
        settings = {'axes': axes, 'method': method}
        settings.update(kwargs)
        settings['standalone'] = False

        plot_pointer_gamut(**settings)

    if method == 'CIE 1931':

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy

        x_limit_min, x_limit_max = [-0.1], [0.9]
        y_limit_min, y_limit_max = [-0.1], [0.9]
    elif method == 'CIE 1960 UCS':

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_UCS_uv(xy)

        x_limit_min, x_limit_max = [-0.1], [0.7]
        y_limit_min, y_limit_max = [-0.2], [0.6]

    elif method == 'CIE 1976 UCS':

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_Luv_uv(xy)

        x_limit_min, x_limit_max = [-0.1], [0.7]
        y_limit_min, y_limit_max = [-0.1], [0.7]
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}}'.format(
                method))

    settings = {'colour_cycle_count': len(colourspaces)}
    settings.update(kwargs)

    cycle = colour_cycle(**settings)

    for colourspace in colourspaces:
        R, G, B, _A = next(cycle)

        # RGB colourspaces such as *ACES2065-1* have primaries with
        # chromaticity coordinates set to 0 thus we prevent nan from being
        # yield by zero division in later colour transformations.
        P = np.where(
            colourspace.primaries == 0,
            EPSILON,
            colourspace.primaries,
        )
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
def plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1931=(
            plot_chromaticity_diagram_CIE1931),
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
    >>> plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
    ...     ['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return plot_RGB_colourspaces_in_chromaticity_diagram(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1931, **settings)


@override_style()
def plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1960UCS=(
            plot_chromaticity_diagram_CIE1960UCS),
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
    >>> plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS(
    ...     ['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return plot_RGB_colourspaces_in_chromaticity_diagram(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1960UCS,
        **settings)


@override_style()
def plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(
        colourspaces=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        chromaticity_diagram_callable_CIE1976UCS=(
            plot_chromaticity_diagram_CIE1976UCS),
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
    >>> plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(
    ...     ['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return plot_RGB_colourspaces_in_chromaticity_diagram(
        colourspaces, cmfs, chromaticity_diagram_callable_CIE1976UCS,
        **settings)


@override_style()
def plot_RGB_chromaticities_in_chromaticity_diagram(
        RGB,
        colourspace='sRGB',
        chromaticity_diagram_callable=(
            plot_RGB_colourspaces_in_chromaticity_diagram),
        method='CIE 1931',
        scatter_parameters=None,
        **kwargs):
    """
    Plots given *RGB* colourspace array in the *Chromaticity Diagram* according
    to given method.

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
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.diagrams.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram(
    ...     RGB, 'ITU-R BT.709')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Chromaticities_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_RGB_chromaticities_in_chromaticity_diagram
    """

    RGB = as_float_array(RGB).reshape(-1, 3)

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

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

    colourspace = first_item(filter_RGB_colourspaces(colourspace).values())
    settings['colourspaces'] = (
        ['^{0}$'.format(colourspace.name)] + settings.get('colourspaces', []))

    chromaticity_diagram_callable(**settings)

    use_RGB_colours = scatter_settings['c'].upper() == 'RGB'
    if use_RGB_colours:
        RGB = RGB[RGB[:, 1].argsort()]
        scatter_settings['c'] = np.clip(
            RGB_to_RGB(
                RGB,
                colourspace,
                COLOUR_STYLE_CONSTANTS.colour.colourspace,
                apply_cctf_encoding=True).reshape(-1, 3), 0, 1)

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
def plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
        RGB,
        colourspace='sRGB',
        chromaticity_diagram_callable_CIE1931=(
            plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931),
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
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.diagrams.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
    ...     RGB, 'ITU-R BT.709')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return plot_RGB_chromaticities_in_chromaticity_diagram(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1931,
        scatter_parameters=scatter_parameters,
        **settings)


@override_style()
def plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS(
        RGB,
        colourspace='sRGB',
        chromaticity_diagram_callable_CIE1960UCS=(
            plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS),
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
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.diagrams.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS(
    ...     RGB, 'ITU-R BT.709')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return plot_RGB_chromaticities_in_chromaticity_diagram(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1960UCS,
        scatter_parameters=scatter_parameters,
        **settings)


@override_style()
def plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
        RGB,
        colourspace='sRGB',
        chromaticity_diagram_callable_CIE1976UCS=(
            plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS),
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
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.diagrams.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
    ...     RGB, 'ITU-R BT.709')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return plot_RGB_chromaticities_in_chromaticity_diagram(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1976UCS,
        scatter_parameters=scatter_parameters,
        **settings)


def ellipses_MacAdam1942(method='CIE 1931'):
    """
    Returns *MacAdam (1942) Ellipses (Observer PGN)* coefficients according to
    given method.

    Parameters
    ----------
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        Computation method.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> ellipses_MacAdam1942()[0]  # doctest: +SKIP
    array([  1.60000000e-01,   5.70000000e-02,   5.00000023e-03,
             1.56666660e-02,  -2.77000015e+01])
    """

    method = method.upper()

    if method == 'CIE 1931':

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy

    elif method == 'CIE 1960 UCS':

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_UCS_uv(xy)

    elif method == 'CIE 1976 UCS':

        def xy_to_ij(xy):
            """
            Converts given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_Luv_uv(xy)

    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}}'.format(
                method))

    x, y, _a, _b, _theta, a, b, theta = tsplit(MACADAM_1942_ELLIPSES_DATA)

    ellipses_coefficients = []
    # pylint: disable=C0200
    for i in range(len(theta)):
        xy = point_at_angle_on_ellipse(
            np.linspace(0, 360, 36),
            [x[i], y[i], a[i] / 60, b[i] / 60, theta[i]],
        )
        ij = xy_to_ij(xy)
        ellipses_coefficients.append(
            ellipse_coefficients_canonical_form(ellipse_fitting(ij)))

    return ellipses_coefficients


@override_style()
def plot_ellipses_MacAdam1942_in_chromaticity_diagram(
        chromaticity_diagram_callable=plot_chromaticity_diagram,
        method='CIE 1931',
        chromaticity_diagram_clipping=False,
        ellipse_parameters=None,
        **kwargs):
    """
    Plots *MacAdam (1942) Ellipses (Observer PGN)* in the
    *Chromaticity Diagram* according to given method.

    Parameters
    ----------
    chromaticity_diagram_callable : callable, optional
        Callable responsible for drawing the *Chromaticity Diagram*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.
    chromaticity_diagram_clipping : bool, optional,
        Whether to clip the *Chromaticity Diagram* colours with the ellipses.
    ellipse_parameters : dict or array_like, optional
        Parameters for the :class:`Ellipse` class, ``ellipse_parameters`` can
        be either a single dictionary applied to all the ellipses with same
        settings or a sequence of dictionaries with different settings for each
        ellipse.

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
    >>> plot_ellipses_MacAdam1942_in_chromaticity_diagram()
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/\
Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_ellipses_MacAdam1942_in_chromaticity_diagram
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    settings = dict(kwargs)
    settings.update({'axes': axes, 'standalone': False})

    ellipses_coefficients = ellipses_MacAdam1942(method=method)

    if chromaticity_diagram_clipping:
        diagram_clipping_path_x = []
        diagram_clipping_path_y = []
        for coefficients in ellipses_coefficients:
            coefficients = np.copy(coefficients)

            coefficients[2:4] /= 2

            x, y = tsplit(
                point_at_angle_on_ellipse(
                    np.linspace(0, 360, 36),
                    coefficients,
                ))
            diagram_clipping_path_x.append(x)
            diagram_clipping_path_y.append(y)

        diagram_clipping_path = np.rollaxis(
            np.array([diagram_clipping_path_x, diagram_clipping_path_y]), 0, 3)
        diagram_clipping_path = Path.make_compound_path_from_polys(
            diagram_clipping_path).vertices
        settings.update({'diagram_clipping_path': diagram_clipping_path})

    chromaticity_diagram_callable(**settings)

    ellipse_settings_collection = [{
        'color': COLOUR_STYLE_CONSTANTS.colour.cycle[4],
        'alpha': 0.4,
        'edgecolor': COLOUR_STYLE_CONSTANTS.colour.cycle[1],
        'linewidth': colour_style()['lines.linewidth']
    } for _ellipses_coefficient in ellipses_coefficients]

    if ellipse_parameters is not None:
        if not isinstance(ellipse_parameters, dict):
            assert len(ellipse_parameters) == len(ellipses_coefficients), (
                'Multiple ellipse parameters defined, but they do not match '
                'the ellipses count!')

        for i, ellipse_settings in enumerate(ellipse_settings_collection):
            if isinstance(ellipse_parameters, dict):
                ellipse_settings.update(ellipse_parameters)
            else:
                ellipse_settings.update(ellipse_parameters[i])

    for i, coefficients in enumerate(ellipses_coefficients):
        x_c, y_c, a_a, a_b, theta_e = coefficients
        ellipse = Ellipse((x_c, y_c), a_a, a_b, theta_e,
                          **ellipse_settings_collection[i])
        axes.add_artist(ellipse)

    settings.update({'standalone': True})
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931(
        chromaticity_diagram_callable_CIE1931=(
            plot_chromaticity_diagram_CIE1931),
        chromaticity_diagram_clipping=False,
        ellipse_parameters=None,
        **kwargs):
    """
    Plots *MacAdam (1942) Ellipses (Observer PGN)* in the
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    chromaticity_diagram_clipping : bool, optional,
        Whether to clip the *CIE 1931 Chromaticity Diagram* colours with the
        ellipses.
    ellipse_parameters : dict or array_like, optional
        Parameters for the :class:`Ellipse` class, ``ellipse_parameters`` can
        be either a single dictionary applied to all the ellipses with same
        settings or a sequence of dictionaries with different settings for each
        ellipse.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram`},
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931()
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/\
Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return plot_ellipses_MacAdam1942_in_chromaticity_diagram(
        chromaticity_diagram_callable_CIE1931,
        chromaticity_diagram_clipping=chromaticity_diagram_clipping,
        ellipse_parameters=ellipse_parameters,
        **settings)


@override_style()
def plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS(
        chromaticity_diagram_callable_CIE1960UCS=(
            plot_chromaticity_diagram_CIE1960UCS),
        chromaticity_diagram_clipping=False,
        ellipse_parameters=None,
        **kwargs):
    """
    Plots *MacAdam (1942) Ellipses (Observer PGN)* in the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.
    chromaticity_diagram_clipping : bool, optional,
        Whether to clip the *CIE 1960 UCS Chromaticity Diagram* colours with
        the ellipses.
    ellipse_parameters : dict or array_like, optional
        Parameters for the :class:`Ellipse` class, ``ellipse_parameters`` can
        be either a single dictionary applied to all the ellipses with same
        settings or a sequence of dictionaries with different settings for each
        ellipse.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram`},
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS()
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/\
Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return plot_ellipses_MacAdam1942_in_chromaticity_diagram(
        chromaticity_diagram_callable_CIE1960UCS,
        chromaticity_diagram_clipping=chromaticity_diagram_clipping,
        ellipse_parameters=ellipse_parameters,
        **settings)


@override_style()
def plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS(
        chromaticity_diagram_callable_CIE1976UCS=(
            plot_chromaticity_diagram_CIE1976UCS),
        chromaticity_diagram_clipping=False,
        ellipse_parameters=None,
        **kwargs):
    """
    Plots *MacAdam (1942) Ellipses (Observer PGN)* in the
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    chromaticity_diagram_callable_CIE1976UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.
    chromaticity_diagram_clipping : bool, optional,
        Whether to clip the *CIE 1976 UCS Chromaticity Diagram* colours with
        the ellipses.
    ellipse_parameters : dict or array_like, optional
        Parameters for the :class:`Ellipse` class, ``ellipse_parameters`` can
        be either a single dictionary applied to all the ellipses with same
        settings or a sequence of dictionaries with different settings for each
        ellipse.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram`},
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS()
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/\
Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return plot_ellipses_MacAdam1942_in_chromaticity_diagram(
        chromaticity_diagram_callable_CIE1976UCS,
        chromaticity_diagram_clipping=chromaticity_diagram_clipping,
        ellipse_parameters=ellipse_parameters,
        **settings)


@override_style()
def plot_single_cctf(cctf='ITU-R BT.709', cctf_decoding=False, **kwargs):
    """
    Plots given colourspace colour component transfer function.

    Parameters
    ----------
    cctf : unicode, optional
        Colour component transfer function to plot.
    cctf_decoding : bool
        Plot the decoding colour component transfer function instead.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_single_cctf('ITU-R BT.709')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Single_CCTF.png
        :align: center
        :alt: plot_single_cctf
    """

    settings = {
        'title':
            '{0} - {1} CCTF'.format(
                cctf, 'Decoding' if cctf_decoding else 'Encoding')
    }
    settings.update(kwargs)

    return plot_multi_cctfs([cctf], cctf_decoding, **settings)


@override_style()
def plot_multi_cctfs(cctfs=None, cctf_decoding=False, **kwargs):
    """
    Plots given colour component transfer functions.

    Parameters
    ----------
    cctfs : array_like, optional
        Colour component transfer function to plot.
    cctf_decoding : bool
        Plot the decoding colour component transfer function instead.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_cctfs(['ITU-R BT.709', 'sRGB'])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Multi_CCTFs.png
        :align: center
        :alt: plot_multi_cctfs
    """

    if cctfs is None:
        cctfs = ('ITU-R BT.709', 'sRGB')

    cctfs = filter_passthrough(
        CCTF_DECODINGS if cctf_decoding else CCTF_ENCODINGS, cctfs)

    mode = 'Decoding' if cctf_decoding else 'Encoding'
    title = '{0} - {1} CCTFs'.format(', '.join([cctf for cctf in cctfs]), mode)

    settings = {
        'bounding_box': (0, 1, 0, 1),
        'legend': True,
        'title': title,
        'x_label': 'Signal Value' if cctf_decoding else 'Tristimulus Value',
        'y_label': 'Tristimulus Value' if cctf_decoding else 'Signal Value',
    }
    settings.update(kwargs)

    with domain_range_scale(1):
        return plot_multi_functions(cctfs, **settings)


@override_style()
def plot_constant_hue_loci(data, model, scatter_parameters=None, **kwargs):
    """
    Plots given constant hue loci colour matches data such as that from
    :cite:`Hung1995` or :cite:`Ebner1998` that are easily loaded with
    `Colour - Datasets <https://github.com/colour-science/colour-datasets>`_.

    Parameters
    ----------
    data : array_like
        Constant hue loci colour matches data expected to be an *array_like* as
        follows::

            [
                ('name', XYZ_r, XYZ_cr, (XYZ_ct, XYZ_ct, XYZ_ct, ...), \
{metadata}),
                ('name', XYZ_r, XYZ_cr, (XYZ_ct, XYZ_ct, XYZ_ct, ...), \
{metadata}),
                ('name', XYZ_r, XYZ_cr, (XYZ_ct, XYZ_ct, XYZ_ct, ...), \
{metadata}),
                ...
            ]

        where ``name`` is the hue angle or name, ``XYZ_r`` the *CIE XYZ*
        tristimulus values of the reference illuminant, ``XYZ_cr`` the
        *CIE XYZ* tristimulus values of the reference colour under the
        reference illuminant, ``XYZ_ct`` the *CIE XYZ* tristimulus values of
        the colour matches under the reference illuminant and ``metadata`` the
        dataset metadata.
    model : unicode, optional
        **{'CIE XYZ', 'CIE xyY', 'CIE xy', 'CIE Lab', 'CIE LCHab', 'CIE Luv',
        'CIE Luv uv', 'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW',
        'DIN 99', 'Hunter Lab', 'Hunter Rdab', 'IPT', 'JzAzBz', 'OSA UCS',
        'hdr-CIELAB', 'hdr-IPT'}**,
        Colourspace model.
    scatter_parameters : dict, optional
        Parameters for the :func:`plt.scatter` definition, if ``c`` is set to
        *RGB*, the scatter will use given ``RGB`` colours.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    References
    ----------
    :cite:`Ebner1998`, :cite:`Hung1995`, :cite:`Mansencal2019`

    Examples
    --------
    >>> data = np.array([
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.40920000, 0.28120000, 0.30600000]),
    ...         np.array([
    ...             [0.02495100, 0.01908600, 0.02032900],
    ...             [0.10944300, 0.06235900, 0.06788100],
    ...             [0.27186500, 0.18418700, 0.19565300],
    ...             [0.48898900, 0.40749400, 0.44854600],
    ...         ]),
    ...         None,
    ...     ],
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.30760000, 0.48280000, 0.42770000]),
    ...         np.array([
    ...             [0.02108000, 0.02989100, 0.02790400],
    ...             [0.06194700, 0.11251000, 0.09334400],
    ...             [0.15255800, 0.28123300, 0.23234900],
    ...             [0.34157700, 0.56681300, 0.47035300],
    ...         ]),
    ...         None,
    ...     ],
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.39530000, 0.28120000, 0.18450000]),
    ...         np.array([
    ...             [0.02436400, 0.01908600, 0.01468800],
    ...             [0.10331200, 0.06235900, 0.02854600],
    ...             [0.26311900, 0.18418700, 0.12109700],
    ...             [0.43158700, 0.40749400, 0.39008600],
    ...         ]),
    ...         None,
    ...     ],
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.20510000, 0.18420000, 0.57130000]),
    ...         np.array([
    ...             [0.03039800, 0.02989100, 0.06123300],
    ...             [0.08870000, 0.08498400, 0.21843500],
    ...             [0.18405800, 0.18418700, 0.40111400],
    ...             [0.32550100, 0.34047200, 0.50296900],
    ...             [0.53826100, 0.56681300, 0.80010400],
    ...         ]),
    ...         None,
    ...     ],
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.35770000, 0.28120000, 0.11250000]),
    ...         np.array([
    ...             [0.03678100, 0.02989100, 0.01481100],
    ...             [0.17127700, 0.11251000, 0.01229900],
    ...             [0.30080900, 0.28123300, 0.21229800],
    ...             [0.52976000, 0.40749400, 0.11720000],
    ...         ]),
    ...         None,
    ...     ],
    ... ])
    >>> plot_constant_hue_loci(data, 'IPT')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Constant_Hue_Loci.png
        :align: center
        :alt: plot_constant_hue_loci
    """

    # TODO: Filter appropriate colour models.

    data = data.values() if isinstance(data, Mapping) else data

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    scatter_settings = {
        's': 40,
        'c': 'RGB',
        'marker': 'o',
        'alpha': 0.85,
    }
    if scatter_parameters is not None:
        scatter_settings.update(scatter_parameters)

    use_RGB_colours = scatter_settings['c'].upper() == 'RGB'

    colourspace = COLOUR_STYLE_CONSTANTS.colour.colourspace
    for hue_data in data:
        _name, XYZ_r, XYZ_cr, XYZ_ct, _metadata = hue_data

        xy_r = XYZ_to_xy(XYZ_r)
        ijk_ct = common_colourspace_model_axis_reorder(
            convert(XYZ_ct, 'CIE XYZ', model, illuminant=xy_r), model)
        ijk_cr = common_colourspace_model_axis_reorder(
            convert(XYZ_cr, 'CIE XYZ', model, illuminant=xy_r), model)

        def _linear_equation(x, a, b):
            """
            Defines the canonical linear equation for a line.
            """

            return a * x + b

        popt, _pcov = scipy.optimize.curve_fit(_linear_equation,
                                               ijk_ct[..., 0], ijk_ct[..., 1])

        axes.plot(
            ijk_ct[..., 0],
            _linear_equation(ijk_ct[..., 0], *popt),
            c=COLOUR_STYLE_CONSTANTS.colour.average)

        if use_RGB_colours:

            def _XYZ_to_RGB(XYZ):
                """
                Converts given *CIE XYZ* tristimulus values to
                ``colour.plotting`` *RGB* colourspace.
                """

                return XYZ_to_RGB(
                    XYZ,
                    xy_r,
                    colourspace.whitepoint,
                    colourspace.XYZ_to_RGB_matrix,
                    cctf_encoding=colourspace.cctf_encoding)

            RGB_ct = _XYZ_to_RGB(XYZ_ct)
            RGB_cr = _XYZ_to_RGB(XYZ_cr)

            scatter_settings['c'] = np.clip(RGB_ct, 0, 1)

        axes.scatter(
            ijk_ct[..., 0], ijk_ct[..., 1], zorder=10, **scatter_settings)

        axes.plot(
            ijk_cr[..., 0],
            ijk_cr[..., 1],
            's',
            zorder=10,
            c=np.clip(np.ravel(RGB_cr), 0, 1),
            markersize=COLOUR_STYLE_CONSTANTS.geometry.short * 8)

    labels = np.array(COLOURSPACE_MODELS_AXIS_LABELS[model])[as_int_array(
        common_colourspace_model_axis_reorder([0, 1, 2], model))]

    settings = {
        'axes': axes,
        'title': 'Constant Hue Loci - '
                 '{0} (Domain-Range Scale: 1)'.format(model),
        'x_label': labels[0],
        'y_label': labels[1],
    }
    settings.update(kwargs)

    return render(**settings)
