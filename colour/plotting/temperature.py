# -*- coding: utf-8 -*-
"""
Colour Temperature & Correlated Colour Temperature Plotting
===========================================================

Defines the colour temperature and correlated colour temperature plotting
objects:

-   :func:`colour.plotting.planckian_locus_chromaticity_diagram_plot_CIE1931`
-   :func:`colour.plotting.\
planckian_locus_chromaticity_diagram_plot_CIE1960UCS`
"""

from __future__ import division

import numpy as np
import pylab

from colour.colorimetry import CMFS, ILLUMINANTS
from colour.models import (UCS_uv_to_xy, XYZ_to_UCS, UCS_to_uv, xy_to_XYZ)
from colour.temperature import CCT_to_uv
from colour.plotting import (DEFAULT_FIGURE_WIDTH, canvas,
                             chromaticity_diagram_plot_CIE1931,
                             chromaticity_diagram_plot_CIE1960UCS, render)
from colour.plotting.diagrams import chromaticity_diagram_plot

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'planckian_locus_plot', 'planckian_locus_chromaticity_diagram_plot',
    'planckian_locus_chromaticity_diagram_plot_CIE1931',
    'planckian_locus_chromaticity_diagram_plot_CIE1960UCS'
]


def planckian_locus_plot(planckian_locus_colours='black',
                         method='CIE 1931',
                         **kwargs):
    """
    Plots the *Planckian Locus* accordingly to given method.

    Parameters
    ----------
    planckian_locus_colours : array_like or unicode, optional
        *Planckian Locus* colours.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

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
    >>> planckian_locus_plot()  # doctest: +SKIP

    .. image:: ../_static/Plotting_Planckian_Locus_Plot.png
        :align: center
        :alt: planckian_locus_plot
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

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

        D_uv = 0.05
    else:
        raise ValueError('Invalid method: "{0}", must be one of '
                         '{\'CIE 1931\', \'CIE 1960 UCS\'}'.format(method))

    start, end = 1667, 100000
    ij = np.array([
        uv_to_ij(CCT_to_uv(x, 'Robertson 1968', D_uv=0))
        for x in np.arange(start, end + 250, 250)
    ])

    pylab.plot(ij[..., 0], ij[..., 1], color=planckian_locus_colours)

    for i in (1667, 2000, 2500, 3000, 4000, 6000, 10000):
        i0, j0 = uv_to_ij(CCT_to_uv(i, 'Robertson 1968', D_uv=-D_uv))
        i1, j1 = uv_to_ij(CCT_to_uv(i, 'Robertson 1968', D_uv=D_uv))
        pylab.plot((i0, i1), (j0, j1), color=planckian_locus_colours)
        pylab.annotate(
            '{0}K'.format(i),
            xy=(i0, j0),
            xytext=(0, -10),
            textcoords='offset points',
            size='x-small')

    return render(**settings)


def planckian_locus_chromaticity_diagram_plot(
        illuminants=None,
        chromaticity_diagram_callable=chromaticity_diagram_plot,
        method='CIE 1931',
        **kwargs):
    """
    Plots the *Planckian Locus* and given illuminants in the
    *Chromaticity Diagram* accordingly to given method.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.
    chromaticity_diagram_callable : callable, optional
        Callable responsible for drawing the *Chromaticity Diagram*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.diagrams.chromaticity_diagram_plot`,
        :func:`colour.plotting.temperature.planckian_locus_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Raises
    ------
    KeyError
        If one of the given illuminant is not found in the factory illuminants.

    Examples
    --------
    >>> planckian_locus_chromaticity_diagram_plot(['A', 'B', 'C'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
Planckian_Locus_Chromaticity_Diagram_Plot.png
        :align: center
        :alt: planckian_locus_chromaticity_diagram_plot
    """

    if illuminants is None:
        illuminants = ('A', 'B', 'C')

    cmfs = CMFS['CIE 1931 2 Degree Standard Observer']

    method = method.upper()
    settings = {'method': method, 'standalone': False}
    settings.update(kwargs)

    chromaticity_diagram_callable(**settings)

    planckian_locus_plot(**settings)

    if method == 'CIE 1931':

        def xy_to_ij(xy):
            """
            Converts given *xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return xy

        limits = (-0.1, 0.9, -0.1, 0.9)
    elif method == 'CIE 1960 UCS':

        def xy_to_ij(xy):
            """
            Converts given *xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(xy)))

        limits = (-0.1, 0.7, -0.2, 0.6)
    else:
        raise ValueError('Invalid method: "{0}", must be one of '
                         '{\'CIE 1931\', \'CIE 1960 UCS\'}'.format(method))

    for illuminant in illuminants:
        xy = ILLUMINANTS.get(cmfs.name).get(illuminant)
        if xy is None:
            raise KeyError(
                ('Illuminant "{0}" not found in factory illuminants: '
                 '"{1}".').format(illuminant,
                                  sorted(ILLUMINANTS[cmfs.name].keys())))
        ij = xy_to_ij(xy)

        pylab.plot(ij[0], ij[1], 'o', color='white')

        pylab.annotate(
            illuminant,
            xy=(ij[0], ij[1]),
            xytext=(-50, 30),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-0.2'))

    settings.update({
        'title': ('{0} Illuminants - Planckian Locus\n'
                  'CIE 1931 Chromaticity Diagram - '
                  'CIE 1931 2 Degree Standard Observer'
                  ).format(', '.join(illuminants)) if illuminants else
                 ('Planckian Locus\nCIE 1931 Chromaticity Diagram - '
                  'CIE 1931 2 Degree Standard Observer'),
        'x_tighten':
            True,
        'y_tighten':
            True,
        'limits':
            limits,
        'standalone':
            True
    })
    settings.update(kwargs)

    return render(**settings)


def planckian_locus_chromaticity_diagram_plot_CIE1931(
        illuminants=None,
        chromaticity_diagram_callable_CIE1931=(
            chromaticity_diagram_plot_CIE1931),
        **kwargs):
    """
    Plots the *Planckian Locus* and given illuminants in
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.temperature.\
planckian_locus_chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Raises
    ------
    KeyError
        If one of the given illuminant is not found in the factory illuminants.

    Examples
    --------
    >>> planckian_locus_chromaticity_diagram_plot_CIE1931(['A', 'B', 'C'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
Planckian_Locus_Chromaticity_Diagram_Plot_CIE1931.png
        :align: center
        :alt: planckian_locus_chromaticity_diagram_plot_CIE1931
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return planckian_locus_chromaticity_diagram_plot(
        illuminants, chromaticity_diagram_callable_CIE1931, **settings)


def planckian_locus_chromaticity_diagram_plot_CIE1960UCS(
        illuminants=None,
        chromaticity_diagram_callable_CIE1960UCS=(
            chromaticity_diagram_plot_CIE1960UCS),
        **kwargs):
    """
    Plots the *Planckian Locus* and given illuminants in
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.temperature.\
planckian_locus_chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Raises
    ------
    KeyError
        If one of the given illuminant is not found in the factory illuminants.

    Examples
    --------
    >>> planckian_locus_chromaticity_diagram_plot_CIE1960UCS(['A', 'C', 'E'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
Planckian_Locus_Chromaticity_Diagram_Plot_CIE1960UCS.png
        :align: center
        :alt: planckian_locus_chromaticity_diagram_plot_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return planckian_locus_chromaticity_diagram_plot(
        illuminants, chromaticity_diagram_callable_CIE1960UCS, **settings)
