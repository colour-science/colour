#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Temperature & Correlated Colour Temperature Plotting
===========================================================

Defines the colour temperature and correlated colour temperature plotting
objects:

-   :func:`planckian_locus_CIE_1931_chromaticity_diagram_plot`
-   :func:`planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot`
"""

from __future__ import division

import numpy as np
import pylab

from colour.colorimetry import (
    CMFS,
    ILLUMINANTS)
from colour.models import (
    UCS_uv_to_xy,
    XYZ_to_UCS,
    UCS_to_uv,
    xy_to_XYZ)
from colour.temperature import CCT_to_uv
from colour.plotting import (
    CIE_1931_chromaticity_diagram_plot,
    CIE_1960_UCS_chromaticity_diagram_plot,
    boundaries,
    decorate,
    display)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['planckian_locus_CIE_1931_chromaticity_diagram_plot',
           'planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot']


def planckian_locus_CIE_1931_chromaticity_diagram_plot(
        illuminants=None,
        **kwargs):
    """
    Plots the planckian locus and given illuminants in
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`boundaries`, :func:`canvas`, :func:`decorate`,
        :func:`display`},
        Please refer to the documentation of the previously listed definitions.
    show_diagram_colours : bool, optional
        {:func:`CIE_1931_chromaticity_diagram_plot`},
        Whether to display the chromaticity diagram background colours.

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
    >>> ils = ['A', 'B', 'C']
    >>> planckian_locus_CIE_1931_chromaticity_diagram_plot(
    ...     ils)  # doctest: +SKIP
    """

    if illuminants is None:
        illuminants = ('A', 'B', 'C')

    cmfs = CMFS['CIE 1931 2 Degree Standard Observer']

    settings = {
        'title': ('{0} Illuminants - Planckian Locus\n'
                  'CIE 1931 Chromaticity Diagram - '
                  'CIE 1931 2 Degree Standard Observer').format(
            ', '.join(illuminants))
        if illuminants else
        ('Planckian Locus\nCIE 1931 Chromaticity Diagram - '
         'CIE 1931 2 Degree Standard Observer'),
        'standalone': False}
    settings.update(kwargs)

    CIE_1931_chromaticity_diagram_plot(**settings)

    start, end = 1667, 100000
    xy = np.array([UCS_uv_to_xy(CCT_to_uv(x, 'Robertson 1968', D_uv=0))
                   for x in np.arange(start, end + 250, 250)])

    pylab.plot(xy[..., 0], xy[..., 1], color='black', linewidth=2)

    for i in (1667, 2000, 2500, 3000, 4000, 6000, 10000):
        x0, y0 = UCS_uv_to_xy(CCT_to_uv(i, 'Robertson 1968', D_uv=-0.025))
        x1, y1 = UCS_uv_to_xy(CCT_to_uv(i, 'Robertson 1968', D_uv=0.025))
        pylab.plot((x0, x1), (y0, y1), color='black', linewidth=2)
        pylab.annotate('{0}K'.format(i),
                       xy=(x0, y0),
                       xytext=(0, -10),
                       color='black',
                       textcoords='offset points',
                       size='x-small')

    for illuminant in illuminants:
        xy = ILLUMINANTS.get(cmfs.name).get(illuminant)
        if xy is None:
            raise KeyError(
                ('Illuminant "{0}" not found in factory illuminants: '
                 '"{1}".').format(illuminant,
                                  sorted(ILLUMINANTS[cmfs.name].keys())))

        pylab.plot(xy[0], xy[1], 'o', color='white', linewidth=2)

        pylab.annotate(illuminant,
                       xy=(xy[0], xy[1]),
                       xytext=(-50, 30),
                       color='black',
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='->',
                                       connectionstyle='arc3, rad=-0.2'))

    settings.update({
        'x_tighten': True,
        'y_tighten': True,
        'limits': (-0.1, 0.9, -0.1, 0.9),
        'standalone': True})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot(
        illuminants=None,
        **kwargs):
    """
    Plots the planckian locus and given illuminants in
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`boundaries`, :func:`canvas`, :func:`decorate`,
        :func:`display`},
        Please refer to the documentation of the previously listed definitions.
    show_diagram_colours : bool, optional
        {:func:`CIE_1960_UCS_chromaticity_diagram_plot`},
        Whether to display the chromaticity diagram background colours.

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
    >>> ils = ['A', 'C', 'E']
    >>> planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot(
    ...     ils)  # doctest: +SKIP
    """

    if illuminants is None:
        illuminants = ('A', 'C', 'E')

    cmfs = CMFS['CIE 1931 2 Degree Standard Observer']

    settings = {
        'title': ('{0} Illuminants - Planckian Locus\n'
                  'CIE 1960 UCS Chromaticity Diagram - '
                  'CIE 1931 2 Degree Standard Observer').format(
            ', '.join(illuminants))
        if illuminants else
        ('Planckian Locus\nCIE 1960 UCS Chromaticity Diagram - '
         'CIE 1931 2 Degree Standard Observer'),
        'standalone': False}
    settings.update(kwargs)

    CIE_1960_UCS_chromaticity_diagram_plot(**settings)

    start, end = 1667, 100000
    uv = np.array([CCT_to_uv(x, 'Robertson 1968', D_uv=0)
                   for x in np.arange(start, end + 250, 250)])

    pylab.plot(uv[..., 0], uv[..., 1], color='black', linewidth=2)

    for i in (1667, 2000, 2500, 3000, 4000, 6000, 10000):
        u0, v0 = CCT_to_uv(i, 'Robertson 1968', D_uv=-0.05)
        u1, v1 = CCT_to_uv(i, 'Robertson 1968', D_uv=0.05)
        pylab.plot((u0, u1), (v0, v1), color='black', linewidth=2)
        pylab.annotate('{0}K'.format(i),
                       xy=(u0, v0),
                       xytext=(0, -10),
                       color='black',
                       textcoords='offset points',
                       size='x-small')

    for illuminant in illuminants:
        xy = ILLUMINANTS.get(cmfs.name).get(illuminant)
        if xy is None:
            raise KeyError(
                ('Illuminant "{0}" not found in factory illuminants: '
                 '"{1}".').format(illuminant,
                                  sorted(ILLUMINANTS[cmfs.name].keys())))

        uv = UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(xy)))

        pylab.plot(uv[0], uv[1], 'o', color='white', linewidth=2)

        pylab.annotate(illuminant,
                       xy=(uv[0], uv[1]),
                       xytext=(-50, 30),
                       color='black',
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='->',
                                       connectionstyle='arc3, rad=-0.2'))

    settings.update({
        'x_tighten': True,
        'y_tighten': True,
        'limits': (-0.1, 0.7, -0.2, 0.6),
        'standalone': True})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)
