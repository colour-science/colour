#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optical Phenomenon Plotting
===========================

Defines the optical phenomenons plotting objects:

-   :func:`single_rayleigh_scattering_spd_plot`
-   :func:`the_blue_sky_plot`
"""

from __future__ import division

import matplotlib.pyplot

from colour.colorimetry import spectral_to_XYZ
from colour.models import XYZ_to_sRGB
from colour.phenomenons import rayleigh_scattering_spd
from colour.phenomenons.rayleigh import (
    AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
    DEFAULT_ALTITUDE,
    DEFAULT_LATITUDE,
    STANDARD_AIR_TEMPERATURE,
    STANDARD_CO2_CONCENTRATION)
from colour.plotting import (
    ASTM_G_173_ETR,
    ColourParameter,
    boundaries,
    canvas,
    decorate,
    display,
    get_cmfs,
    single_colour_plot,
    single_spd_plot)
from colour.utilities import normalise_maximum

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['single_rayleigh_scattering_spd_plot', 'the_blue_sky_plot']


def single_rayleigh_scattering_spd_plot(
        CO2_concentration=STANDARD_CO2_CONCENTRATION,
        temperature=STANDARD_AIR_TEMPERATURE,
        pressure=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
        latitude=DEFAULT_LATITUDE,
        altitude=DEFAULT_ALTITUDE,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots a single *Rayleigh* scattering spectral power distribution.

    Parameters
    ----------
    CO2_concentration : numeric, optional
        :math:`CO_2` concentration in parts per million (ppm).
    temperature : numeric, optional
        Air temperature :math:`T[K]` in kelvin degrees.
    pressure : numeric
        Surface pressure :math:`P` of the measurement site.
    latitude : numeric, optional
        Latitude of the site in degrees.
    altitude : numeric, optional
        Altitude of the site in meters.
    cmfs : unicode, optional
        Standard observer colour matching functions.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`boundaries`, :func:`canvas`, :func:`decorate`,
        :func:`display`},
        Please refer to the documentation of the previously listed definitions.
    out_of_gamut_clipping : bool, optional
        {:func:`single_spd_plot`},
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother. [1]_

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> single_rayleigh_scattering_spd_plot()  # doctest: +SKIP
    """

    title = 'Rayleigh Scattering'

    cmfs = get_cmfs(cmfs)

    settings = {
        'title': title,
        'y_label': 'Optical Depth'}
    settings.update(kwargs)

    spd = rayleigh_scattering_spd(cmfs.shape,
                                  CO2_concentration,
                                  temperature,
                                  pressure,
                                  latitude,
                                  altitude)

    return single_spd_plot(spd, **settings)


def the_blue_sky_plot(
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots the blue sky.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`boundaries`, :func:`canvas`, :func:`decorate`,
        :func:`display`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> the_blue_sky_plot()  # doctest: +SKIP
    """

    canvas(**kwargs)

    cmfs, name = get_cmfs(cmfs), cmfs

    ASTM_G_173_spd = ASTM_G_173_ETR.clone()
    rayleigh_spd = rayleigh_scattering_spd()
    ASTM_G_173_spd.align(rayleigh_spd.shape)

    spd = rayleigh_spd * ASTM_G_173_spd

    matplotlib.pyplot.subplots_adjust(hspace=0.4)

    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.subplot(211)

    settings = {
        'title': 'The Blue Sky - Synthetic Spectral Power Distribution',
        'y_label': u'W / m-2 / nm-1',
        'standalone': False}
    settings.update(kwargs)

    single_spd_plot(spd, name, **settings)

    matplotlib.pyplot.subplot(212)

    settings = {
        'title': 'The Blue Sky - Colour',
        'x_label': ('The sky is blue because molecules in the atmosphere '
                    'scatter shorter wavelengths more than longer ones.\n'
                    'The synthetic spectral power distribution is computed as '
                    'follows: '
                    '(ASTM G-173 ETR * Standard Air Rayleigh Scattering).'),
        'y_label': '',
        'aspect': None,
        'standalone': False}

    blue_sky_color = XYZ_to_sRGB(spectral_to_XYZ(spd))
    single_colour_plot(ColourParameter('', normalise_maximum(blue_sky_color)),
                       **settings)

    settings = {'standalone': True}
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)
