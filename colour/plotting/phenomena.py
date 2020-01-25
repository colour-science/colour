# -*- coding: utf-8 -*-
"""
Optical Phenomenon Plotting
===========================

Defines the optical phenomena plotting objects:

-   :func:`colour.plotting.plot_single_sd_rayleigh_scattering`
-   :func:`colour.plotting.plot_the_blue_sky`
"""

from __future__ import division

import matplotlib.pyplot as plt

from colour.colorimetry import sd_to_XYZ
from colour.phenomena import sd_rayleigh_scattering
from colour.phenomena.rayleigh import (
    AVERAGE_PRESSURE_MEAN_SEA_LEVEL, DEFAULT_ALTITUDE, DEFAULT_LATITUDE,
    STANDARD_AIR_TEMPERATURE, STANDARD_CO2_CONCENTRATION)
from colour.plotting import (ASTMG173_ETR, COLOUR_STYLE_CONSTANTS,
                             ColourSwatch, XYZ_to_plotting_colourspace,
                             filter_cmfs, override_style, render,
                             plot_single_colour_swatch, plot_single_sd)
from colour.utilities import first_item, normalise_maximum

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['plot_single_sd_rayleigh_scattering', 'plot_the_blue_sky']


@override_style()
def plot_single_sd_rayleigh_scattering(
        CO2_concentration=STANDARD_CO2_CONCENTRATION,
        temperature=STANDARD_AIR_TEMPERATURE,
        pressure=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
        latitude=DEFAULT_LATITUDE,
        altitude=DEFAULT_ALTITUDE,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots a single *Rayleigh* scattering spectral distribution.

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
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_single_sd`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    out_of_gamut_clipping : bool, optional
        {:func:`colour.plotting.plot_single_sd`},
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_single_sd_rayleigh_scattering()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_Single_SD_Rayleigh_Scattering.png
        :align: center
        :alt: plot_single_sd_rayleigh_scattering
    """

    title = 'Rayleigh Scattering'

    cmfs = first_item(filter_cmfs(cmfs).values())

    settings = {'title': title, 'y_label': 'Optical Depth'}
    settings.update(kwargs)

    sd = sd_rayleigh_scattering(cmfs.shape, CO2_concentration, temperature,
                                pressure, latitude, altitude)

    return plot_single_sd(sd, **settings)


@override_style()
def plot_the_blue_sky(cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
    """
    Plots the blue sky.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_single_sd`,
        :func:`colour.plotting.plot_multi_colour_swatches`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_the_blue_sky()  # doctest: +ELLIPSIS
    (<Figure size ... with 2 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_The_Blue_Sky.png
        :align: center
        :alt: plot_the_blue_sky
    """

    figure = plt.figure()

    figure.subplots_adjust(hspace=COLOUR_STYLE_CONSTANTS.geometry.short / 2)

    cmfs = first_item(filter_cmfs(cmfs).values())

    ASTMG173_sd = ASTMG173_ETR.copy()
    rayleigh_sd = sd_rayleigh_scattering()
    ASTMG173_sd.align(rayleigh_sd.shape)

    sd = rayleigh_sd * ASTMG173_sd

    axes = figure.add_subplot(211)

    settings = {
        'axes': axes,
        'title': 'The Blue Sky - Synthetic Spectral Distribution',
        'y_label': u'W / m-2 / nm-1',
    }
    settings.update(kwargs)
    settings['standalone'] = False

    plot_single_sd(sd, cmfs, **settings)

    axes = figure.add_subplot(212)

    x_label = ('The sky is blue because molecules in the atmosphere '
               'scatter shorter wavelengths more than longer ones.\n'
               'The synthetic spectral distribution is computed as '
               'follows: '
               '(ASTM G-173 ETR * Standard Air Rayleigh Scattering).')

    settings = {
        'axes': axes,
        'aspect': None,
        'title': 'The Blue Sky - Colour',
        'x_label': x_label,
        'y_label': '',
        'x_ticker': False,
        'y_ticker': False,
    }
    settings.update(kwargs)
    settings['standalone'] = False

    blue_sky_color = XYZ_to_plotting_colourspace(sd_to_XYZ(sd))

    figure, axes = plot_single_colour_swatch(
        ColourSwatch('', normalise_maximum(blue_sky_color)), **settings)

    settings = {'axes': axes, 'standalone': True}
    settings.update(kwargs)

    return render(**settings)
