# -*- coding: utf-8 -*-
"""
Optical Phenomenon Plotting
===========================

Defines the optical phenomena plotting objects:

-   :func:`colour.plotting.plot_single_rayleigh_scattering_spd`
-   :func:`colour.plotting.the_blue_sky_plot`
"""

from __future__ import division

import matplotlib.pyplot as plt

from colour.colorimetry import spectral_to_XYZ
from colour.phenomena import rayleigh_scattering_spd
from colour.phenomena.rayleigh import (
    AVERAGE_PRESSURE_MEAN_SEA_LEVEL, DEFAULT_ALTITUDE, DEFAULT_LATITUDE,
    STANDARD_AIR_TEMPERATURE, STANDARD_CO2_CONCENTRATION)
from colour.plotting import (ASTM_G_173_ETR, COLOUR_STYLE_CONSTANTS,
                             ColourSwatch, XYZ_to_plotting_colourspace,
                             filter_cmfs, override_style, render,
                             plot_single_colour_swatch, plot_single_spd)
from colour.utilities import first_item, normalise_maximum

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['plot_single_rayleigh_scattering_spd', 'the_blue_sky_plot']


@override_style()
def plot_single_rayleigh_scattering_spd(
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
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    out_of_gamut_clipping : bool, optional
        {:func:`colour.plotting.plot_single_spd`},
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_single_rayleigh_scattering_spd()  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Single_Rayleigh_Scattering_SPD.png
        :align: center
        :alt: plot_single_rayleigh_scattering_spd
    """

    title = 'Rayleigh Scattering'

    cmfs = first_item(filter_cmfs(cmfs).values())

    settings = {'title': title, 'y_label': 'Optical Depth'}
    settings.update(kwargs)

    spd = rayleigh_scattering_spd(cmfs.shape, CO2_concentration, temperature,
                                  pressure, latitude, altitude)

    return plot_single_spd(spd, **settings)


@override_style()
def the_blue_sky_plot(cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
    """
    Plots the blue sky.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions.

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
    >>> the_blue_sky_plot()  # doctest: +SKIP

    .. image:: ../_static/Plotting_The_Blue_Sky_Plot.png
        :align: center
        :alt: the_blue_sky_plot
    """

    figure = plt.figure()

    figure.subplots_adjust(hspace=COLOUR_STYLE_CONSTANTS.geometry.short / 2)

    cmfs = first_item(filter_cmfs(cmfs).values())

    ASTM_G_173_spd = ASTM_G_173_ETR.copy()
    rayleigh_spd = rayleigh_scattering_spd()
    ASTM_G_173_spd.align(rayleigh_spd.shape)

    spd = rayleigh_spd * ASTM_G_173_spd

    axes = figure.add_subplot(211)

    settings = {
        'axes': axes,
        'title': 'The Blue Sky - Synthetic Spectral Power Distribution',
        'y_label': u'W / m-2 / nm-1',
    }
    settings.update(kwargs)
    settings['standalone'] = False

    plot_single_spd(spd, cmfs, **settings)

    axes = figure.add_subplot(212)

    x_label = ('The sky is blue because molecules in the atmosphere '
               'scatter shorter wavelengths more than longer ones.\n'
               'The synthetic spectral power distribution is computed as '
               'follows: '
               '(ASTM G-173 ETR * Standard Air Rayleigh Scattering).')

    settings = {
        'axes': axes,
        'aspect': None,
        'title': 'The Blue Sky - Colour',
        'x_label': x_label,
        'y_label': '',
    }
    settings.update(kwargs)
    settings['standalone'] = False

    blue_sky_color = XYZ_to_plotting_colourspace(spectral_to_XYZ(spd))

    figure, axes = plot_single_colour_swatch(
        ColourSwatch('', normalise_maximum(blue_sky_color)), **settings)

    # Removing "x" and "y" ticks.
    axes.set_xticks([])
    axes.set_yticks([])

    settings = {'axes': axes, 'standalone': True}
    settings.update(kwargs)

    return render(**settings)
