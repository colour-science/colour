#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colorimetry Plotting
====================

Defines the colorimetry plotting objects:

-   :func:`single_spd_plot`
-   :func:`multi_spd_plot`
-   :func:`single_cmfs_plot`
-   :func:`multi_cmfs_plot`
-   :func:`single_illuminant_relative_spd_plot`
-   :func:`multi_illuminants_relative_spd_plot`
-   :func:`visible_spectrum_plot`
-   :func:`single_lightness_function_plot`
-   :func:`multi_lightness_function_plot`
-   :func:`blackbody_spectral_radiance_plot`
-   :func:`blackbody_colours_plot`
"""

from __future__ import division

import matplotlib.pyplot
import numpy as np
import pylab

from colour.colorimetry import (
    CMFS,
    DEFAULT_SPECTRAL_SHAPE,
    ILLUMINANTS_RELATIVE_SPDS,
    LIGHTNESS_METHODS,
    SpectralShape,
    blackbody_spd,
    spectral_to_XYZ,
    wavelength_to_XYZ)
from colour.models import XYZ_to_sRGB
from colour.plotting import (
    DEFAULT_FIGURE_WIDTH,
    boundaries,
    canvas,
    colour_parameter,
    colour_parameters_plot,
    decorate,
    display,
    single_colour_plot)
from colour.utilities import normalise

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['get_cmfs',
           'get_illuminant',
           'single_spd_plot',
           'multi_spd_plot',
           'single_cmfs_plot',
           'multi_cmfs_plot',
           'single_illuminant_relative_spd_plot',
           'multi_illuminants_relative_spd_plot',
           'visible_spectrum_plot',
           'single_lightness_function_plot',
           'multi_lightness_function_plot',
           'blackbody_spectral_radiance_plot',
           'blackbody_colours_plot']


def get_cmfs(cmfs):
    """
    Returns the colour matching functions with given name.

    Parameters
    ----------
    cmfs : unicode
        Colour matching functions name.

    Returns
    -------
    RGB_ColourMatchingFunctions or XYZ_ColourMatchingFunctions
        Colour matching functions.

    Raises
    ------
    KeyError
        If the given colour matching functions is not found in the factory
        colour matching functions.
    """

    cmfs, name = CMFS.get(cmfs), cmfs
    if cmfs is None:
        raise KeyError(
            ('"{0}" not found in factory colour matching functions: '
             '"{1}".').format(name, ', '.join(sorted(CMFS.keys()))))
    return cmfs


def get_illuminant(illuminant):
    """
    Returns the illuminant with given name.

    Parameters
    ----------
    illuminant : unicode
        Illuminant name.

    Returns
    -------
    SpectralPowerDistribution
        Illuminant.

    Raises
    ------
    KeyError
        If the given illuminant is not found in the factory illuminants.
    """

    illuminant, name = ILLUMINANTS_RELATIVE_SPDS.get(illuminant), illuminant
    if illuminant is None:
        raise KeyError(
            '"{0}" not found in factory illuminants: "{1}".'.format(
                name, ', '.join(sorted(ILLUMINANTS_RELATIVE_SPDS.keys()))))

    return illuminant


def single_spd_plot(spd, cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
    """
    Plots given spectral power distribution.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution to plot.
    cmfs : unicode
        Standard observer colour matching functions used for spectrum creation.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> from colour import SpectralPowerDistribution
    >>> data = {400: 0.0641, 420: 0.0645, 440: 0.0562}
    >>> spd = SpectralPowerDistribution('Custom', data)
    >>> single_spd_plot(spd)  # doctest: +SKIP
    True
    """

    cmfs = get_cmfs(cmfs)

    shape = cmfs.shape
    spd = spd.clone().interpolate(shape, 'Linear')
    wavelengths = spd.wavelengths
    values = spd.values

    colours = XYZ_to_sRGB(wavelength_to_XYZ(wavelengths, cmfs))
    y1 = values

    colours = normalise(colours)

    settings = {
        'title': '{0} - {1}'.format(spd.title, cmfs.title),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Power Distribution',
        'x_tighten': True,
        'x_ticker': True,
        'y_ticker': True}

    settings.update(kwargs)
    return colour_parameters_plot(
        [colour_parameter(x=x[0], y1=x[1], RGB=x[2])
         for x in tuple(zip(wavelengths, y1, colours))],
        **settings)


def multi_spd_plot(spds,
                   cmfs='CIE 1931 2 Degree Standard Observer',
                   use_spds_colours=False,
                   normalise_spds_colours=False,
                   **kwargs):
    """
    Plots given spectral power distributions.

    Parameters
    ----------
    spds : list
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for spectrum creation.
    use_spds_colours : bool, optional
        Use spectral power distributions colours.
    normalise_spds_colours : bool
        Should spectral power distributions colours normalised.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> from colour import SpectralPowerDistribution
    >>> data1 = {400: 0.0641, 420: 0.0645, 440: 0.0562}
    >>> data2 = {400: 0.134, 420: 0.789, 440: 1.289}
    >>> spd1 = SpectralPowerDistribution('Custom1', data1)
    >>> spd2 = SpectralPowerDistribution('Custom2', data2)
    >>> multi_spd_plot([spd1, spd2])  # doctest: +SKIP
    True
    """

    canvas(**kwargs)

    cmfs = get_cmfs(cmfs)

    if use_spds_colours:
        illuminant = ILLUMINANTS_RELATIVE_SPDS.get('D65')

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for spd in spds:
        wavelengths, values = tuple(zip(*spd.items))

        shape = spd.shape
        x_limit_min.append(shape.start)
        x_limit_max.append(shape.end)
        y_limit_min.append(min(values))
        y_limit_max.append(max(values))

        matplotlib.pyplot.rc("axes", color_cycle=["r", "g", "b", "y"])

        if use_spds_colours:
            XYZ = spectral_to_XYZ(spd, cmfs, illuminant) / 100
            if normalise_spds_colours:
                XYZ = normalise(XYZ, clip=False)
            RGB = np.clip(XYZ_to_sRGB(XYZ), 0, 1)

            pylab.plot(wavelengths, values, color=RGB, label=spd.title,
                       linewidth=2)
        else:
            pylab.plot(wavelengths, values, label=spd.title, linewidth=2)

    settings = {
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Power Distribution',
        'x_tighten': True,
        'legend': True,
        'legend_location': 'upper left',
        'x_ticker': True,
        'y_ticker': True,
        'limits': (min(x_limit_min), max(x_limit_max),
                   min(y_limit_min), max(y_limit_max))}
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def single_cmfs_plot(cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
    """
    Plots given colour matching functions.

    Parameters
    ----------
    cmfs : unicode, optional
        Colour matching functions to plot.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> single_cmfs_plot()  # doctest: +SKIP
    True
    """

    cmfs = get_cmfs(cmfs)
    settings = {
        'title': '{0} - Colour Matching Functions'.format(cmfs.title)}
    settings.update(kwargs)

    return multi_cmfs_plot((cmfs.name, ), **settings)


def multi_cmfs_plot(cmfs=None, **kwargs):
    """
    Plots given colour matching functions.

    Parameters
    ----------
    cmfs : array_like, optional
        Colour matching functions to plot.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> cmfs = [
    ... 'CIE 1931 2 Degree Standard Observer',
    ... 'CIE 1964 10 Degree Standard Observer']
    >>> multi_cmfs_plot(cmfs)  # doctest: +SKIP
    True
    """

    canvas(**kwargs)

    if cmfs is None:
        cmfs = ('CIE 1931 2 Degree Standard Observer',
                'CIE 1964 10 Degree Standard Observer')

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for axis, rgb in (('x', (1, 0, 0)),
                      ('y', (0, 1, 0)),
                      ('z', (0, 0, 1))):
        for i, cmfs_i in enumerate(cmfs):
            cmfs_i = get_cmfs(cmfs_i)

            rgb = [reduce(lambda y, _: y * 0.5, range(i), x) for x in rgb]
            wavelengths, values = tuple(
                zip(*[(key, value) for key, value in getattr(cmfs_i, axis)]))

            shape = cmfs_i.shape
            x_limit_min.append(shape.start)
            x_limit_max.append(shape.end)
            y_limit_min.append(min(values))
            y_limit_max.append(max(values))

            pylab.plot(wavelengths,
                       values,
                       color=rgb,
                       label=u'{0} - {1}'.format(
                           cmfs_i.labels.get(axis), cmfs_i.title),
                       linewidth=2)

    settings = {
        'title': '{0} - Colour Matching Functions'.format(', '.join(
            [get_cmfs(cmfs_i).title for cmfs_i in cmfs])),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Tristimulus Values',
        'x_tighten': True,
        'legend': True,
        'legend_location': 'upper right',
        'x_ticker': True,
        'y_ticker': True,
        'grid': True,
        'y_axis_line': True,
        'limits': (min(x_limit_min), max(x_limit_max),
                   min(y_limit_min), max(y_limit_max))}
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def single_illuminant_relative_spd_plot(
        illuminant='A',
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots given single illuminant relative spectral power distribution.

    Parameters
    ----------
    illuminant : unicode, optional
        Factory illuminant to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions to plot.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> single_illuminant_relative_spd_plot()  # doctest: +SKIP
    True
    """

    cmfs = get_cmfs(cmfs)
    title = 'Illuminant {0} - {1}'.format(illuminant, cmfs.title)

    illuminant = get_illuminant(illuminant)

    settings = {
        'title': title,
        'y_label': 'Relative Spectral Power Distribution'}
    settings.update(kwargs)

    return single_spd_plot(illuminant, **settings)


def multi_illuminants_relative_spd_plot(illuminants=None, **kwargs):
    """
    Plots given illuminants relative spectral power distributions.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> multi_illuminants_relative_spd_plot(['A', 'B', 'C'])  # doctest: +SKIP
    True
    """

    if illuminants is None:
        illuminants = ('A', 'B', 'C')

    spds = []
    for illuminant in illuminants:
        spds.append(get_illuminant(illuminant))

    settings = {
        'title': (
            '{0} - Illuminants Relative Spectral Power Distribution').format(
            ', '.join([spd.title for spd in spds])),
        'y_label': 'Relative Spectral Power Distribution'}
    settings.update(kwargs)

    return multi_spd_plot(spds, **settings)


def visible_spectrum_plot(cmfs='CIE 1931 2 Degree Standard Observer',
                          **kwargs):
    """
    Plots the visible colours spectrum using given standard observer *CIE XYZ*
    colour matching functions.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for spectrum creation.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> visible_spectrum_plot()  # doctest: +SKIP
    True
    """

    cmfs = get_cmfs(cmfs)
    cmfs = cmfs.clone().align(DEFAULT_SPECTRAL_SHAPE)

    wavelengths = cmfs.shape.range()

    colours = XYZ_to_sRGB(wavelength_to_XYZ(wavelengths, cmfs))
    colours *= 1 / np.max(colours)
    colours = np.clip(colours, 0, 1)

    settings = {
        'title': 'The Visible Spectrum - {0}'.format(cmfs.title),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'x_tighten': True}
    settings.update(kwargs)

    return colour_parameters_plot([colour_parameter(x=x[0], RGB=x[1])
                                   for x in tuple(zip(wavelengths, colours))],
                                  **settings)


def single_lightness_function_plot(function='CIE 1976', **kwargs):
    """
    Plots given *Lightness* function.

    Parameters
    ----------
    function : unicode, optional
        *Lightness* function to plot.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> single_lightness_function_plot()  # doctest: +SKIP
    True
    """

    settings = {
        'title': '{0} - Lightness Function'.format(function)}
    settings.update(kwargs)

    return multi_lightness_function_plot((function, ), **settings)


def multi_lightness_function_plot(functions=None, **kwargs):
    """
    Plots given *Lightness* functions.

    Parameters
    ----------
    functions : array_like, optional
        *Lightness* functions to plot.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Raises
    ------
    KeyError
        If one of the given *Lightness* function is not found in the factory
        *Lightness* functions.

    Examples
    --------
    >>> fs = ('CIE 1976', 'Wyszecki 1963')
    >>> multi_lightness_function_plot(fs)  # doctest: +SKIP
    True
    """

    settings = {
        'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if functions is None:
        functions = ('CIE 1976', 'Wyszecki 1963')

    samples = np.linspace(0, 100, 1000)
    for function in functions:
        function, name = LIGHTNESS_METHODS.get(function), function
        if function is None:
            raise KeyError(
                ('"{0}" "Lightness" function not found in factory '
                 '"Lightness" functions: "{1}".').format(
                    name, sorted(LIGHTNESS_METHODS.keys())))

        pylab.plot(samples,
                   [function(x) for x in samples],
                   label='{0}'.format(name),
                   linewidth=2)

    settings.update({
        'title': '{0} - Lightness Functions'.format(', '.join(functions)),
        'x_label': 'Luminance Y',
        'y_label': 'Lightness L*',
        'x_tighten': True,
        'legend': True,
        'legend_location': 'upper left',
        'x_ticker': True,
        'y_ticker': True,
        'grid': True,
        'limits': (0, 100, 0, 100),
        'aspect': 'equal'})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def blackbody_spectral_radiance_plot(
        temperature=3500,
        cmfs='CIE 1931 2 Degree Standard Observer',
        blackbody='VY Canis Major',
        **kwargs):
    """
    Plots given blackbody spectral radiance.

    Parameters
    ----------
    temperature : numeric, optional
        Blackbody temperature.
    cmfs : unicode, optional
        Standard observer colour matching functions.
    blackbody : unicode, optional
        Blackbody name.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> blackbody_spectral_radiance_plot()  # doctest: +SKIP
    True
    """

    canvas(**kwargs)

    cmfs = get_cmfs(cmfs)

    matplotlib.pyplot.subplots_adjust(hspace=0.4)

    spd = blackbody_spd(temperature, cmfs.shape)

    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.subplot(211)

    settings = {
        'title': '{0} - Spectral Radiance'.format(blackbody),
        'y_label': 'W / (sr m$^2$) / m',
        'standalone': False}
    settings.update(kwargs)

    single_spd_plot(spd, cmfs.name, **settings)

    XYZ = spectral_to_XYZ(spd, cmfs)
    RGB = normalise(XYZ_to_sRGB(XYZ / 100))

    matplotlib.pyplot.subplot(212)

    settings = {'title': '{0} - Colour'.format(blackbody),
                'x_label': '{0}K'.format(temperature),
                'y_label': '',
                'aspect': None,
                'standalone': False}

    single_colour_plot(colour_parameter(name='', RGB=RGB), **settings)

    settings = {
        'standalone': True}
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)
    return display(**settings)


def blackbody_colours_plot(shape=SpectralShape(150, 12500, 50),
                           cmfs='CIE 1931 2 Degree Standard Observer',
                           **kwargs):
    """
    Plots blackbody colours.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape to use as plot boundaries.
    cmfs : unicode, optional
        Standard observer colour matching functions.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> blackbody_colours_plot()  # doctest: +SKIP
    True
    """

    cmfs = get_cmfs(cmfs)

    colours = []
    temperatures = []

    for temperature in shape:
        spd = blackbody_spd(temperature, cmfs.shape)

        XYZ = spectral_to_XYZ(spd, cmfs)
        RGB = normalise(XYZ_to_sRGB(XYZ / 100))

        colours.append(RGB)
        temperatures.append(temperature)

    settings = {
        'title': 'Blackbody Colours',
        'x_label': 'Temperature K',
        'y_label': '',
        'x_tighten': True,
        'x_ticker': True,
        'y_ticker': False}
    settings.update(kwargs)

    return colour_parameters_plot([colour_parameter(x=x[0], RGB=x[1])
                                   for x in tuple(zip(temperatures, colours))],
                                  **settings)
