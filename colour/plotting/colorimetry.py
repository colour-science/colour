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

References
----------
.. [1]  Spiker, N. (2015). Private Discussion with Mansencal, T. Retrieved from
        http://www.repairfaq.org/sam/repspec/
"""

from __future__ import division

import matplotlib.pyplot
import numpy as np
import pylab
from six.moves import reduce

from colour.colorimetry import (
    DEFAULT_SPECTRAL_SHAPE,
    ILLUMINANTS,
    ILLUMINANTS_RELATIVE_SPDS,
    LIGHTNESS_METHODS,
    SpectralShape,
    blackbody_spd,
    spectral_to_XYZ,
    wavelength_to_XYZ)
from colour.models import XYZ_to_sRGB
from colour.plotting import (
    ColourParameter,
    DEFAULT_PLOTTING_ENCODING_CCTF,
    DEFAULT_FIGURE_WIDTH,
    boundaries,
    canvas,
    colour_parameters_plot,
    decorate,
    display,
    get_cmfs,
    get_illuminant,
    single_colour_plot)
from colour.utilities import normalise_maximum

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['single_spd_plot',
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


def single_spd_plot(spd,
                    cmfs='CIE 1931 2 Degree Standard Observer',
                    out_of_gamut_clipping=True,
                    **kwargs):
    """
    Plots given spectral power distribution.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution to plot.
    out_of_gamut_clipping : bool, optional
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother. [1]_
    cmfs : unicode
        Standard observer colour matching functions used for spectrum creation.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`boundaries`, :func:`canvas`, :func:`decorate`,
        :func:`display`},
        Please refer to the documentation of the previously listed definitions.
    y0_plot : bool, optional
        {:func:`colour_parameters_plot`},
        Whether to plot *y0* line.
    y1_plot : bool, optional
        {:func:`colour_parameters_plot`},
        Whether to plot *y1* line.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> from colour import SpectralPowerDistribution
    >>> data = {400: 0.0641, 420: 0.0645, 440: 0.0562}
    >>> spd = SpectralPowerDistribution('Custom', data)
    >>> single_spd_plot(spd)  # doctest: +SKIP
    """

    cmfs = get_cmfs(cmfs)

    shape = cmfs.shape
    spd = spd.clone().interpolate(shape, 'Linear')
    wavelengths = spd.wavelengths
    values = spd.values

    y1 = values
    colours = XYZ_to_sRGB(
        wavelength_to_XYZ(wavelengths, cmfs),
        ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E'],
        apply_encoding_cctf=False)

    if not out_of_gamut_clipping:
        colours += np.abs(np.min(colours))

    colours = DEFAULT_PLOTTING_ENCODING_CCTF(normalise_maximum(colours))

    settings = {
        'title': '{0} - {1}'.format(spd.title, cmfs.title),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Power Distribution',
        'x_tighten': True,
        'y_tighten': True}

    settings.update(kwargs)

    return colour_parameters_plot(
        [ColourParameter(x=x[0], y1=x[1], RGB=x[2])
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
        Whether to use spectral power distributions colours.
    normalise_spds_colours : bool
        Whether to normalise spectral power distributions colours.

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
    >>> from colour import SpectralPowerDistribution
    >>> data1 = {400: 0.0641, 420: 0.0645, 440: 0.0562}
    >>> data2 = {400: 0.134, 420: 0.789, 440: 1.289}
    >>> spd1 = SpectralPowerDistribution('Custom1', data1)
    >>> spd2 = SpectralPowerDistribution('Custom2', data2)
    >>> multi_spd_plot([spd1, spd2])  # doctest: +SKIP
    """

    canvas(**kwargs)

    cmfs = get_cmfs(cmfs)

    if use_spds_colours:
        illuminant = ILLUMINANTS_RELATIVE_SPDS['D65']

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for spd in spds:
        wavelengths, values = tuple(zip(*spd.items))

        shape = spd.shape
        x_limit_min.append(shape.start)
        x_limit_max.append(shape.end)
        y_limit_min.append(min(values))
        y_limit_max.append(max(values))

        if use_spds_colours:
            XYZ = spectral_to_XYZ(spd, cmfs, illuminant) / 100
            if normalise_spds_colours:
                XYZ = normalise_maximum(XYZ, clip=False)
            RGB = np.clip(XYZ_to_sRGB(XYZ), 0, 1)

            pylab.plot(wavelengths, values, color=RGB, label=spd.title,
                       linewidth=2)
        else:
            pylab.plot(wavelengths, values, label=spd.title, linewidth=2)

    settings = {
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Power Distribution',
        'x_tighten': True,
        'y_tighten': True,
        'legend': True,
        'legend_location': 'upper left',
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
    >>> single_cmfs_plot()  # doctest: +SKIP
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
    >>> cmfs = [
    ... 'CIE 1931 2 Degree Standard Observer',
    ... 'CIE 1964 10 Degree Standard Observer']
    >>> multi_cmfs_plot(cmfs)  # doctest: +SKIP
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

            rgb = [reduce(lambda y, _: y * 0.5, range(i), x)  # noqa
                   for x in rgb]
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
                           cmfs_i.labels[axis], cmfs_i.title),
                       linewidth=2)

    settings = {
        'title': '{0} - Colour Matching Functions'.format(', '.join(
            [get_cmfs(c).title for c in cmfs])),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Tristimulus Values',
        'x_tighten': True,
        'y_tighten': True,
        'legend': True,
        'legend_location': 'upper right',
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
    >>> single_illuminant_relative_spd_plot()  # doctest: +SKIP
    """

    cmfs = get_cmfs(cmfs)
    title = 'Illuminant {0} - {1}'.format(illuminant, cmfs.title)

    illuminant = get_illuminant(illuminant)

    settings = {
        'title': title,
        'y_label': 'Relative Power'}
    settings.update(kwargs)

    return single_spd_plot(illuminant, **settings)


def multi_illuminants_relative_spd_plot(illuminants=None, **kwargs):
    """
    Plots given illuminants relative spectral power distributions.

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
    use_spds_colours : bool, optional
        {:func:`multi_spd_plot`}
        Whether to use spectral power distributions colours.
    normalise_spds_colours : bool
        {:func:`multi_spd_plot`}
        Whether to normalise spectral power distributions colours.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> multi_illuminants_relative_spd_plot(['A', 'B', 'C'])  # doctest: +SKIP
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
        'y_label': 'Relative Power'}
    settings.update(kwargs)

    return multi_spd_plot(spds, **settings)


def visible_spectrum_plot(cmfs='CIE 1931 2 Degree Standard Observer',
                          out_of_gamut_clipping=True,
                          **kwargs):
    """
    Plots the visible colours spectrum using given standard observer *CIE XYZ*
    colour matching functions.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for spectrum creation.
    out_of_gamut_clipping : bool, optional
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother. [1]_

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`boundaries`, :func:`canvas`, :func:`decorate`,
        :func:`display`},
        Please refer to the documentation of the previously listed definitions.
    y0_plot : bool, optional
        {:func:`colour_parameters_plot`},
        Whether to plot *y0* line.
    y1_plot : bool, optional
        {:func:`colour_parameters_plot`},
        Whether to plot *y1* line.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> visible_spectrum_plot()  # doctest: +SKIP
    """

    cmfs = get_cmfs(cmfs)
    cmfs = cmfs.clone().align(DEFAULT_SPECTRAL_SHAPE)

    wavelengths = cmfs.shape.range()

    colours = XYZ_to_sRGB(
        wavelength_to_XYZ(wavelengths, cmfs),
        ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E'],
        apply_encoding_cctf=False)

    if not out_of_gamut_clipping:
        colours += np.abs(np.min(colours))

    colours = DEFAULT_PLOTTING_ENCODING_CCTF(normalise_maximum(colours))

    settings = {
        'title': 'The Visible Spectrum - {0}'.format(cmfs.title),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': False,
        'x_tighten': True,
        'y_tighten': True,
        'y_ticker': False}
    settings.update(kwargs)

    return colour_parameters_plot([ColourParameter(x=x[0], RGB=x[1])
                                   for x in tuple(zip(wavelengths, colours))],
                                  **settings)


def single_lightness_function_plot(function='CIE 1976', **kwargs):
    """
    Plots given *Lightness* function.

    Parameters
    ----------
    function : unicode, optional
        *Lightness* function to plot.

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
    >>> single_lightness_function_plot()  # doctest: +SKIP
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

    Raises
    ------
    KeyError
        If one of the given *Lightness* function is not found in the factory
        *Lightness* functions.

    Examples
    --------
    >>> fs = ('CIE 1976', 'Wyszecki 1963')
    >>> multi_lightness_function_plot(fs)  # doctest: +SKIP
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
        'x_label': 'Relative Luminance Y',
        'y_label': 'Lightness',
        'x_tighten': True,
        'legend': True,
        'legend_location': 'upper left',
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
    >>> blackbody_spectral_radiance_plot()  # doctest: +SKIP
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
    RGB = normalise_maximum(XYZ_to_sRGB(XYZ / 100))

    matplotlib.pyplot.subplot(212)

    settings = {'title': '{0} - Colour'.format(blackbody),
                'x_label': '{0}K'.format(temperature),
                'y_label': '',
                'aspect': None,
                'standalone': False}

    single_colour_plot(ColourParameter(name='', RGB=RGB), **settings)

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

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`boundaries`, :func:`canvas`, :func:`decorate`,
        :func:`display`},
        Please refer to the documentation of the previously listed definitions.
    y0_plot : bool, optional
        {:func:`colour_parameters_plot`},
        Whether to plot *y0* line.
    y1_plot : bool, optional
        {:func:`colour_parameters_plot`},
        Whether to plot *y1* line.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> blackbody_colours_plot()  # doctest: +SKIP
    """

    cmfs = get_cmfs(cmfs)

    colours = []
    temperatures = []

    for temperature in shape:
        spd = blackbody_spd(temperature, cmfs.shape)

        XYZ = spectral_to_XYZ(spd, cmfs)
        RGB = normalise_maximum(XYZ_to_sRGB(XYZ / 100))

        colours.append(RGB)
        temperatures.append(temperature)

    settings = {
        'title': 'Blackbody Colours',
        'x_label': 'Temperature K',
        'y_label': '',
        'x_tighten': True,
        'y_tighten': True,
        'y_ticker': False}
    settings.update(kwargs)

    return colour_parameters_plot([ColourParameter(x=x[0], RGB=x[1])
                                   for x in tuple(zip(temperatures, colours))],
                                  **settings)
