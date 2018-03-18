# -*- coding: utf-8 -*-
"""
Colorimetry Plotting
====================

Defines the colorimetry plotting objects:

-   :func:`colour.plotting.single_spd_plot`
-   :func:`colour.plotting.multi_spd_plot`
-   :func:`colour.plotting.single_cmfs_plot`
-   :func:`colour.plotting.multi_cmfs_plot`
-   :func:`colour.plotting.single_illuminant_spd_plot`
-   :func:`colour.plotting.multi_illuminant_spd_plot`
-   :func:`colour.plotting.visible_spectrum_plot`
-   :func:`colour.plotting.single_lightness_function_plot`
-   :func:`colour.plotting.multi_lightness_function_plot`
-   :func:`colour.plotting.blackbody_spectral_radiance_plot`
-   :func:`colour.plotting.blackbody_colours_plot`

References
----------
-   :cite:`Spiker2015a` : Spiker, N. (2015). Private Discussion with
    Mansencal, T. Retrieved from http://www.invisiblelightimages.com/
"""

from __future__ import division

import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from matplotlib.patches import Polygon
from six.moves import reduce

from colour.algebra import LinearInterpolator
from colour.colorimetry import (
    ILLUMINANTS, ILLUMINANTS_SPDS, LIGHTNESS_METHODS, SpectralShape,
    blackbody_spd, ones_spd, spectral_to_XYZ, wavelength_to_XYZ)
from colour.plotting import (ColourSwatch, COLOUR_STYLE_CONSTANTS,
                             XYZ_to_plotting_colourspace, artist, filter_cmfs,
                             filter_illuminants, override_style, render,
                             single_colour_swatch_plot)
from colour.utilities import (domain_range_scale, first_item,
                              normalise_maximum, suppress_warnings, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'single_spd_plot', 'multi_spd_plot', 'single_cmfs_plot', 'multi_cmfs_plot',
    'single_illuminant_spd_plot', 'multi_illuminant_spd_plot',
    'visible_spectrum_plot', 'single_lightness_function_plot',
    'multi_lightness_function_plot', 'blackbody_spectral_radiance_plot',
    'blackbody_colours_plot'
]


@override_style()
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
        gray background, less saturated and smoother.
    cmfs : unicode
        Standard observer colour matching functions used for spectrum creation.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    References
    ----------
    -   :cite:`Spiker2015a`

    Examples
    --------
    >>> from colour import SpectralPowerDistribution
    >>> data = {
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360
    ... }
    >>> spd = SpectralPowerDistribution(data, name='Custom')
    >>> single_spd_plot(spd)  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_SPD_Plot.png
        :align: center
        :alt: single_spd_plot
    """

    figure, axes = artist(**kwargs)

    cmfs = first_item(filter_cmfs(cmfs))

    spd = spd.copy()
    spd.interpolator = LinearInterpolator
    wavelengths = cmfs.wavelengths[np.logical_and(
        cmfs.wavelengths >= max(min(cmfs.wavelengths), min(spd.wavelengths)),
        cmfs.wavelengths <= min(max(cmfs.wavelengths), max(spd.wavelengths)),
    )]
    values = spd[wavelengths]

    colours = XYZ_to_plotting_colourspace(
        wavelength_to_XYZ(wavelengths, cmfs),
        ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E'],
        apply_encoding_cctf=False)

    if not out_of_gamut_clipping:
        colours += np.abs(np.min(colours))

    colours = COLOUR_STYLE_CONSTANTS.colour.colourspace.encoding_cctf(
        normalise_maximum(colours))

    x_min, x_max = min(wavelengths), max(wavelengths)
    y_min, y_max = 0, max(values) + max(values) * 0.05

    polygon = Polygon(
        np.vstack([
            (x_min, 0),
            tstack((wavelengths, values)),
            (x_max, 0),
        ]),
        facecolor='none',
        edgecolor='none')
    axes.add_patch(polygon)

    padding = 0.1
    axes.bar(
        x=wavelengths - padding,
        height=max(values),
        width=1 + padding,
        color=colours,
        align='edge',
        clip_path=polygon)

    axes.plot(wavelengths, values, color=COLOUR_STYLE_CONSTANTS.colour.dark)

    settings = {
        'axes': axes,
        'bounding_box': (x_min, x_max, y_min, y_max),
        'title': '{0} - {1}'.format(spd.strict_name, cmfs.strict_name),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Power Distribution',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
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
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import SpectralPowerDistribution
    >>> data_1 = {
    ...     500: 0.004900,
    ...     510: 0.009300,
    ...     520: 0.063270,
    ...     530: 0.165500,
    ...     540: 0.290400,
    ...     550: 0.433450,
    ...     560: 0.594500
    ... }
    >>> data_2 = {
    ...     500: 0.323000,
    ...     510: 0.503000,
    ...     520: 0.710000,
    ...     530: 0.862000,
    ...     540: 0.954000,
    ...     550: 0.994950,
    ...     560: 0.995000
    ... }
    >>> spd1 = SpectralPowerDistribution(data_1, name='Custom 1')
    >>> spd2 = SpectralPowerDistribution(data_2, name='Custom 2')
    >>> multi_spd_plot([spd1, spd2])  # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_SPD_Plot.png
        :align: center
        :alt: multi_spd_plot
    """

    figure, axes = artist(**kwargs)

    cmfs = first_item(filter_cmfs(cmfs))

    illuminant = ILLUMINANTS_SPDS[
        COLOUR_STYLE_CONSTANTS.colour.colourspace.illuminant]

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for spd in spds:
        wavelengths, values = spd.wavelengths, spd.values

        shape = spd.shape
        x_limit_min.append(shape.start)
        x_limit_max.append(shape.end)
        y_limit_min.append(min(values))
        y_limit_max.append(max(values))

        if use_spds_colours:
            with domain_range_scale('1'):
                XYZ = spectral_to_XYZ(spd, cmfs, illuminant)

            if normalise_spds_colours:
                XYZ = normalise_maximum(XYZ, clip=False)

            RGB = np.clip(XYZ_to_plotting_colourspace(XYZ), 0, 1)

            axes.plot(wavelengths, values, color=RGB, label=spd.strict_name)
        else:
            axes.plot(wavelengths, values, label=spd.strict_name)

    bounding_box = (min(x_limit_min), max(x_limit_max), min(y_limit_min),
                    max(y_limit_max) + max(y_limit_max) * 0.05)
    settings = {
        'axes': axes,
        'bounding_box': bounding_box,
        'legend': True,
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Power Distribution',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
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
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> single_cmfs_plot('CIE 1931 2 Degree Standard Observer')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_CMFS_Plot.png
        :align: center
        :alt: single_cmfs_plot
    """

    cmfs = first_item(filter_cmfs(cmfs))
    settings = {
        'title': '{0} - Colour Matching Functions'.format(cmfs.strict_name)
    }
    settings.update(kwargs)

    return multi_cmfs_plot((cmfs.name, ), **settings)


@override_style()
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
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> cmfs = ('CIE 1931 2 Degree Standard Observer',
    ...         'CIE 1964 10 Degree Standard Observer')
    >>> multi_cmfs_plot(cmfs)  # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_CMFS_Plot.png
        :align: center
        :alt: multi_cmfs_plot
    """

    if cmfs is None:
        cmfs = ('CIE 1931 2 Degree Standard Observer',
                'CIE 1964 10 Degree Standard Observer')

    cmfs = list(
        OrderedDict.fromkeys(
            itertools.chain.from_iterable(
                [filter_cmfs(cmfs_i) for cmfs_i in cmfs])))

    figure, axes = artist(**kwargs)

    axes.axhline(color=COLOUR_STYLE_CONSTANTS.colour.dark, linestyle='--')

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for i, cmfs_i in enumerate(cmfs):
        for j, RGB in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            RGB = [reduce(lambda y, _: y * 0.5, range(i), x) for x in RGB]
            values = cmfs_i.values[:, j]

            shape = cmfs_i.shape
            x_limit_min.append(shape.start)
            x_limit_max.append(shape.end)
            y_limit_min.append(min(values))
            y_limit_max.append(max(values))

            axes.plot(
                cmfs_i.wavelengths,
                values,
                color=RGB,
                label=u'{0} - {1}'.format(cmfs_i.strict_labels[j],
                                          cmfs_i.strict_name))

    bounding_box = (min(x_limit_min), max(x_limit_max),
                    min(y_limit_min) - abs(min(y_limit_min)) * 0.05,
                    max(y_limit_max) + abs(max(y_limit_max)) * 0.05)
    title = '{0} - Colour Matching Functions'.format(', '.join(
        [cmfs_i.strict_name for cmfs_i in cmfs]))

    settings = {
        'axes': axes,
        'bounding_box': bounding_box,
        'legend': True,
        'title': title,
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Tristimulus Values',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def single_illuminant_spd_plot(illuminant='A',
                               cmfs='CIE 1931 2 Degree Standard Observer',
                               **kwargs):
    """
    Plots given single illuminant spectral power distribution.

    Parameters
    ----------
    illuminant : unicode, optional
        Factory illuminant to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions to plot.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    out_of_gamut_clipping : bool, optional
        {:func:`colour.plotting.single_spd_plot`},
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother.

    Returns
    -------
    tuple
        Current figure and axes.

    References
    ----------
    -   :cite:`Spiker2015a`

    Examples
    --------
    >>> single_illuminant_spd_plot('A')  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_Illuminant_SPD_Plot.png
        :align: center
        :alt: single_illuminant_spd_plot
    """

    cmfs = first_item(filter_cmfs(cmfs))
    title = 'Illuminant {0} - {1}'.format(illuminant, cmfs.strict_name)

    illuminant = first_item(filter_illuminants(illuminant))

    settings = {'title': title, 'y_label': 'Relative Power'}
    settings.update(kwargs)

    return single_spd_plot(illuminant, **settings)


@override_style()
def multi_illuminant_spd_plot(illuminants=None, **kwargs):
    """
    Plots given illuminants spectral power distributions.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    use_spds_colours : bool, optional
        {:func:`colour.plotting.multi_spd_plot`}
        Whether to use spectral power distributions colours.
    normalise_spds_colours : bool
        {:func:`colour.plotting.multi_spd_plot`}
        Whether to normalise spectral power distributions colours.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> multi_illuminant_spd_plot(['A', 'B', 'C'])  # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_Illuminant_SPD_Plot.png
        :align: center
        :alt: multi_illuminant_spd_plot
    """

    if illuminants is None:
        illuminants = ('A', 'B', 'C')

    illuminants = list(
        OrderedDict.fromkeys(
            itertools.chain.from_iterable([
                filter_illuminants(illuminant) for illuminant in illuminants
            ])))

    title = '{0} - Illuminants Spectral Power Distributions'.format(
        ', '.join([illuminant.strict_name for illuminant in illuminants]))

    settings = {'title': title, 'y_label': 'Relative Power'}
    settings.update(kwargs)

    return multi_spd_plot(illuminants, **settings)


@override_style(**{
    'ytick.left': False,
    'ytick.labelleft': False,
})
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
        gray background, less saturated and smoother.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    References
    ----------
    -   :cite:`Spiker2015a`

    Examples
    --------
    >>> visible_spectrum_plot()  # doctest: +SKIP

    .. image:: ../_static/Plotting_Visible_Spectrum_Plot.png
        :align: center
        :alt: visible_spectrum_plot
    """

    cmfs = first_item(filter_cmfs(cmfs))

    bounding_box = (min(cmfs.wavelengths), max(cmfs.wavelengths), 0, 1)

    settings = {'bounding_box': bounding_box, 'y_label': None}
    settings.update(kwargs)
    settings['standalone'] = False

    figure, axes = single_spd_plot(
        ones_spd(cmfs.shape),
        cmfs=cmfs,
        out_of_gamut_clipping=out_of_gamut_clipping,
        **settings)

    # Removing wavelength line as it doubles with the axes spine.
    axes.lines.pop(0)

    settings = {
        'axes': axes,
        'standalone': True,
        'title': 'The Visible Spectrum - {0}'.format(cmfs.strict_name),
        'x_label': 'Wavelength $\\lambda$ (nm)',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
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
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> single_lightness_function_plot('CIE 1976')  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_Lightness_Function_Plot.png
        :align: center
        :alt: single_lightness_function_plot
    """

    settings = {'title': '{0} - Lightness Function'.format(function)}
    settings.update(kwargs)

    return multi_lightness_function_plot((function, ), **settings)


@override_style()
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
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Raises
    ------
    KeyError
        If one of the given *Lightness* function is not found in the factory
        *Lightness* functions.

    Examples
    --------
    >>> multi_lightness_function_plot(['CIE 1976', 'Wyszecki 1963'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_Lightness_Function_Plot.png
        :align: center
        :alt: multi_lightness_function_plot
    """

    if functions is None:
        functions = ('CIE 1976', 'Wyszecki 1963')

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    samples = np.linspace(0, 100, 1000)
    for function in functions:
        function, name = LIGHTNESS_METHODS.get(function), function
        if function is None:
            raise KeyError(('"{0}" "Lightness" function not found in factory '
                            '"Lightness" functions: "{1}".').format(
                                name, sorted(LIGHTNESS_METHODS.keys())))

        axes.plot(samples, function(samples), label='{0}'.format(name))

    settings = {
        'axes': axes,
        'aspect': 'equal',
        'bounding_box': (0, 100, 0, 100),
        'legend': True,
        'title': '{0} - Lightness Functions'.format(', '.join(functions)),
        'x_label': 'Relative Luminance Y',
        'y_label': 'Lightness',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
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
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> blackbody_spectral_radiance_plot(3500, blackbody='VY Canis Major')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Blackbody_Spectral_Radiance_Plot.png
        :align: center
        :alt: blackbody_spectral_radiance_plot
    """

    figure = plt.figure()

    figure.subplots_adjust(hspace=COLOUR_STYLE_CONSTANTS.geometry.short / 2)

    cmfs = first_item(filter_cmfs(cmfs))

    spd = blackbody_spd(temperature, cmfs.shape)

    axes = figure.add_subplot(211)
    settings = {
        'axes': axes,
        'title': '{0} - Spectral Radiance'.format(blackbody),
        'y_label': 'W / (sr m$^2$) / m',
    }
    settings.update(kwargs)
    settings['standalone'] = False

    single_spd_plot(spd, cmfs.name, **settings)

    axes = figure.add_subplot(212)

    with domain_range_scale('1'):
        XYZ = spectral_to_XYZ(spd, cmfs)

    RGB = normalise_maximum(XYZ_to_plotting_colourspace(XYZ))

    settings = {
        'axes': axes,
        'aspect': None,
        'title': '{0} - Colour'.format(blackbody),
        'x_label': '{0}K'.format(temperature),
        'y_label': '',
    }
    settings.update(kwargs)
    settings['standalone'] = False

    figure, axes = single_colour_swatch_plot(
        ColourSwatch(name='', RGB=RGB), **settings)

    # Removing "x" and "y" ticks.
    axes.set_xticks([])
    axes.set_yticks([])

    settings = {'axes': axes, 'standalone': True}
    settings.update(kwargs)

    return render(**settings)


@override_style(**{
    'ytick.left': False,
    'ytick.labelleft': False,
})
def blackbody_colours_plot(
        shape=SpectralShape(150, 12500, 50),
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
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> blackbody_colours_plot(SpectralShape(150, 12500, 50))  # doctest: +SKIP

    .. image:: ../_static/Plotting_Blackbody_Colours_Plot.png
        :align: center
        :alt: blackbody_colours_plot
    """

    figure, axes = artist(**kwargs)

    cmfs = first_item(filter_cmfs(cmfs))

    colours = []
    temperatures = []

    with suppress_warnings():
        for temperature in shape:
            spd = blackbody_spd(temperature, cmfs.shape)

            with domain_range_scale('1'):
                XYZ = spectral_to_XYZ(spd, cmfs)

            RGB = normalise_maximum(XYZ_to_plotting_colourspace(XYZ))

            colours.append(RGB)
            temperatures.append(temperature)

    x_min, x_max = min(temperatures), max(temperatures)
    y_min, y_max = 0, 1

    padding = 0.1
    axes.bar(
        x=np.array(temperatures) - padding,
        height=1,
        width=shape.interval + (padding * shape.interval),
        color=colours,
        align='edge')

    settings = {
        'axes': axes,
        'bounding_box': (x_min, x_max, y_min, y_max),
        'title': 'Blackbody Colours',
        'x_label': 'Temperature K',
        'y_label': None,
    }
    settings.update(kwargs)

    return render(**settings)
