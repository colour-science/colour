# -*- coding: utf-8 -*-
"""
Colorimetry Plotting
====================

Defines the colorimetry plotting objects:

-   :func:`colour.plotting.single_spd_plot`
-   :func:`colour.plotting.multi_spd_plot`
-   :func:`colour.plotting.single_cmfs_plot`
-   :func:`colour.plotting.multi_cmfs_plot`
-   :func:`colour.plotting.single_illuminant_relative_spd_plot`
-   :func:`colour.plotting.multi_illuminant_relative_spd_plot`
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

import matplotlib.pyplot
import numpy as np
import pylab
from matplotlib.patches import Polygon
from six.moves import reduce

from colour.algebra import LinearInterpolator
from colour.colorimetry import (
    ILLUMINANTS, ILLUMINANTS_SPDS, LIGHTNESS_METHODS, SpectralShape,
    blackbody_spd, ones_spd, spectral_to_XYZ, wavelength_to_XYZ)
from colour.plotting import (ColourSwatch, DEFAULT_PLOTTING_COLOURSPACE,
                             DEFAULT_FIGURE_WIDTH, XYZ_to_plotting_colourspace,
                             canvas, get_cmfs, get_illuminant, render,
                             single_colour_swatch_plot)
from colour.utilities import (domain_range_scale, normalise_maximum,
                              suppress_warnings, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'single_spd_plot', 'multi_spd_plot', 'single_cmfs_plot', 'multi_cmfs_plot',
    'single_illuminant_relative_spd_plot',
    'multi_illuminant_relative_spd_plot', 'visible_spectrum_plot',
    'single_lightness_function_plot', 'multi_lightness_function_plot',
    'blackbody_spectral_radiance_plot', 'blackbody_colours_plot'
]


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

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

    axes = canvas(**kwargs).gca()

    cmfs = get_cmfs(cmfs)

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

    colours = DEFAULT_PLOTTING_COLOURSPACE.encoding_cctf(
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
    axes.bar(
        x=wavelengths,
        height=max(values),
        width=1.025,
        color=colours,
        align='edge',
        clip_path=polygon)
    axes.plot(wavelengths, values, color='black')

    settings = {
        'title': '{0} - {1}'.format(spd.strict_name, cmfs.strict_name),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Power Distribution',
        'limits': (x_min, x_max, y_min, y_max),
        'x_tighten': True,
        'y_tighten': True
    }
    settings.update(kwargs)

    return render(**settings)


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

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

    canvas(**kwargs)

    cmfs = get_cmfs(cmfs)

    illuminant = ILLUMINANTS_SPDS[DEFAULT_PLOTTING_COLOURSPACE.illuminant]

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

            pylab.plot(wavelengths, values, color=RGB, label=spd.strict_name)
        else:
            pylab.plot(wavelengths, values, label=spd.strict_name)

    settings = {
        'x_label':
            'Wavelength $\\lambda$ (nm)',
        'y_label':
            'Spectral Power Distribution',
        'x_tighten':
            True,
        'y_tighten':
            True,
        'legend':
            True,
        'legend_location':
            'upper left',
        'limits': (min(x_limit_min), max(x_limit_max), min(y_limit_min),
                   max(y_limit_max) + max(y_limit_max) * 0.05)
    }
    settings.update(kwargs)

    return render(**settings)


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> single_cmfs_plot('CIE 1931 2 Degree Standard Observer')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_CMFS_Plot.png
        :align: center
        :alt: single_cmfs_plot
    """

    cmfs = get_cmfs(cmfs)
    settings = {
        'title': '{0} - Colour Matching Functions'.format(cmfs.strict_name)
    }
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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> cmfs = ('CIE 1931 2 Degree Standard Observer',
    ...         'CIE 1964 10 Degree Standard Observer')
    >>> multi_cmfs_plot(cmfs)  # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_CMFS_Plot.png
        :align: center
        :alt: multi_cmfs_plot
    """

    canvas(**kwargs)

    if cmfs is None:
        cmfs = ('CIE 1931 2 Degree Standard Observer',
                'CIE 1964 10 Degree Standard Observer')

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for i, rgb in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        for j, cmfs_i in enumerate(cmfs):
            cmfs_i = get_cmfs(cmfs_i)

            rgb = [reduce(lambda y, _: y * 0.5, range(j), x) for x in rgb]
            values = cmfs_i.values[:, i]

            shape = cmfs_i.shape
            x_limit_min.append(shape.start)
            x_limit_max.append(shape.end)
            y_limit_min.append(min(values))
            y_limit_max.append(max(values))

            pylab.plot(
                cmfs_i.wavelengths,
                values,
                color=rgb,
                label=u'{0} - {1}'.format(cmfs_i.strict_labels[i],
                                          cmfs_i.strict_name))

    settings = {
        'title':
            '{0} - Colour Matching Functions'
            .format(', '.join([get_cmfs(c).strict_name for c in cmfs])),
        'x_label':
            'Wavelength $\\lambda$ (nm)',
        'y_label':
            'Tristimulus Values',
        'x_tighten':
            True,
        'y_tighten':
            True,
        'legend':
            True,
        'legend_location':
            'upper right',
        'grid':
            True,
        'y_axis_line':
            True,
        'limits': (min(x_limit_min), max(x_limit_max), min(y_limit_min),
                   max(y_limit_max) + max(y_limit_max) * 0.05)
    }
    settings.update(kwargs)

    return render(**settings)


def single_illuminant_relative_spd_plot(
        illuminant='A', cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.
    out_of_gamut_clipping : bool, optional
        {:func:`colour.plotting.single_spd_plot`},
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother.

    Returns
    -------
    Figure
        Current figure or None.

    References
    ----------
    -   :cite:`Spiker2015a`

    Examples
    --------
    >>> single_illuminant_relative_spd_plot('A')  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_Illuminant_Relative_SPD_Plot.png
        :align: center
        :alt: single_illuminant_relative_spd_plot
    """

    cmfs = get_cmfs(cmfs)
    title = 'Illuminant {0} - {1}'.format(illuminant, cmfs.strict_name)

    illuminant = get_illuminant(illuminant)

    settings = {'title': title, 'y_label': 'Relative Power'}
    settings.update(kwargs)

    return single_spd_plot(illuminant, **settings)


def multi_illuminant_relative_spd_plot(illuminants=None, **kwargs):
    """
    Plots given illuminants relative spectral power distributions.

    Parameters
    ----------
    illuminants : array_like, optional
        Factory illuminants to plot.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.
    use_spds_colours : bool, optional
        {:func:`colour.plotting.multi_spd_plot`}
        Whether to use spectral power distributions colours.
    normalise_spds_colours : bool
        {:func:`colour.plotting.multi_spd_plot`}
        Whether to normalise spectral power distributions colours.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> multi_illuminant_relative_spd_plot(['A', 'B', 'C'])  # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_Illuminant_Relative_SPD_Plot.png
        :align: center
        :alt: multi_illuminant_relative_spd_plot
    """

    if illuminants is None:
        illuminants = ('A', 'B', 'C')

    spds = []
    for illuminant in illuminants:
        spds.append(get_illuminant(illuminant))

    settings = {
        'title':
            '{0} - Illuminants Relative Spectral Power Distribution'.format(
                ', '.join([spd.strict_name for spd in spds])),
        'y_label':
            'Relative Power'
    }
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
        gray background, less saturated and smoother.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

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

    cmfs, name = get_cmfs(cmfs), cmfs

    x_min, x_max = min(cmfs.wavelengths), max(cmfs.wavelengths)
    y_min, y_max = 0, 1

    settings = {
        'standalone': False,
        'y_label': None,
        'y_ticker': False,
        'limits': (x_min, x_max, y_min, y_max),
    }

    single_spd_plot(
        ones_spd(cmfs.shape),
        cmfs=name,
        out_of_gamut_clipping=out_of_gamut_clipping,
        **settings)

    settings = {
        'title': 'The Visible Spectrum - {0}'.format(cmfs.strict_name),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'standalone': True
    }
    settings.update(kwargs)

    return render(**settings)


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

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
    >>> multi_lightness_function_plot(['CIE 1976', 'Wyszecki 1963'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_Lightness_Function_Plot.png
        :align: center
        :alt: multi_lightness_function_plot
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    if functions is None:
        functions = ('CIE 1976', 'Wyszecki 1963')

    samples = np.linspace(0, 100, 1000)
    for function in functions:
        function, name = LIGHTNESS_METHODS.get(function), function
        if function is None:
            raise KeyError(('"{0}" "Lightness" function not found in factory '
                            '"Lightness" functions: "{1}".').format(
                                name, sorted(LIGHTNESS_METHODS.keys())))
        pylab.plot(samples, function(samples), label='{0}'.format(name))

    settings.update({
        'title': '{0} - Lightness Functions'.format(', '.join(functions)),
        'x_label': 'Relative Luminance Y',
        'y_label': 'Lightness',
        'x_tighten': True,
        'legend': True,
        'legend_location': 'upper left',
        'grid': True,
        'limits': (0, 100, 0, 100),
        'aspect': 'equal'
    })
    settings.update(kwargs)

    return render(**settings)


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> blackbody_spectral_radiance_plot(3500, blackbody='VY Canis Major')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Blackbody_Spectral_Radiance_Plot.png
        :align: center
        :alt: blackbody_spectral_radiance_plot
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
        'standalone': False
    }
    settings.update(kwargs)

    single_spd_plot(spd, cmfs.name, **settings)

    with domain_range_scale('1'):
        XYZ = spectral_to_XYZ(spd, cmfs)

    RGB = normalise_maximum(XYZ_to_plotting_colourspace(XYZ))

    matplotlib.pyplot.subplot(212)

    settings = {
        'title': '{0} - Colour'.format(blackbody),
        'x_label': '{0}K'.format(temperature),
        'y_label': '',
        'aspect': None,
        'standalone': False
    }

    single_colour_swatch_plot(ColourSwatch(name='', RGB=RGB), **settings)

    settings = {'standalone': True}
    settings.update(kwargs)

    return render(**settings)


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
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> blackbody_colours_plot(SpectralShape(150, 12500, 50))  # doctest: +SKIP

    .. image:: ../_static/Plotting_Blackbody_Colours_Plot.png
        :align: center
        :alt: blackbody_colours_plot
    """

    axes = canvas(**kwargs).gca()

    cmfs = get_cmfs(cmfs)

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

    axes.bar(
        x=temperatures,
        height=1,
        width=shape.interval + (0.025 * shape.interval),
        color=colours,
        align='edge')

    settings = {
        'title': 'Blackbody Colours',
        'x_label': 'Temperature K',
        'y_label': None,
        'limits': (x_min, x_max, y_min, y_max),
        'x_tighten': True,
        'y_tighten': True,
        'y_ticker': False
    }
    settings.update(kwargs)

    return render(**settings)
