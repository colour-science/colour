# -*- coding: utf-8 -*-
"""
Colorimetry Plotting
====================

Defines the colorimetry plotting objects:

-   :func:`colour.plotting.plot_single_sd`
-   :func:`colour.plotting.plot_multi_sds`
-   :func:`colour.plotting.plot_single_cmfs`
-   :func:`colour.plotting.plot_multi_cmfs`
-   :func:`colour.plotting.plot_single_illuminant_sd`
-   :func:`colour.plotting.plot_multi_illuminant_sds`
-   :func:`colour.plotting.plot_visible_spectrum`
-   :func:`colour.plotting.plot_single_lightness_function`
-   :func:`colour.plotting.plot_multi_lightness_functions`
-   :func:`colour.plotting.plot_single_luminance_function`
-   :func:`colour.plotting.plot_multi_luminance_functions`
-   :func:`colour.plotting.plot_blackbody_spectral_radiance`
-   :func:`colour.plotting.plot_blackbody_colours`

References
----------
-   :cite:`Spiker2015a` : Borer, T. (2017). Private Discussion with Mansencal,
    T. and Shaw, N.
"""

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from matplotlib.patches import Polygon

from colour.algebra import LinearInterpolator, normalise_maximum
from colour.colorimetry import (
    CCS_ILLUMINANTS,
    SDS_ILLUMINANTS,
    LIGHTNESS_METHODS,
    LUMINANCE_METHODS,
    SpectralShape,
    sd_blackbody,
    sd_ones,
    sd_to_XYZ,
    sds_and_msds_to_sds,
    wavelength_to_XYZ,
)
from colour.plotting import (
    ColourSwatch,
    CONSTANTS_COLOUR_STYLE,
    XYZ_to_plotting_colourspace,
    artist,
    filter_passthrough,
    filter_cmfs,
    filter_illuminants,
    override_style,
    render,
    plot_single_colour_swatch,
    plot_multi_functions,
    update_settings_collection,
)
from colour.utilities import (
    domain_range_scale,
    first_item,
    ones,
    tstack,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'plot_single_sd',
    'plot_multi_sds',
    'plot_single_cmfs',
    'plot_multi_cmfs',
    'plot_single_illuminant_sd',
    'plot_multi_illuminant_sds',
    'plot_visible_spectrum',
    'plot_single_lightness_function',
    'plot_multi_lightness_functions',
    'plot_single_luminance_function',
    'plot_multi_luminance_functions',
    'plot_blackbody_spectral_radiance',
    'plot_blackbody_colours',
]


@override_style()
def plot_single_sd(sd,
                   cmfs='CIE 1931 2 Degree Standard Observer',
                   out_of_gamut_clipping=True,
                   modulate_colours_with_sd_amplitude=False,
                   equalize_sd_amplitude=False,
                   **kwargs):
    """
    Plots given spectral distribution.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution to plot.
    cmfs : str or LMS_ConeFundamentals or \
RGB_ColourMatchingFunctions or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    out_of_gamut_clipping : bool, optional
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother.
    modulate_colours_with_sd_amplitude : bool, optional
        Whether to modulate the colours with the spectral distribution
        amplitude.
    equalize_sd_amplitude : bool, optional
        Whether to equalize the spectral distribution amplitude.
        Equalization occurs after the colours modulation thus setting both
        arguments to *True* will generate a spectrum strip where each
        wavelength colour is modulated by the spectral distribution amplitude.
        The usual 5% margin above the spectral distribution is also omitted.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    References
    ----------
    :cite:`Spiker2015a`

    Examples
    --------
    >>> from colour import SpectralDistribution
    >>> data = {
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360
    ... }
    >>> sd = SpectralDistribution(data, name='Custom')
    >>> plot_single_sd(sd)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_SD.png
        :align: center
        :alt: plot_single_sd
    """

    _figure, axes = artist(**kwargs)

    cmfs = first_item(filter_cmfs(cmfs).values())

    sd = sd.copy()
    sd.interpolator = LinearInterpolator
    wavelengths = cmfs.wavelengths[np.logical_and(
        cmfs.wavelengths >= max(min(cmfs.wavelengths), min(sd.wavelengths)),
        cmfs.wavelengths <= min(max(cmfs.wavelengths), max(sd.wavelengths)),
    )]
    values = sd[wavelengths]

    RGB = XYZ_to_plotting_colourspace(
        wavelength_to_XYZ(wavelengths, cmfs),
        CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E'],
        apply_cctf_encoding=False)

    if not out_of_gamut_clipping:
        RGB += np.abs(np.min(RGB))

    RGB = normalise_maximum(RGB)

    if modulate_colours_with_sd_amplitude:
        RGB *= (values / np.max(values))[..., np.newaxis]

    RGB = CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(RGB)

    if equalize_sd_amplitude:
        values = ones(values.shape)

    margin = 0 if equalize_sd_amplitude else 0.05

    x_min, x_max = min(wavelengths), max(wavelengths)
    y_min, y_max = 0, max(values) + max(values) * margin

    polygon = Polygon(
        np.vstack([
            (x_min, 0),
            tstack([wavelengths, values]),
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
        color=RGB,
        align='edge',
        clip_path=polygon)

    axes.plot(wavelengths, values, color=CONSTANTS_COLOUR_STYLE.colour.dark)

    settings = {
        'axes': axes,
        'bounding_box': (x_min, x_max, y_min, y_max),
        'title': '{0} - {1}'.format(sd.strict_name, cmfs.strict_name),
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Distribution',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_multi_sds(sds, plot_kwargs=None, **kwargs):
    """
    Plots given spectral distributions.

    Parameters
    ----------
    sds : array_like or MultiSpectralDistributions
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single
        :class:`colour.MultiSpectralDistributions` class instance, a list
        of :class:`colour.MultiSpectralDistributions` class instances or a
        list of :class:`colour.SpectralDistribution` class instances.
    plot_kwargs : dict or array_like, optional
        Keyword arguments for the :func:`plt.plot` definition, used to control
        the style of the plotted spectral distributions. ``plot_kwargs`` can be
        either a single dictionary applied to all the plotted spectral
        distributions with same settings or a sequence of dictionaries with
        different settings for each plotted spectral distributions.
        The following special keyword arguments can also be used:

        -   *illuminant* : str or :class:`colour.SpectralDistribution`, the
            illuminant used to compute the spectral distributions colours. The
            default is the illuminant associated with the whitepoint of the
            default plotting colourspace. ``illuminant`` can be of any type or
            form supported by the :func:`colour.plotting.filter_cmfs`
            definition.
        -   *cmfs* : str, the standard observer colour matching functions
            used for computing the spectral distributions colours. ``cmfs`` can
            be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   *normalise_sd_colours* : bool, whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   *use_sd_colours* : bool, whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the :func:`plt.plot`
            definition ``color`` argument with pre-computed values. The default
            is *True*.

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
    >>> from colour import SpectralDistribution
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
    >>> sd_1 = SpectralDistribution(data_1, name='Custom 1')
    >>> sd_2 = SpectralDistribution(data_2, name='Custom 2')
    >>> plot_kwargs = [
    ...     {'use_sd_colours': True},
    ...     {'use_sd_colours': True, 'linestyle': 'dashed'},
    ... ]
    >>> plot_multi_sds([sd_1, sd_2], plot_kwargs=plot_kwargs)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_SDS.png
        :align: center
        :alt: plot_multi_sds
    """

    _figure, axes = artist(**kwargs)

    sds = sds_and_msds_to_sds(sds)

    plot_settings_collection = [{
        'label':
            '{0}'.format(sd.strict_name),
        'cmfs':
            'CIE 1931 2 Degree Standard Observer',
        'illuminant':
            SDS_ILLUMINANTS[
                CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint_name
            ],
        'use_sd_colours':
            False,
        'normalise_sd_colours':
            False,
    } for sd in sds]

    if plot_kwargs is not None:
        update_settings_collection(plot_settings_collection, plot_kwargs,
                                   len(sds))

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for i, sd in enumerate(sds):
        plot_settings = plot_settings_collection[i]

        cmfs = first_item(filter_cmfs(plot_settings.pop('cmfs')).values())
        illuminant = first_item(
            filter_illuminants(plot_settings.pop('illuminant')).values())
        normalise_sd_colours = plot_settings.pop('normalise_sd_colours')
        use_sd_colours = plot_settings.pop('use_sd_colours')

        wavelengths, values = sd.wavelengths, sd.values

        shape = sd.shape
        x_limit_min.append(shape.start)
        x_limit_max.append(shape.end)
        y_limit_min.append(min(values))
        y_limit_max.append(max(values))

        if use_sd_colours:
            with domain_range_scale('1'):
                XYZ = sd_to_XYZ(sd, cmfs, illuminant)

            if normalise_sd_colours:
                XYZ /= XYZ[..., 1]

            plot_settings['color'] = np.clip(
                XYZ_to_plotting_colourspace(XYZ), 0, 1)

        axes.plot(wavelengths, values, **plot_settings)

    bounding_box = (min(x_limit_min), max(x_limit_max), min(y_limit_min),
                    max(y_limit_max) + max(y_limit_max) * 0.05)
    settings = {
        'axes': axes,
        'bounding_box': bounding_box,
        'legend': True,
        'x_label': 'Wavelength $\\lambda$ (nm)',
        'y_label': 'Spectral Distribution',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_cmfs(cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
    """
    Plots given colour matching functions.

    Parameters
    ----------
    cmfs : str or LMS_ConeFundamentals or \
RGB_ColourMatchingFunctions or XYZ_ColourMatchingFunctions, optional
        Colour matching functions to plot. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_cmfs`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_single_cmfs('CIE 1931 2 Degree Standard Observer')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_CMFS.png
        :align: center
        :alt: plot_single_cmfs
    """

    cmfs = first_item(filter_cmfs(cmfs).values())
    settings = {
        'title': '{0} - Colour Matching Functions'.format(cmfs.strict_name)
    }
    settings.update(kwargs)

    return plot_multi_cmfs((cmfs, ), **settings)


@override_style()
def plot_multi_cmfs(cmfs, **kwargs):
    """
    Plots given colour matching functions.

    Parameters
    ----------
    cmfs : str or LMS_ConeFundamentals or \
RGB_ColourMatchingFunctions or XYZ_ColourMatchingFunctions or array_like
        Colour matching functions to plot. ``cmfs`` elements can be of any
        type or form supported by the :func:`colour.plotting.filter_cmfs`
        definition.

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
    >>> cmfs = [
    ...     'CIE 1931 2 Degree Standard Observer',
    ...     'CIE 1964 10 Degree Standard Observer',
    ... ]
    >>> plot_multi_cmfs(cmfs)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_CMFS.png
        :align: center
        :alt: plot_multi_cmfs
    """

    cmfs = filter_cmfs(cmfs).values()

    _figure, axes = artist(**kwargs)

    axes.axhline(color=CONSTANTS_COLOUR_STYLE.colour.dark, linestyle='--')

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
                label='{0} - {1}'.format(cmfs_i.strict_labels[j],
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
def plot_single_illuminant_sd(illuminant,
                              cmfs='CIE 1931 2 Degree Standard Observer',
                              **kwargs):
    """
    Plots given single illuminant spectral distribution.

    Parameters
    ----------
    illuminant : str or SpectralDistribution, optional
        Illuminant to plot. ``illuminant`` can be of any type or form supported
        by the :func:`colour.plotting.filter_illuminants` definition.
    cmfs : str or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.

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

    References
    ----------
    :cite:`Spiker2015a`

    Examples
    --------
    >>> plot_single_illuminant_sd('A')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_Illuminant_SD.png
        :align: center
        :alt: plot_single_illuminant_sd
    """

    cmfs = first_item(filter_cmfs(cmfs).values())
    title = 'Illuminant {0} - {1}'.format(illuminant, cmfs.strict_name)

    illuminant = first_item(filter_illuminants(illuminant).values())

    settings = {'title': title, 'y_label': 'Relative Power'}
    settings.update(kwargs)

    return plot_single_sd(illuminant, **settings)


@override_style()
def plot_multi_illuminant_sds(illuminants, **kwargs):
    """
    Plots given illuminants spectral distributions.

    Parameters
    ----------
    illuminants : str or SpectralDistribution or array_like
        Illuminants to plot. ``illuminants`` elements can be of any type or
        form supported by the :func:`colour.plotting.filter_illuminants`
        definition.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_sds`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_illuminant_sds(['A', 'B', 'C'])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_Illuminant_SDS.png
        :align: center
        :alt: plot_multi_illuminant_sds
    """

    if 'plot_kwargs' not in kwargs:
        kwargs['plot_kwargs'] = {}

    SD_E = SDS_ILLUMINANTS['E']
    if isinstance(kwargs['plot_kwargs'], dict):
        kwargs['plot_kwargs']['illuminant'] = SD_E
    else:
        for i in range(len(kwargs['plot_kwargs'])):
            kwargs['plot_kwargs'][i]['illuminant'] = SD_E

    illuminants = filter_illuminants(illuminants).values()

    title = '{0} - Illuminants Spectral Distributions'.format(', '.join(
        [illuminant.strict_name for illuminant in illuminants]))

    settings = {'title': title, 'y_label': 'Relative Power'}
    settings.update(kwargs)

    return plot_multi_sds(illuminants, **settings)


@override_style(**{
    'ytick.left': False,
    'ytick.labelleft': False,
})
def plot_visible_spectrum(cmfs='CIE 1931 2 Degree Standard Observer',
                          out_of_gamut_clipping=True,
                          **kwargs):
    """
    Plots the visible colours spectrum using given standard observer *CIE XYZ*
    colour matching functions.

    Parameters
    ----------
    cmfs : str or LMS_ConeFundamentals or \
RGB_ColourMatchingFunctions or XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    out_of_gamut_clipping : bool, optional
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_single_sd`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    References
    ----------
    :cite:`Spiker2015a`

    Examples
    --------
    >>> plot_visible_spectrum()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Visible_Spectrum.png
        :align: center
        :alt: plot_visible_spectrum
    """

    cmfs = first_item(filter_cmfs(cmfs).values())

    bounding_box = (min(cmfs.wavelengths), max(cmfs.wavelengths), 0, 1)

    settings = {'bounding_box': bounding_box, 'y_label': None}
    settings.update(kwargs)
    settings['standalone'] = False

    _figure, axes = plot_single_sd(
        sd_ones(cmfs.shape),
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
def plot_single_lightness_function(function, **kwargs):
    """
    Plots given *Lightness* function.

    Parameters
    ----------
    function : str or object
        *Lightness* function to plot. ``function`` can be of any type or form
        supported by the :func:`colour.plotting.filter_passthrough` definition.

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
    >>> plot_single_lightness_function('CIE 1976')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_Lightness_Function.png
        :align: center
        :alt: plot_single_lightness_function
    """

    settings = {'title': '{0} - Lightness Function'.format(function)}
    settings.update(kwargs)

    return plot_multi_lightness_functions((function, ), **settings)


@override_style()
def plot_multi_lightness_functions(functions, **kwargs):
    """
    Plots given *Lightness* functions.

    Parameters
    ----------
    functions : str or object or array_like
        *Lightness* functions to plot. ``functions`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.filter_passthrough` definition.

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
    >>> plot_multi_lightness_functions(['CIE 1976', 'Wyszecki 1963'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_Lightness_Functions.png
        :align: center
        :alt: plot_multi_lightness_functions
    """

    functions = filter_passthrough(LIGHTNESS_METHODS, functions)

    settings = {
        'bounding_box': (0, 1, 0, 1),
        'legend': True,
        'title': '{0} - Lightness Functions'.format(', '.join(functions)),
        'x_label': 'Normalised Relative Luminance Y',
        'y_label': 'Normalised Lightness',
    }
    settings.update(kwargs)

    with domain_range_scale('1'):
        return plot_multi_functions(functions, **settings)


@override_style()
def plot_single_luminance_function(function, **kwargs):
    """
    Plots given *Luminance* function.

    Parameters
    ----------
    function : str or object, optional
        *Luminance* function to plot.

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
    >>> plot_single_luminance_function('CIE 1976')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_Luminance_Function.png
        :align: center
        :alt: plot_single_luminance_function
    """

    settings = {'title': '{0} - Luminance Function'.format(function)}
    settings.update(kwargs)

    return plot_multi_luminance_functions((function, ), **settings)


@override_style()
def plot_multi_luminance_functions(functions, **kwargs):
    """
    Plots given *Luminance* functions.

    Parameters
    ----------
    functions : str or object or array_like
        *Luminance* functions to plot. ``functions`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.filter_passthrough` definition.

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
    >>> plot_multi_luminance_functions(['CIE 1976', 'Newhall 1943'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_Luminance_Functions.png
        :align: center
        :alt: plot_multi_luminance_functions
    """

    functions = filter_passthrough(LUMINANCE_METHODS, functions)

    settings = {
        'bounding_box': (0, 1, 0, 1),
        'legend': True,
        'title': '{0} - Luminance Functions'.format(', '.join(functions)),
        'x_label': 'Normalised Munsell Value / Lightness',
        'y_label': 'Normalised Relative Luminance Y',
    }
    settings.update(kwargs)

    with domain_range_scale('1'):
        return plot_multi_functions(functions, **settings)


@override_style()
def plot_blackbody_spectral_radiance(
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
    cmfs : str, optional
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    blackbody : str, optional
        Blackbody name.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_single_sd`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_blackbody_spectral_radiance(3500, blackbody='VY Canis Major')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 2 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Blackbody_Spectral_Radiance.png
        :align: center
        :alt: plot_blackbody_spectral_radiance
    """

    figure = plt.figure()

    figure.subplots_adjust(hspace=CONSTANTS_COLOUR_STYLE.geometry.short / 2)

    cmfs = first_item(filter_cmfs(cmfs).values())

    sd = sd_blackbody(temperature, cmfs.shape)

    axes = figure.add_subplot(211)
    settings = {
        'axes': axes,
        'title': '{0} - Spectral Radiance'.format(blackbody),
        'y_label': 'W / (sr m$^2$) / m',
    }
    settings.update(kwargs)
    settings['standalone'] = False

    plot_single_sd(sd, cmfs.name, **settings)

    axes = figure.add_subplot(212)

    with domain_range_scale('1'):
        XYZ = sd_to_XYZ(sd, cmfs)

    RGB = normalise_maximum(XYZ_to_plotting_colourspace(XYZ))

    settings = {
        'axes': axes,
        'aspect': None,
        'title': '{0} - Colour'.format(blackbody),
        'x_label': '{0}K'.format(temperature),
        'y_label': '',
        'x_ticker': False,
        'y_ticker': False,
    }
    settings.update(kwargs)
    settings['standalone'] = False

    figure, axes = plot_single_colour_swatch(
        ColourSwatch(name='', RGB=RGB), **settings)

    settings = {'axes': axes, 'standalone': True}
    settings.update(kwargs)

    return render(**settings)


@override_style(**{
    'ytick.left': False,
    'ytick.labelleft': False,
})
def plot_blackbody_colours(
        shape=SpectralShape(150, 12500, 50),
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots blackbody colours.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape to use as plot boundaries.
    cmfs : str, optional
        Standard observer colour matching functions used for computing the
        blackbody colours. ``cmfs`` can be of any type or form supported by the
        :func:`colour.plotting.filter_cmfs` definition.

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
    >>> plot_blackbody_colours(SpectralShape(150, 12500, 50))
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Blackbody_Colours.png
        :align: center
        :alt: plot_blackbody_colours
    """

    _figure, axes = artist(**kwargs)

    cmfs = first_item(filter_cmfs(cmfs).values())

    RGB = []
    temperatures = []

    for temperature in shape:
        sd = sd_blackbody(temperature, cmfs.shape)

        with domain_range_scale('1'):
            XYZ = sd_to_XYZ(sd, cmfs)

        RGB.append(normalise_maximum(XYZ_to_plotting_colourspace(XYZ)))
        temperatures.append(temperature)

    x_min, x_max = min(temperatures), max(temperatures)
    y_min, y_max = 0, 1

    padding = 0.1
    axes.bar(
        x=np.array(temperatures) - padding,
        height=1,
        width=shape.interval + (padding * shape.interval),
        color=RGB,
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
