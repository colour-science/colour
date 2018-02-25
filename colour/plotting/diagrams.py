# -*- coding: utf-8 -*-
"""
CIE Chromaticity Diagrams Plotting
==================================

Defines the *CIE* chromaticity diagrams plotting objects:

-   :func:`colour.plotting.chromaticity_diagram_plot_CIE1931`
-   :func:`colour.plotting.chromaticity_diagram_plot_CIE1960UCS`
-   :func:`colour.plotting.chromaticity_diagram_plot_CIE1976UCS`
-   :func:`colour.plotting.spds_chromaticity_diagram_plot_CIE1931`
-   :func:`colour.plotting.spds_chromaticity_diagram_plot_CIE1960UCS`
-   :func:`colour.plotting.spds_chromaticity_diagram_plot_CIE1976UCS`
"""

from __future__ import division

import bisect
import numpy as np
import pylab
from matplotlib.patches import Polygon

from colour.algebra import normalise_vector
from colour.colorimetry import spectral_to_XYZ
from colour.models import (Luv_to_uv, Luv_uv_to_xy, UCS_to_uv, UCS_uv_to_xy,
                           XYZ_to_Luv, XYZ_to_UCS, XYZ_to_sRGB, XYZ_to_xy,
                           xy_to_XYZ)
from colour.plotting import (DEFAULT_FIGURE_WIDTH, DEFAULT_PLOTTING_ILLUMINANT,
                             canvas, get_cmfs, render)
from colour.utilities import normalise_maximum, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'chromaticity_diagram_colours', 'chromaticity_diagram_plot',
    'chromaticity_diagram_plot_CIE1931',
    'chromaticity_diagram_plot_CIE1960UCS',
    'chromaticity_diagram_plot_CIE1976UCS', 'spds_chromaticity_diagram_plot',
    'spds_chromaticity_diagram_plot_CIE1931',
    'spds_chromaticity_diagram_plot_CIE1960UCS',
    'spds_chromaticity_diagram_plot_CIE1976UCS'
]


def chromaticity_diagram_colours(samples=192,
                                 cmfs='CIE 1931 2 Degree Standard Observer',
                                 method='CIE 1931',
                                 **kwargs):
    """
    Plots the *Chromaticity Diagram* colours accordingly to given method.

    Parameters
    ----------
    samples : numeric, optional
        Samples count on one axis.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
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
    >>> chromaticity_diagram_colours()  # doctest: +SKIP
    """

    axes = canvas(**kwargs).gca()

    cmfs = get_cmfs(cmfs)

    illuminant = DEFAULT_PLOTTING_ILLUMINANT

    ii, jj = np.meshgrid(
        np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    ij = tstack((ii, jj))

    method = method.upper()
    if method == 'CIE 1931':
        XYZ = xy_to_XYZ(ij)
        spectral_locus = XYZ_to_xy(cmfs.values, illuminant)
    elif method == 'CIE 1960 UCS':
        XYZ = xy_to_XYZ(UCS_uv_to_xy(ij))
        spectral_locus = UCS_to_uv(XYZ_to_UCS(cmfs.values))
    elif method == 'CIE 1976 UCS':
        XYZ = xy_to_XYZ(Luv_uv_to_xy(ij))
        spectral_locus = Luv_to_uv(
            XYZ_to_Luv(cmfs.values, illuminant), illuminant)
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
                method))

    RGB = normalise_maximum(XYZ_to_sRGB(XYZ, illuminant), axis=-1)

    polygon = Polygon(spectral_locus, facecolor='none', edgecolor='none')
    axes.add_patch(polygon)
    # Preventing bounding box related issues as per
    # https://github.com/matplotlib/matplotlib/issues/10529
    image = pylab.imshow(
        RGB, interpolation='bilinear', extent=(0, 1, 0, 1), clip_path=None)
    image.set_clip_path(polygon)

    return render(**kwargs)


def chromaticity_diagram_plot(cmfs='CIE 1931 2 Degree Standard Observer',
                              show_diagram_colours=True,
                              wavelength_labels=None,
                              method='CIE 1931',
                              **kwargs):
    """
    Plots the *Chromaticity Diagram* accordingly to given method.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    show_diagram_colours : bool, optional
        Whether to display the chromaticity diagram background colours.
    wavelength_labels : array_like, optional
        Array of wavelength labels used to customise which labels will be drawn
        around the spectral locus. Passing an empty array will result in no
        wavelength labels being drawn.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_colours,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> chromaticity_diagram_plot()  # doctest: +SKIP
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    illuminant = DEFAULT_PLOTTING_ILLUMINANT

    if show_diagram_colours:
        settings = {'method': method, 'standalone': False}
        settings.update(kwargs)
        chromaticity_diagram_colours(**settings)

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    method = method.upper()
    if method == 'CIE 1931':
        ij = XYZ_to_xy(cmfs.values, illuminant)
        x_label, y_label = 'CIE x', 'CIE y'
        wl_labels = (390, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580,
                     600, 620, 700)
    elif method == 'CIE 1960 UCS':
        ij = UCS_to_uv(XYZ_to_UCS(cmfs.values))
        x_label, y_label = 'CIE u', 'CIE v'
        wl_labels = (420, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530,
                     540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 645,
                     680)
    elif method == 'CIE 1976 UCS':
        ij = Luv_to_uv(XYZ_to_Luv(cmfs.values, illuminant), illuminant)
        x_label, y_label = 'CIE u\'', 'CIE v\'',
        wl_labels = (420, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530,
                     540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 645,
                     680)
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
                method))

    wavelengths_chromaticity_coordinates = dict(tuple(zip(wavelengths, ij)))

    pylab.plot(ij[..., 0], ij[..., 1], color='black', linewidth=1)
    pylab.plot(
        (ij[-1][0], ij[0][0]), (ij[-1][1], ij[0][1]),
        color='black',
        linewidth=1)

    if wavelength_labels is not None:
        wl_labels = wavelength_labels

    for wl_label in wl_labels:
        i, j = wavelengths_chromaticity_coordinates[wl_label]
        pylab.plot(i, j, 'o', color='black', linewidth=1)

        index = bisect.bisect(wavelengths, wl_label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else wavelengths[-1])

        dx = (wavelengths_chromaticity_coordinates[right][0] -
              wavelengths_chromaticity_coordinates[left][0])
        dy = (wavelengths_chromaticity_coordinates[right][1] -
              wavelengths_chromaticity_coordinates[left][1])

        ij = np.array([i, j])
        direction = np.array([-dy, dx])

        normal = (np.array([-dy, dx]) if np.dot(
            normalise_vector(ij - equal_energy), normalise_vector(direction)) >
                  0 else np.array([dy, -dx]))
        normal = normalise_vector(normal)
        normal /= 25

        pylab.plot(
            (i, i + normal[0] * 0.75), (j, j + normal[1] * 0.75),
            color='black',
            linewidth=1.5)
        pylab.text(
            i + normal[0],
            j + normal[1],
            wl_label,
            color='black',
            clip_on=True,
            ha='left' if normal[0] >= 0 else 'right',
            va='center',
            fontdict={'size': 'small'})

    ticks = np.arange(-10, 10, 0.1)

    pylab.xticks(ticks)
    pylab.yticks(ticks)

    settings.update({
        'standalone':
            True,
        'title':
            '{0} Chromaticity Diagram - {1}'.format(method, cmfs.strict_name),
        'x_label':
            x_label,
        'y_label':
            y_label,
        'grid':
            True,
        'bounding_box': (0, 1, 0, 1)
    })
    settings.update(kwargs)

    return render(**settings)


def chromaticity_diagram_plot_CIE1931(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        wavelength_labels=None,
        **kwargs):
    """
    Plots the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    show_diagram_colours : bool, optional
        Whether to display the chromaticity diagram background colours.
    wavelength_labels : array_like, optional
        Array of wavelength labels used to customise which labels will be drawn
        around the spectral locus. Passing an empty array will result in no
        wavelength labels being drawn.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_colours,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1931()  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return chromaticity_diagram_plot(cmfs, show_diagram_colours,
                                     wavelength_labels, **settings)


def chromaticity_diagram_plot_CIE1960UCS(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        wavelength_labels=None,
        **kwargs):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    show_diagram_colours : bool, optional
        Whether to display the chromaticity diagram background colours.
    wavelength_labels : array_like, optional
        Array of wavelength labels used to customise which labels will be drawn
        around the spectral locus. Passing an empty array will result in no
        wavelength labels being drawn.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_colours,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1960UCS()  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return chromaticity_diagram_plot(cmfs, show_diagram_colours,
                                     wavelength_labels, **settings)


def chromaticity_diagram_plot_CIE1976UCS(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        wavelength_labels=None,
        **kwargs):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    show_diagram_colours : bool, optional
        Whether to display the chromaticity diagram background colours.
    wavelength_labels : array_like, optional
        Array of wavelength labels used to customise which labels will be drawn
        around the spectral locus. Passing an empty array will result in no
        wavelength labels being drawn.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_colours,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1976UCS()  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return chromaticity_diagram_plot(cmfs, show_diagram_colours,
                                     wavelength_labels, **settings)


def spds_chromaticity_diagram_plot(
        spds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        annotate=True,
        chromaticity_diagram_callable=chromaticity_diagram_plot,
        method='CIE 1931',
        **kwargs):
    """
    Plots given spectral power distribution chromaticity coordinates into the
    *Chromaticity Diagram* using given method.

    Parameters
    ----------
    spds : array_like, optional
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    annotate : bool
        Should resulting chromaticity coordinates annotated with their
        respective spectral power distribution names.
    chromaticity_diagram_callable : callable, optional
        Callable responsible for drawing the *Chromaticity Diagram*.
    method : unicode, optional
        **{'CIE 1931', 'CIE 1960 UCS', 'CIE 1976 UCS'}**,
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> A = ILLUMINANTS_RELATIVE_SPDS['A']
    >>> D65 = ILLUMINANTS_RELATIVE_SPDS['D65']
    >>> spds_chromaticity_diagram_plot([A, D65])  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': method, 'cmfs': cmfs, 'standalone': False})

    chromaticity_diagram_callable(**settings)

    method = method.upper()
    if method == 'CIE 1931':

        def XYZ_to_ij(XYZ):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return XYZ_to_xy(XYZ)

        limits = (-0.1, 0.9, -0.1, 0.9)
    elif method == 'CIE 1960 UCS':

        def XYZ_to_ij(XYZ):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(XYZ))

        limits = (-0.1, 0.7, -0.2, 0.6)

    elif method == 'CIE 1976 UCS':

        def XYZ_to_ij(XYZ):
            """
            Converts given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return Luv_to_uv(XYZ_to_Luv(XYZ))

        limits = (-0.1, 0.7, -0.1, 0.7)
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}'.format(
                method))

    for spd in spds:
        XYZ = spectral_to_XYZ(spd) / 100
        ij = XYZ_to_ij(XYZ)

        pylab.plot(ij[0], ij[1], 'o', color='white')

        if spd.name is not None and annotate:
            pylab.annotate(
                spd.name,
                xy=ij,
                xytext=(50, 30),
                color='black',
                textcoords='offset points',
                arrowprops=dict(
                    arrowstyle='->', connectionstyle='arc3, rad=0.2'))

    settings.update({
        'x_tighten': True,
        'y_tighten': True,
        'limits': limits,
        'standalone': True
    })
    settings.update(kwargs)

    return render(**settings)


def spds_chromaticity_diagram_plot_CIE1931(
        spds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        annotate=True,
        chromaticity_diagram_callable_CIE1931=(
            chromaticity_diagram_plot_CIE1931),
        **kwargs):
    """
    Plots given spectral power distribution chromaticity coordinates into the
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    spds : array_like, optional
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    annotate : bool
        Should resulting chromaticity coordinates annotated with their
        respective spectral power distribution names.
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> A = ILLUMINANTS_RELATIVE_SPDS['A']
    >>> D65 = ILLUMINANTS_RELATIVE_SPDS['D65']
    >>> spds_chromaticity_diagram_plot_CIE1931([A, D65])  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1931'})

    return spds_chromaticity_diagram_plot(
        spds, cmfs, annotate, chromaticity_diagram_callable_CIE1931,
        **settings)


def spds_chromaticity_diagram_plot_CIE1960UCS(
        spds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        annotate=True,
        chromaticity_diagram_callable_CIE1960UCS=(
            chromaticity_diagram_plot_CIE1960UCS),
        **kwargs):
    """
    Plots given spectral power distribution chromaticity coordinates into the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    spds : array_like, optional
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    annotate : bool
        Should resulting chromaticity coordinates annotated with their
        respective spectral power distribution names.
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> A = ILLUMINANTS_RELATIVE_SPDS['A']
    >>> D65 = ILLUMINANTS_RELATIVE_SPDS['D65']
    >>> spds_chromaticity_diagram_plot_CIE1960UCS([A, D65])  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1960 UCS'})

    return spds_chromaticity_diagram_plot(
        spds, cmfs, annotate, chromaticity_diagram_callable_CIE1960UCS,
        **settings)


def spds_chromaticity_diagram_plot_CIE1976UCS(
        spds,
        cmfs='CIE 1931 2 Degree Standard Observer',
        annotate=True,
        chromaticity_diagram_callable_CIE1976UCS=(
            chromaticity_diagram_plot_CIE1976UCS),
        **kwargs):
    """
    Plots given spectral power distribution chromaticity coordinates into the
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    spds : array_like, optional
        Spectral power distributions to plot.
    cmfs : unicode, optional
        Standard observer colour matching functions used for
        *Chromaticity Diagram* bounds.
    annotate : bool
        Should resulting chromaticity coordinates annotated with their
        respective spectral power distribution names.
    chromaticity_diagram_callable_CIE1976UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.chromaticity_diagram_plot`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> A = ILLUMINANTS_RELATIVE_SPDS['A']
    >>> D65 = ILLUMINANTS_RELATIVE_SPDS['D65']
    >>> spds_chromaticity_diagram_plot_CIE1976UCS([A, D65])  # doctest: +SKIP
    """

    settings = dict(kwargs)
    settings.update({'method': 'CIE 1976 UCS'})

    return spds_chromaticity_diagram_plot(
        spds, cmfs, annotate, chromaticity_diagram_callable_CIE1976UCS,
        **settings)
