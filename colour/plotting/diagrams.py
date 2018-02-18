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
import os

import matplotlib
import matplotlib.image
import matplotlib.pyplot
import numpy as np
import pylab
from scipy.ndimage.filters import convolve
from scipy.spatial import Delaunay

from colour.algebra import normalise_vector
from colour.colorimetry import spectral_to_XYZ
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.models import (Luv_to_uv, Luv_uv_to_xy, UCS_to_uv, UCS_uv_to_xy,
                           XYZ_to_Luv, XYZ_to_UCS, XYZ_to_sRGB, XYZ_to_xy,
                           xy_to_XYZ)
from colour.plotting import (DEFAULT_FIGURE_WIDTH, DEFAULT_PLOTTING_ILLUMINANT,
                             PLOTTING_RESOURCES_DIRECTORY, canvas, get_cmfs,
                             render)
from colour.utilities import normalise_maximum, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'chromaticity_diagram_colours_CIE1931',
    'chromaticity_diagram_plot_CIE1931',
    'chromaticity_diagram_colours_CIE1960UCS',
    'chromaticity_diagram_plot_CIE1960UCS',
    'chromaticity_diagram_colours_CIE1976UCS',
    'chromaticity_diagram_plot_CIE1976UCS',
    'spds_chromaticity_diagram_plot_CIE1931',
    'spds_chromaticity_diagram_plot_CIE1960UCS',
    'spds_chromaticity_diagram_plot_CIE1976UCS'
]


def chromaticity_diagram_colours_CIE1931(
        samples=4096,
        cmfs='CIE 1931 2 Degree Standard Observer',
        antialiasing=True):
    """
    Plots the *CIE 1931 Chromaticity Diagram* colours.

    Parameters
    ----------
    samples : numeric, optional
        Samples count on one axis.
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    antialiasing : bool, optional
        Whether to apply anti-aliasing to the image.

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
    >>> chromaticity_diagram_colours_CIE1931()  # doctest: +SKIP
    """

    cmfs = get_cmfs(cmfs)

    illuminant = DEFAULT_PLOTTING_ILLUMINANT

    triangulation = Delaunay(
        XYZ_to_xy(cmfs.values, illuminant), qhull_options='Qu QJ')
    xx, yy = np.meshgrid(
        np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    xy = tstack((xx, yy))
    mask = (triangulation.find_simplex(xy) < 0).astype(DEFAULT_FLOAT_DTYPE)
    if antialiasing:
        kernel = np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]).astype(DEFAULT_FLOAT_DTYPE)
        kernel /= np.sum(kernel)
        mask = convolve(mask, kernel)

    mask = 1 - mask[:, :, np.newaxis]

    XYZ = xy_to_XYZ(xy)

    RGB = normalise_maximum(XYZ_to_sRGB(XYZ, illuminant), axis=-1)

    return np.dstack([RGB, mask])


def chromaticity_diagram_plot_CIE1931(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        use_cached_diagram_colours=True,
        **kwargs):
    """
    Plots the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    show_diagram_colours : bool, optional
        Whether to display the chromaticity diagram background colours.
    use_cached_diagram_colours : bool, optional
        Whether to used the cached chromaticity diagram background colours
        image.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.diagrams.chromaticity_diagram_colours_CIE1931`,
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

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    illuminant = DEFAULT_PLOTTING_ILLUMINANT

    if show_diagram_colours:
        if use_cached_diagram_colours:
            image = matplotlib.image.imread(
                os.path.join(PLOTTING_RESOURCES_DIRECTORY,
                             'CIE_1931_Chromaticity_Diagram_{0}.png'.format(
                                 cmfs.name.replace(' ', '_'))))
        else:
            image = chromaticity_diagram_colours_CIE1931(
                samples=kwargs.get('samples', 256),
                cmfs=cmfs.name,
                antialiasing=kwargs.get('antialiasing', True))

        pylab.imshow(image, interpolation='bilinear', extent=(0, 1, 0, 1))

    labels = (390, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620,
              700)

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    xy = XYZ_to_xy(cmfs.values, illuminant)

    wavelengths_chromaticity_coordinates = dict(tuple(zip(wavelengths, xy)))

    pylab.plot(xy[..., 0], xy[..., 1], color='black', linewidth=1)
    pylab.plot(
        (xy[-1][0], xy[0][0]), (xy[-1][1], xy[0][1]),
        color='black',
        linewidth=1)

    for label in labels:
        x, y = wavelengths_chromaticity_coordinates[label]
        pylab.plot(x, y, 'o', color='black', linewidth=1)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else wavelengths[-1])

        dx = (wavelengths_chromaticity_coordinates[right][0] -
              wavelengths_chromaticity_coordinates[left][0])
        dy = (wavelengths_chromaticity_coordinates[right][1] -
              wavelengths_chromaticity_coordinates[left][1])

        xy = np.array([x, y])
        direction = np.array([-dy, dx])

        normal = (np.array([-dy, dx]) if np.dot(
            normalise_vector(xy - equal_energy), normalise_vector(direction)) >
                  0 else np.array([dy, -dx]))
        normal = normalise_vector(normal)
        normal /= 25

        pylab.plot(
            (x, x + normal[0] * 0.75), (y, y + normal[1] * 0.75),
            color='black',
            linewidth=1.5)
        pylab.text(
            x + normal[0],
            y + normal[1],
            label,
            color='black',
            clip_on=True,
            ha='left' if normal[0] >= 0 else 'right',
            va='center',
            fontdict={'size': 'small'})

    ticks = np.arange(-10, 10, 0.1)

    pylab.xticks(ticks)
    pylab.yticks(ticks)

    settings.update({
        'title':
            'CIE 1931 Chromaticity Diagram - {0}'.format(cmfs.strict_name),
        'x_label':
            'CIE x',
        'y_label':
            'CIE y',
        'grid':
            True,
        'bounding_box': (0, 1, 0, 1)
    })
    settings.update(kwargs)

    return render(**settings)


def chromaticity_diagram_colours_CIE1960UCS(
        samples=4096,
        cmfs='CIE 1931 2 Degree Standard Observer',
        antialiasing=True):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram* colours.

    Parameters
    ----------
    samples : numeric, optional
        Samples count on one axis.
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    antialiasing : bool, optional
        Whether to apply anti-aliasing to the image.

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
    >>> chromaticity_diagram_colours_CIE1960UCS()  # doctest: +SKIP
    """

    cmfs = get_cmfs(cmfs)

    illuminant = DEFAULT_PLOTTING_ILLUMINANT

    triangulation = Delaunay(
        UCS_to_uv(XYZ_to_UCS(cmfs.values)), qhull_options='Qu QJ')
    xx, yy = np.meshgrid(
        np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    xy = tstack((xx, yy))
    mask = (triangulation.find_simplex(xy) < 0).astype(DEFAULT_FLOAT_DTYPE)
    if antialiasing:
        kernel = np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]).astype(DEFAULT_FLOAT_DTYPE)
        kernel /= np.sum(kernel)
        mask = convolve(mask, kernel)

    mask = 1 - mask[:, :, np.newaxis]

    XYZ = xy_to_XYZ(UCS_uv_to_xy(xy))

    RGB = normalise_maximum(XYZ_to_sRGB(XYZ, illuminant), axis=-1)

    return np.dstack([RGB, mask])


def chromaticity_diagram_plot_CIE1960UCS(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        use_cached_diagram_colours=True,
        **kwargs):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    show_diagram_colours : bool, optional
        Whether to display the chromaticity diagram background colours.
    use_cached_diagram_colours : bool, optional
        Whether to used the cached chromaticity diagram background colours
        image.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.diagrams.\
chromaticity_diagram_colours_CIE1960UCS`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1960UCS()  # doctest: +SKIP
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    if show_diagram_colours:
        if use_cached_diagram_colours:
            image = matplotlib.image.imread(
                os.path.join(PLOTTING_RESOURCES_DIRECTORY,
                             'CIE_1960_UCS_Chromaticity_Diagram_{0}.png'.
                             format(cmfs.name.replace(' ', '_'))))
        else:
            image = chromaticity_diagram_colours_CIE1960UCS(
                samples=kwargs.get('samples', 256),
                cmfs=cmfs.name,
                antialiasing=kwargs.get('antialiasing', True))

        pylab.imshow(image, interpolation='bilinear', extent=(0, 1, 0, 1))

    labels = (420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
              550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 680)

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    uv = UCS_to_uv(XYZ_to_UCS(cmfs.values))

    wavelengths_chromaticity_coordinates = dict(tuple(zip(wavelengths, uv)))

    pylab.plot(uv[..., 0], uv[..., 1], color='black', linewidth=1)
    pylab.plot(
        (uv[-1][0], uv[0][0]), (uv[-1][1], uv[0][1]),
        color='black',
        linewidth=1)

    for label in labels:
        u, v = wavelengths_chromaticity_coordinates[label]
        pylab.plot(u, v, 'o', color='black', linewidth=1)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else wavelengths[-1])

        dx = (wavelengths_chromaticity_coordinates[right][0] -
              wavelengths_chromaticity_coordinates[left][0])
        dy = (wavelengths_chromaticity_coordinates[right][1] -
              wavelengths_chromaticity_coordinates[left][1])

        uv = np.array([u, v])
        direction = np.array([-dy, dx])

        normal = (np.array([-dy, dx]) if np.dot(
            normalise_vector(uv - equal_energy), normalise_vector(direction)) >
                  0 else np.array([dy, -dx]))
        normal = normalise_vector(normal)
        normal /= 25

        pylab.plot(
            (u, u + normal[0] * 0.75), (v, v + normal[1] * 0.75),
            color='black',
            linewidth=1.5)
        pylab.text(
            u + normal[0],
            v + normal[1],
            label,
            color='black',
            clip_on=True,
            ha='left' if normal[0] >= 0 else 'right',
            va='center',
            fontdict={'size': 'small'})

    ticks = np.arange(-10, 10, 0.1)

    pylab.xticks(ticks)
    pylab.yticks(ticks)

    settings.update({
        'title':
            'CIE 1960 UCS Chromaticity Diagram - {0}'.format(cmfs.strict_name),
        'x_label':
            'CIE u',
        'y_label':
            'CIE v',
        'grid':
            True,
        'bounding_box': (0, 1, 0, 1)
    })
    settings.update(kwargs)

    return render(**settings)


def chromaticity_diagram_colours_CIE1976UCS(
        samples=4096,
        cmfs='CIE 1931 2 Degree Standard Observer',
        antialiasing=True):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram* colours.

    Parameters
    ----------
    samples : numeric, optional
        Samples count on one axis.
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    antialiasing : bool, optional
        Whether to apply anti-aliasing to the image.

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
    >>> chromaticity_diagram_colours_CIE1976UCS()  # doctest: +SKIP
    """

    cmfs = get_cmfs(cmfs)

    illuminant = DEFAULT_PLOTTING_ILLUMINANT

    triangulation = Delaunay(
        Luv_to_uv(XYZ_to_Luv(cmfs.values, illuminant), illuminant),
        qhull_options='Qu QJ')
    xx, yy = np.meshgrid(
        np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    xy = tstack((xx, yy))
    mask = (triangulation.find_simplex(xy) < 0).astype(DEFAULT_FLOAT_DTYPE)
    if antialiasing:
        kernel = np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]).astype(DEFAULT_FLOAT_DTYPE)
        kernel /= np.sum(kernel)
        mask = convolve(mask, kernel)

    mask = 1 - mask[:, :, np.newaxis]

    XYZ = xy_to_XYZ(Luv_uv_to_xy(xy))

    RGB = normalise_maximum(XYZ_to_sRGB(XYZ, illuminant), axis=-1)

    return np.dstack([RGB, mask])


def chromaticity_diagram_plot_CIE1976UCS(
        cmfs='CIE 1931 2 Degree Standard Observer',
        show_diagram_colours=True,
        use_cached_diagram_colours=True,
        **kwargs):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    show_diagram_colours : bool, optional
        Whether to display the chromaticity diagram background colours.
    use_cached_diagram_colours : bool, optional
        Whether to used the cached chromaticity diagram background colours
        image.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.diagrams.\
chromaticity_diagram_colours_CIE1976UCS`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1976UCS()  # doctest: +SKIP
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    illuminant = DEFAULT_PLOTTING_ILLUMINANT

    if show_diagram_colours:
        if use_cached_diagram_colours:
            image = matplotlib.image.imread(
                os.path.join(PLOTTING_RESOURCES_DIRECTORY,
                             'CIE_1976_UCS_Chromaticity_Diagram_{0}.png'.
                             format(cmfs.name.replace(' ', '_'))))
        else:
            image = chromaticity_diagram_colours_CIE1976UCS(
                samples=kwargs.get('samples', 256),
                cmfs=cmfs.name,
                antialiasing=kwargs.get('antialiasing', True))

        pylab.imshow(image, interpolation='bilinear', extent=(0, 1, 0, 1))

    labels = (420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
              550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 680)

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    uv = Luv_to_uv(XYZ_to_Luv(cmfs.values, illuminant), illuminant)

    wavelengths_chromaticity_coordinates = dict(zip(wavelengths, uv))

    pylab.plot(uv[..., 0], uv[..., 1], color='black', linewidth=1)
    pylab.plot(
        (uv[-1][0], uv[0][0]), (uv[-1][1], uv[0][1]),
        color='black',
        linewidth=1)

    for label in labels:
        u, v = wavelengths_chromaticity_coordinates[label]
        pylab.plot(u, v, 'o', color='black', linewidth=1)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else wavelengths[-1])

        dx = (wavelengths_chromaticity_coordinates[right][0] -
              wavelengths_chromaticity_coordinates[left][0])
        dy = (wavelengths_chromaticity_coordinates[right][1] -
              wavelengths_chromaticity_coordinates[left][1])

        uv = np.array([u, v])
        direction = np.array([-dy, dx])

        normal = (np.array([-dy, dx]) if np.dot(
            normalise_vector(uv - equal_energy), normalise_vector(direction)) >
                  0 else np.array([dy, -dx]))
        normal = normalise_vector(normal)
        normal /= 25

        pylab.plot(
            (u, u + normal[0] * 0.75), (v, v + normal[1] * 0.75),
            color='black',
            linewidth=1.5)
        pylab.text(
            u + normal[0],
            v + normal[1],
            label,
            color='black',
            clip_on=True,
            ha='left' if normal[0] >= 0 else 'right',
            va='center',
            fontdict={'size': 'small'})

    ticks = np.arange(-10, 10, 0.1)

    pylab.xticks(ticks)
    pylab.yticks(ticks)

    settings.update({
        'title':
            'CIE 1976 UCS Chromaticity Diagram - {0}'.format(cmfs.strict_name),
        'x_label':
            'CIE u\'',
        'y_label':
            'CIE v\'',
        'grid':
            True,
        'bounding_box': (0, 1, 0, 1)
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
        Standard observer colour matching functions used for diagram bounds.
    annotate : bool
        Should resulting chromaticity coordinates annotated with their
        respective spectral power distribution names.
    chromaticity_diagram_callable_CIE1931 : callable, optional
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.
    show_diagram_colours : bool, optional
        {:func:`colour.plotting.chromaticity_diagram_plot_CIE1931`},
        Whether to display the chromaticity diagram background colours.
    use_cached_diagram_colours : bool, optional
        {:func:`colour.plotting.chromaticity_diagram_plot_CIE1931`},
        Whether to used the cached chromaticity diagram background colours
        image.

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

    settings = {}
    settings.update(kwargs)
    settings.update({'cmfs': cmfs, 'standalone': False})

    chromaticity_diagram_callable_CIE1931(**settings)

    for spd in spds:
        XYZ = spectral_to_XYZ(spd) / 100
        xy = XYZ_to_xy(XYZ)

        pylab.plot(xy[0], xy[1], 'o', color='white')

        if spd.name is not None and annotate:
            pylab.annotate(
                spd.name,
                xy=xy,
                xytext=(50, 30),
                color='black',
                textcoords='offset points',
                arrowprops=dict(
                    arrowstyle='->', connectionstyle='arc3, rad=0.2'))

    settings.update({
        'x_tighten': True,
        'y_tighten': True,
        'limits': (-0.1, 0.9, -0.1, 0.9),
        'standalone': True
    })
    settings.update(kwargs)

    return render(**settings)


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
        Standard observer colour matching functions used for diagram bounds.
    annotate : bool
        Should resulting chromaticity coordinates annotated with their
        respective spectral power distribution names.
    chromaticity_diagram_callable_CIE1960UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.
    show_diagram_colours : bool, optional
        {:func:`colour.plotting.chromaticity_diagram_plot_CIE1960UCS`},
        Whether to display the chromaticity diagram background colours.
    use_cached_diagram_colours : bool, optional
        {:func:`colour.plotting.chromaticity_diagram_plot_CIE1960UCS`},
        Whether to used the cached chromaticity diagram background colours
        image.

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

    settings = {}
    settings.update(kwargs)
    settings.update({'cmfs': cmfs, 'standalone': False})

    chromaticity_diagram_callable_CIE1960UCS(**settings)

    for spd in spds:
        XYZ = spectral_to_XYZ(spd) / 100
        uv = UCS_to_uv(XYZ_to_UCS(XYZ))

        pylab.plot(uv[0], uv[1], 'o', color='white')

        if spd.name is not None and annotate:
            pylab.annotate(
                spd.name,
                xy=uv,
                xytext=(50, 30),
                color='black',
                textcoords='offset points',
                arrowprops=dict(
                    arrowstyle='->', connectionstyle='arc3, rad=0.2'))

    settings.update({
        'x_tighten': True,
        'y_tighten': True,
        'limits': (-0.1, 0.7, -0.2, 0.6),
        'standalone': True
    })
    settings.update(kwargs)

    return render(**settings)


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
        Standard observer colour matching functions used for diagram bounds.
    annotate : bool
        Should resulting chromaticity coordinates annotated with their
        respective spectral power distribution names.
    chromaticity_diagram_callable_CIE1976UCS : callable, optional
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.
    show_diagram_colours : bool, optional
        {:func:`colour.plotting.chromaticity_diagram_plot_CIE1976UCS`},
        Whether to display the chromaticity diagram background colours.
    use_cached_diagram_colours : bool, optional
        {:func:`colour.plotting.chromaticity_diagram_plot_CIE1976UCS`},
        Whether to used the cached chromaticity diagram background colours
        image.

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

    settings = {}
    settings.update(kwargs)
    settings.update({'cmfs': cmfs, 'standalone': False})

    chromaticity_diagram_callable_CIE1976UCS(**settings)

    for spd in spds:
        XYZ = spectral_to_XYZ(spd) / 100
        uv = Luv_to_uv(XYZ_to_Luv(XYZ))

        pylab.plot(uv[0], uv[1], 'o', color='white')

        if spd.name is not None and annotate:
            pylab.annotate(
                spd.name,
                xy=uv,
                xytext=(50, 30),
                color='black',
                textcoords='offset points',
                arrowprops=dict(
                    arrowstyle='->', connectionstyle='arc3, rad=0.2'))

    settings.update({
        'x_tighten': True,
        'y_tighten': True,
        'limits': (-0.1, 0.7, -0.1, 0.7),
        'standalone': True
    })
    settings.update(kwargs)

    return render(**settings)
