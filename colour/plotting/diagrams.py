#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE Chromaticity Diagrams Plotting
==================================

Defines the *CIE* chromaticity diagrams plotting objects:

-   :func:`CIE_1931_chromaticity_diagram_plot`
-   :func:`CIE_1960_UCS_chromaticity_diagram_plot`
-   :func:`CIE_1976_UCS_chromaticity_diagram_plot`
"""

from __future__ import division

import bisect
import os

import matplotlib
import matplotlib.image
import matplotlib.path
import numpy as np
import pylab

from colour.algebra import normalise
from colour.colorimetry import ILLUMINANTS
from colour.models import (
    UCS_uv_to_xy,
    XYZ_to_UCS,
    XYZ_to_xy,
    UCS_to_uv,
    xy_to_XYZ,
    XYZ_to_Luv,
    Luv_to_uv,
    Luv_uv_to_xy,
    XYZ_to_sRGB)
from colour.plotting import (
    DEFAULT_FIGURE_WIDTH,
    PLOTTING_RESOURCES_DIRECTORY,
    canvas,
    decorate,
    boundaries,
    display,
    get_cmfs)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CIE_1931_chromaticity_diagram_colours_plot',
           'CIE_1931_chromaticity_diagram_plot',
           'CIE_1960_UCS_chromaticity_diagram_colours_plot',
           'CIE_1960_UCS_chromaticity_diagram_plot',
           'CIE_1976_UCS_chromaticity_diagram_colours_plot',
           'CIE_1976_UCS_chromaticity_diagram_plot']


def CIE_1931_chromaticity_diagram_colours_plot(
        surface=1.25,
        spacing=0.00075,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots the *CIE 1931 Chromaticity Diagram* colours.

    Parameters
    ----------
    surface : numeric, optional
        Generated markers surface.
    spacing : numeric, optional
        Spacing between markers.
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> CIE_1931_chromaticity_diagram_colours_plot()  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (32, 32)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    illuminant = ILLUMINANTS.get(
        'CIE 1931 2 Degree Standard Observer').get('E')

    XYZs = [value for key, value in cmfs]

    path = matplotlib.path.Path([XYZ_to_xy(x) for x in XYZs])
    x_dot, y_dot, colours = [], [], []
    for i in np.arange(0, 1, spacing):
        for j in np.arange(0, 1, spacing):
            if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
                x_dot.append(i)
                y_dot.append(j)

                XYZ = xy_to_XYZ((i, j))
                RGB = normalise(XYZ_to_sRGB(XYZ, illuminant))

                colours.append(RGB)

    pylab.scatter(x_dot, y_dot, color=colours, s=surface)

    settings.update({
        'no_ticks': True,
        'bounding_box': [0, 1, 0, 1],
        'bbox_inches': 'tight',
        'pad_inches': 0})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def CIE_1931_chromaticity_diagram_plot(
        cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
    """
    Plots the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> CIE_1931_chromaticity_diagram_plot()  # doctest: +SKIP
    True

    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    image = matplotlib.image.imread(
        os.path.join(PLOTTING_RESOURCES_DIRECTORY,
                     'CIE_1931_Chromaticity_Diagram_{0}_Large.png'.format(
                         cmfs.name.replace(' ', '_'))))
    pylab.imshow(image, interpolation='nearest', extent=(0, 1, 0, 1))

    labels = (
        [390, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700])

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    xy = np.array([XYZ_to_xy(XYZ) for XYZ in cmfs.values])

    wavelengths_chromaticity_coordinates = dict(tuple(zip(wavelengths, xy)))

    pylab.plot(xy[:, 0], xy[:, 1], color='black', linewidth=2)
    pylab.plot((xy[-1][0], xy[0][0]),
               (xy[-1][1], xy[0][1]),
               color='black',
               linewidth=2)

    for label in labels:
        x, y = wavelengths_chromaticity_coordinates.get(label)
        pylab.plot(x, y, 'o', color='black', linewidth=2)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else
                 wavelengths[-1])

        dx = (wavelengths_chromaticity_coordinates.get(right)[0] -
              wavelengths_chromaticity_coordinates.get(left)[0])
        dy = (wavelengths_chromaticity_coordinates.get(right)[1] -
              wavelengths_chromaticity_coordinates.get(left)[1])

        norme = lambda x: x / np.linalg.norm(x)

        xy = np.array([x, y])
        direction = np.array((-dy, dx))

        normal = (np.array((-dy, dx))
                  if np.dot(norme(xy - equal_energy),
                            norme(direction)) > 0 else
                  np.array((dy, -dx)))
        normal = norme(normal)
        normal /= 25

        pylab.plot([x, x + normal[0] * 0.75],
                   [y, y + normal[1] * 0.75],
                   color='black',
                   linewidth=1.5)
        pylab.text(x + normal[0],
                   y + normal[1],
                   label,
                   clip_on=True,
                   ha='left' if normal[0] >= 0 else 'right',
                   va='center',
                   fontdict={'size': 'small'})

    settings.update({
        'title': 'CIE 1931 Chromaticity Diagram - {0}'.format(cmfs.title),
        'x_label': 'CIE x',
        'y_label': 'CIE y',
        'x_ticker': True,
        'y_ticker': True,
        'grid': True,
        'bounding_box': [-0.1, 0.9, -0.1, 0.9],
        'bbox_inches': 'tight',
        'pad_inches': 0})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def CIE_1960_UCS_chromaticity_diagram_colours_plot(
        surface=1.25,
        spacing=0.00075,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram* colours.

    Parameters
    ----------
    surface : numeric, optional
        Generated markers surface.
    spacing : numeric, optional
        Spacing between markers.
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> CIE_1960_UCS_chromaticity_diagram_colours_plot()  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (32, 32)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    illuminant = ILLUMINANTS.get(
        'CIE 1931 2 Degree Standard Observer').get('E')

    uv = np.array([UCS_to_uv(XYZ_to_UCS(XYZ)) for XYZ in cmfs.values])

    path = matplotlib.path.Path(uv)
    x_dot, y_dot, colours = [], [], []
    for i in np.arange(0, 1, spacing):
        for j in np.arange(0, 1, spacing):
            if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
                x_dot.append(i)
                y_dot.append(j)

                XYZ = xy_to_XYZ(UCS_uv_to_xy((i, j)))
                RGB = normalise(XYZ_to_sRGB(XYZ, illuminant))

                colours.append(RGB)

    pylab.scatter(x_dot, y_dot, color=colours, s=surface)

    settings.update({
        'no_ticks': True,
        'bounding_box': [0, 1, 0, 1],
        'bbox_inches': 'tight',
        'pad_inches': 0})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def CIE_1960_UCS_chromaticity_diagram_plot(
        cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
    """
    Plots the *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> CIE_1960_UCS_chromaticity_diagram_plot()  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    image = matplotlib.image.imread(
        os.path.join(PLOTTING_RESOURCES_DIRECTORY,
                     'CIE_1960_UCS_Chromaticity_Diagram_{0}_Large.png'.format(
                         cmfs.name.replace(' ', '_'))))
    pylab.imshow(image, interpolation='nearest', extent=(0, 1, 0, 1))

    labels = [420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530,
              540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 680]

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    uv = np.array([UCS_to_uv(XYZ_to_UCS(XYZ)) for XYZ in cmfs.values])

    wavelengths_chromaticity_coordinates = dict(tuple(zip(wavelengths, uv)))

    pylab.plot(uv[:, 0], uv[:, 1], color='black', linewidth=2)
    pylab.plot((uv[-1][0], uv[0][0]),
               (uv[-1][1], uv[0][1]),
               color='black',
               linewidth=2)

    for label in labels:
        u, v = wavelengths_chromaticity_coordinates.get(label)
        pylab.plot(u, v, 'o', color='black', linewidth=2)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else
                 wavelengths[-1])

        dx = (wavelengths_chromaticity_coordinates.get(right)[0] -
              wavelengths_chromaticity_coordinates.get(left)[0])
        dy = (wavelengths_chromaticity_coordinates.get(right)[1] -
              wavelengths_chromaticity_coordinates.get(left)[1])

        norme = lambda x: x / np.linalg.norm(x)

        uv = np.array([u, v])
        direction = np.array((-dy, dx))

        normal = (np.array((-dy, dx))
                  if np.dot(norme(uv - equal_energy),
                            norme(direction)) > 0 else
                  np.array((dy, -dx)))
        normal = norme(normal)
        normal /= 25

        pylab.plot([u, u + normal[0] * 0.75],
                   [v, v + normal[1] * 0.75],
                   color='black',
                   linewidth=1.5)
        pylab.text(u + normal[0],
                   v + normal[1],
                   label,
                   clip_on=True,
                   ha='left' if normal[0] >= 0 else 'right',
                   va='center',
                   fontdict={'size': 'small'})

    settings.update({
        'title': 'CIE 1960 UCS Chromaticity Diagram - {0}'.format(cmfs.title),
        'x_label': 'CIE u',
        'y_label': 'CIE v',
        'x_ticker': True,
        'y_ticker': True,
        'grid': True,
        'bounding_box': [-0.075, 0.675, -0.15, 0.6],
        'bbox_inches': 'tight',
        'pad_inches': 0})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def CIE_1976_UCS_chromaticity_diagram_colours_plot(
        surface=1.25,
        spacing=0.00075,
        cmfs='CIE 1931 2 Degree Standard Observer',
        **kwargs):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram* colours.

    Parameters
    ----------
    surface : numeric, optional
        Generated markers surface.
    spacing : numeric, optional
        Spacing between markers.
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> CIE_1976_UCS_chromaticity_diagram_colours_plot()  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (32, 32)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    illuminant = ILLUMINANTS.get(
        'CIE 1931 2 Degree Standard Observer').get('D50')

    uv = np.array([Luv_to_uv(XYZ_to_Luv(XYZ, illuminant))
                   for XYZ in cmfs.values])

    path = matplotlib.path.Path(uv)
    x_dot, y_dot, colours = [], [], []
    for i in np.arange(0, 1, spacing):
        for j in np.arange(0, 1, spacing):
            if path.contains_path(matplotlib.path.Path([[i, j], [i, j]])):
                x_dot.append(i)
                y_dot.append(j)

                XYZ = xy_to_XYZ(Luv_uv_to_xy((i, j)))
                RGB = normalise(XYZ_to_sRGB(XYZ, illuminant))

                colours.append(RGB)

    pylab.scatter(x_dot, y_dot, color=colours, s=surface)

    settings.update({
        'no_ticks': True,
        'bounding_box': [0, 1, 0, 1],
        'bbox_inches': 'tight',
        'pad_inches': 0})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)


def CIE_1976_UCS_chromaticity_diagram_plot(
        cmfs='CIE 1931 2 Degree Standard Observer', **kwargs):
    """
    Plots the *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs : unicode, optional
        Standard observer colour matching functions used for diagram bounds.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> CIE_1976_UCS_chromaticity_diagram_plot()  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    cmfs = get_cmfs(cmfs)

    image = matplotlib.image.imread(
        os.path.join(PLOTTING_RESOURCES_DIRECTORY,
                     'CIE_1976_UCS_Chromaticity_Diagram_{0}_Large.png'.format(
                         cmfs.name.replace(' ', '_'))))
    pylab.imshow(image, interpolation='nearest', extent=(0, 1, 0, 1))

    labels = [420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530,
              540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 680]

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    illuminant = ILLUMINANTS.get(
        'CIE 1931 2 Degree Standard Observer').get('D50')

    uv = np.array([Luv_to_uv(XYZ_to_Luv(XYZ, illuminant))
                   for XYZ in cmfs.values])

    wavelengths_chromaticity_coordinates = dict(zip(wavelengths, uv))

    pylab.plot(uv[:, 0], uv[:, 1], color='black', linewidth=2)
    pylab.plot((uv[-1][0], uv[0][0]),
               (uv[-1][1], uv[0][1]),
               color='black',
               linewidth=2)

    for label in labels:
        u, v = wavelengths_chromaticity_coordinates.get(label)
        pylab.plot(u, v, 'o', color='black', linewidth=2)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                 if index < len(wavelengths) else
                 wavelengths[-1])

        dx = (wavelengths_chromaticity_coordinates.get(right)[0] -
              wavelengths_chromaticity_coordinates.get(left)[0])
        dy = (wavelengths_chromaticity_coordinates.get(right)[1] -
              wavelengths_chromaticity_coordinates.get(left)[1])

        norme = lambda x: x / np.linalg.norm(x)

        uv = np.array([u, v])
        direction = np.array((-dy, dx))

        normal = (np.array((-dy, dx))
                  if np.dot(norme(uv - equal_energy),
                            norme(direction)) > 0 else
                  np.array((dy, -dx)))
        normal = norme(normal)
        normal /= 25

        pylab.plot([u, u + normal[0] * 0.75],
                   [v, v + normal[1] * 0.75],
                   color='black',
                   linewidth=1.5)
        pylab.text(u + normal[0],
                   v + normal[1],
                   label,
                   clip_on=True,
                   ha='left' if normal[0] >= 0 else 'right',
                   va='center',
                   fontdict={'size': 'small'})

    settings.update({
        'title': 'CIE 1976 UCS Chromaticity Diagram - {0}'.format(cmfs.title),
        'x_label': 'CIE u\'',
        'y_label': 'CIE v\'',
        'x_ticker': True,
        'y_ticker': True,
        'grid': True,
        'bounding_box': [-0.1, .7, -.1, .7],
        'bbox_inches': 'tight',
        'pad_inches': 0})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)

    return display(**settings)
