# -*- coding: utf-8 -*-
"""
Gamut Section Plotting
======================

Defines the gamut section plotting objects:

-   :func:`colour.plotting.section.plot_hull_section_colours`
-   :func:`colour.plotting.section.plot_hull_section_contour`
-   :func:`colour.plotting.plot_visible_spectrum_section`
-   :func:`colour.plotting.plot_RGB_colourspace_section`
"""

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from colour.colorimetry import SpectralShape, reshape_msds
from colour.geometry import hull_section, primitive_cube
from colour.graph import convert
from colour.models import (
    COLOURSPACE_MODELS_AXIS_LABELS,
    COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE, RGB_to_XYZ)
from colour.notation import HEX_to_RGB
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE, XYZ_to_plotting_colourspace, artist,
    colourspace_model_axis_reorder, filter_cmfs, filter_RGB_colourspaces,
    filter_illuminants, override_style, render)
from colour.volume import solid_RoschMacAdam
from colour.utilities import (CaseInsensitiveMapping, as_int_array, first_item,
                              full, required, suppress_warnings, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'AXIS_TO_PLANE_MAPPING',
    'plot_hull_section_colours',
    'plot_hull_section_contour',
    'plot_visible_spectrum_section',
    'plot_RGB_colourspace_section',
]

AXIS_TO_PLANE_MAPPING = CaseInsensitiveMapping({
    '+x': (1, 2),
    '+y': (0, 2),
    '+z': (0, 1)
})
AXIS_TO_PLANE_MAPPING.__doc__ = """
Axis to plane mapping.

AXIS_TO_PLANE_MAPPING : CaseInsensitiveMapping
    **{'+x', '+y', '+z'}**
"""


@required('trimesh')
@override_style()
def plot_hull_section_colours(hull,
                              model='CIE xyY',
                              axis='+z',
                              origin=0.5,
                              normalise=True,
                              section_colours=None,
                              section_opacity=1.0,
                              convert_kwargs=None,
                              samples=256,
                              **kwargs):
    """
    Plots the section colours of given *trimesh* hull along given axis and
    origin.

    Parameters
    ----------
    hull : Trimesh
        *Trimesh* hull.
    model : str, optional
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    axis : str, optional
        **{'+z', '+x', '+y'}**,
        Axis the hull section will be normal to.
    origin : numeric, optional
        Coordinate along ``axis`` at which to plot the hull section.
    normalise : bool, optional
        Whether to normalise ``axis`` to the extent of the hull along it.
    section_colours : array_like or str, optional
        Colours of the hull section, if ``section_colours`` is set to *RGB*,
        the colours will be computed according to the corresponding
        coordinates.
    section_opacity : numeric, optional
        Opacity of the hull section colours.
    convert_kwargs : dict, optional
        Keyword arguments for the :func:`colour.convert` definition.
    samples : numeric, optional
        Samples count when computing the hull section colours.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB
    >>> from colour.utilities import is_trimesh_installed
    >>> vertices, faces, _outline = primitive_cube(1, 1, 1, 64, 64, 64)
    >>> XYZ_vertices = RGB_to_XYZ(
    ...     vertices['position'] + 0.5,
    ...     RGB_COLOURSPACE_sRGB.whitepoint,
    ...     RGB_COLOURSPACE_sRGB.whitepoint,
    ...     RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ,
    ... )
    >>> if is_trimesh_installed:
    ...     import trimesh
    ...     hull = trimesh.Trimesh(XYZ_vertices, faces, process=False)
    ...     plot_hull_section_colours(hull, section_colours='RGB')
    ...     # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Hull_Section_Colours.png
        :align: center
        :alt: plot_hull_section_colours
    """

    hull = hull.copy()

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if section_colours is None:
        section_colours = HEX_to_RGB(CONSTANTS_COLOUR_STYLE.colour.average)

    if convert_kwargs is None:
        convert_kwargs = {}

    # Luminance / Lightness reordered along "z" axis.
    with suppress_warnings(python_warnings=True):
        ijk_vertices = colourspace_model_axis_reorder(
            convert(hull.vertices, 'CIE XYZ', model, **convert_kwargs), model)
        ijk_vertices = np.nan_to_num(ijk_vertices)
        ijk_vertices *= (
            COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model])

    hull.vertices = ijk_vertices

    if axis == '+x':
        index_origin = 0
    elif axis == '+y':
        index_origin = 1
    elif axis == '+z':
        index_origin = 2
    plane = AXIS_TO_PLANE_MAPPING[axis]

    section = hull_section(hull, axis, origin, normalise)

    padding = 0.1 * np.mean(
        COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model])
    min_x = np.min(ijk_vertices[..., plane[0]]) - padding
    max_x = np.max(ijk_vertices[..., plane[0]]) + padding
    min_y = np.min(ijk_vertices[..., plane[1]]) - padding
    max_y = np.max(ijk_vertices[..., plane[1]]) + padding
    extent = (min_x, max_x, min_y, max_y)

    is_section_colours_RGB = str(section_colours).upper() == 'RGB'
    if is_section_colours_RGB:
        ii, jj = np.meshgrid(
            np.linspace(min_x, max_x, samples),
            np.linspace(max_y, min_y, samples))
        ij = tstack([ii, jj])
        ijk_section = full([samples, samples, 3],
                           np.median(section[..., index_origin]))
        ijk_section[..., plane] = ij
        ijk_section /= (
            COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model])
        XYZ_section = convert(
            colourspace_model_axis_reorder(ijk_section, model, 'Inverse'),
            model, 'CIE XYZ', **convert_kwargs)
        RGB_section = XYZ_to_plotting_colourspace(XYZ_section)
    else:
        section_colours = np.hstack([section_colours, section_opacity])

    facecolor = 'none' if is_section_colours_RGB else section_colours
    polygon = Polygon(
        section[..., plane], facecolor=facecolor, edgecolor='none')
    axes.add_patch(polygon)
    if is_section_colours_RGB:
        image = axes.imshow(
            np.clip(RGB_section, 0, 1),
            interpolation='bilinear',
            extent=extent,
            clip_path=None,
            alpha=section_opacity)
        image.set_clip_path(polygon)

    settings = {
        'axes': axes,
        'bounding_box': extent,
    }
    settings.update(kwargs)

    return render(**settings)


@required('trimesh')
@override_style()
def plot_hull_section_contour(hull,
                              model='CIE xyY',
                              axis='+z',
                              origin=0.5,
                              normalise=True,
                              contour_colours=None,
                              contour_opacity=1,
                              convert_kwargs=None,
                              **kwargs):
    """
    Plots the section contour of given *trimesh* hull along given axis and
    origin.

    Parameters
    ----------
    hull : Trimesh
        *Trimesh* hull.
    model : str, optional
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    axis : str, optional
        **{'+z', '+x', '+y'}**,
        Axis the hull section will be normal to.
    origin : numeric, optional
        Coordinate along ``axis`` at which to plot the hull section.
    normalise : bool, optional
        Whether to normalise ``axis`` to the extent of the hull along it.
    contour_colours : array_like or str, optional
        Colours of the hull section contour, if ``contour_colours`` is set to
        *RGB*, the colours will be computed according to the corresponding
        coordinates.
    contour_opacity : numeric, optional
        Opacity of the hull section contour.
    convert_kwargs : dict, optional
        Keyword arguments for the :func:`colour.convert` definition.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB
    >>> from colour.utilities import is_trimesh_installed
    >>> vertices, faces, _outline = primitive_cube(1, 1, 1, 64, 64, 64)
    >>> XYZ_vertices = RGB_to_XYZ(
    ...     vertices['position'] + 0.5,
    ...     RGB_COLOURSPACE_sRGB.whitepoint,
    ...     RGB_COLOURSPACE_sRGB.whitepoint,
    ...     RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ,
    ... )
    >>> if is_trimesh_installed:
    ...     import trimesh
    ...     hull = trimesh.Trimesh(XYZ_vertices, faces, process=False)
    ...     plot_hull_section_contour(hull, contour_colours='RGB')
    ...     # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Hull_Section_Contour.png
        :align: center
        :alt: plot_hull_section_contour
    """

    hull = hull.copy()

    if contour_colours is None:
        contour_colours = CONSTANTS_COLOUR_STYLE.colour.dark

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if convert_kwargs is None:
        convert_kwargs = {}

    # Luminance / Lightness is re-ordered along "z-up" axis.
    with suppress_warnings(python_warnings=True):
        ijk_vertices = colourspace_model_axis_reorder(
            convert(hull.vertices, 'CIE XYZ', model, **convert_kwargs), model)
        ijk_vertices = np.nan_to_num(ijk_vertices)
        ijk_vertices *= (
            COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model])

    hull.vertices = ijk_vertices

    plane = AXIS_TO_PLANE_MAPPING[axis]

    padding = 0.1 * np.mean(
        COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model])
    min_x = np.min(ijk_vertices[..., plane[0]]) - padding
    max_x = np.max(ijk_vertices[..., plane[0]]) + padding
    min_y = np.min(ijk_vertices[..., plane[1]]) - padding
    max_y = np.max(ijk_vertices[..., plane[1]]) + padding
    extent = (min_x, max_x, min_y, max_y)

    contour_colours_RGB = str(contour_colours).upper() == 'RGB'
    section = hull_section(hull, axis, origin, normalise)
    if contour_colours_RGB:
        ijk_section = section / (
            COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model])
        XYZ_section = convert(
            colourspace_model_axis_reorder(ijk_section, model, 'Inverse'),
            model, 'CIE XYZ', **convert_kwargs)
        contour_colours = np.clip(
            XYZ_to_plotting_colourspace(XYZ_section), 0, 1)

    section = section[..., plane].reshape(-1, 1, 2)
    line_collection = LineCollection(
        np.concatenate([section[:-1], section[1:]], axis=1),
        colors=contour_colours,
        alpha=contour_opacity)
    axes.add_collection(line_collection)

    settings = {
        'axes': axes,
        'bounding_box': extent,
    }
    settings.update(kwargs)

    return render(**settings)


@required('trimesh')
@override_style()
def plot_visible_spectrum_section(cmfs='CIE 1931 2 Degree Standard Observer',
                                  illuminant='D65',
                                  model='CIE xyY',
                                  axis='+z',
                                  origin=0.5,
                                  normalise=True,
                                  show_section_colours=True,
                                  show_section_contour=True,
                                  **kwargs):
    """
    Plots the visible spectrum volume, i.e. *Rösch-MacAdam* colour solid,
    section colours along given axis and origin.

    Parameters
    ----------
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution, default to *CIE Illuminant D65*.
    model : str, optional
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    axis : str, optional
        **{'+z', '+x', '+y'}**,
        Axis the hull section will be normal to.
    origin : numeric, optional
        Coordinate along ``axis`` at which to plot the hull section.
    normalise : bool, optional
        Whether to normalise ``axis`` to the extent of the hull along it.
    show_section_colours : bool, optional
        Whether to show the hull section colours.
    show_section_contour : bool, optional
        Whether to show the hull section contour.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`,
        :func:`colour.plotting.section.plot_hull_section_colours`
        :func:`colour.plotting.section.plot_hull_section_contour`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour.utilities import is_trimesh_installed
    >>> if is_trimesh_installed:
    ...     plot_visible_spectrum_section(
    ...         section_colours='RGB', section_opacity=0.15)
    ...     # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Visible_Spectrum_Section.png
        :align: center
        :alt: plot_visible_spectrum_section
    """

    import trimesh

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    # pylint: disable=E1102
    cmfs = reshape_msds(
        first_item(filter_cmfs(cmfs).values()), SpectralShape(360, 780, 1))
    illuminant = first_item(filter_illuminants(illuminant).values())

    vertices = solid_RoschMacAdam(
        cmfs,
        illuminant,
        point_order='Pulse Wave Width',
        filter_jagged_points=True,
    )
    mesh = trimesh.Trimesh(vertices)
    hull = trimesh.convex.convex_hull(mesh)

    if show_section_colours:
        settings = {'axes': axes}
        settings.update(kwargs)
        settings['standalone'] = False

        plot_hull_section_colours(hull, model, axis, origin, normalise,
                                  **settings)

    if show_section_contour:
        settings = {'axes': axes}
        settings.update(kwargs)
        settings['standalone'] = False

        plot_hull_section_contour(hull, model, axis, origin, normalise,
                                  **settings)

    title = 'Visible Spectrum Section - {0} - {1} - {2}'.format(
        '{0}%'.format(origin * 100)
        if normalise else origin, model, cmfs.strict_name)

    plane = AXIS_TO_PLANE_MAPPING[axis]

    labels = np.array(COLOURSPACE_MODELS_AXIS_LABELS[model])[as_int_array(
        colourspace_model_axis_reorder([0, 1, 2], model))]
    x_label, y_label = labels[plane[0]], labels[plane[1]]

    settings.update({
        'axes': axes,
        'standalone': True,
        'title': title,
        'x_label': x_label,
        'y_label': y_label,
    })
    settings.update(kwargs)

    return render(**settings)


@required('trimesh')
@override_style()
def plot_RGB_colourspace_section(colourspace,
                                 model='CIE xyY',
                                 axis='+z',
                                 origin=0.5,
                                 normalise=True,
                                 show_section_colours=True,
                                 show_section_contour=True,
                                 **kwargs):
    """
    Plots given *RGB* colourspace section colours along given axis and origin.

    Parameters
    ----------
    colourspace : str or RGB_Colourspace, optional
        *RGB* colourspace of the *RGB* array. ``colourspace`` can be of any
        type or form supported by the
        :func:`colour.plotting.filter_RGB_colourspaces` definition.
    model : str, optional
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    axis : str, optional
        **{'+z', '+x', '+y'}**,
        Axis the hull section will be normal to.
    origin : numeric, optional
        Coordinate along ``axis`` at which to plot the hull section.
    normalise : bool, optional
        Whether to normalise ``axis`` to the extent of the hull along it.
    show_section_colours : bool, optional
        Whether to show the hull section colours.
    show_section_contour : bool, optional
        Whether to show the hull section contour.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`,
        :func:`colour.plotting.section.plot_hull_section_colours`
        :func:`colour.plotting.section.plot_hull_section_contour`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour.utilities import is_trimesh_installed
    >>> if is_trimesh_installed:
    ...     plot_RGB_colourspace_section(
    ...         'sRGB', section_colours='RGB', section_opacity=0.15)
    ...     # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_RGB_Colourspace_Section.png
        :align: center
        :alt: plot_RGB_colourspace_section
    """

    import trimesh

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    colourspace = first_item(filter_RGB_colourspaces(colourspace).values())

    vertices, faces, _outline = primitive_cube(1, 1, 1, 64, 64, 64)
    XYZ_vertices = RGB_to_XYZ(
        vertices['position'] + 0.5,
        colourspace.whitepoint,
        colourspace.whitepoint,
        colourspace.matrix_RGB_to_XYZ,
    )
    hull = trimesh.Trimesh(XYZ_vertices, faces, process=False)

    if show_section_colours:
        settings = {'axes': axes}
        settings.update(kwargs)
        settings['standalone'] = False

        plot_hull_section_colours(hull, model, axis, origin, normalise,
                                  **settings)

    if show_section_contour:
        settings = {'axes': axes}
        settings.update(kwargs)
        settings['standalone'] = False

        plot_hull_section_contour(hull, model, axis, origin, normalise,
                                  **settings)

    title = '{0} Section - {1} - {2}'.format(
        colourspace.name, '{0}%'.format(origin * 100)
        if normalise else origin, model)

    plane = AXIS_TO_PLANE_MAPPING[axis]

    labels = np.array(COLOURSPACE_MODELS_AXIS_LABELS[model])[as_int_array(
        colourspace_model_axis_reorder([0, 1, 2], model))]
    x_label, y_label = labels[plane[0]], labels[plane[1]]

    settings.update({
        'axes': axes,
        'standalone': True,
        'title': title,
        'x_label': x_label,
        'y_label': y_label,
    })
    settings.update(kwargs)

    return render(**settings)
