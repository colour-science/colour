"""
Gamut Section Plotting
======================

Defines the gamut section plotting objects:

-   :func:`colour.plotting.section.plot_hull_section_colours`
-   :func:`colour.plotting.section.plot_hull_section_contour`
-   :func:`colour.plotting.plot_visible_spectrum_section`
-   :func:`colour.plotting.plot_RGB_colourspace_section`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    reshape_msds,
)
from colour.geometry import hull_section, primitive_cube
from colour.graph import convert
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Dict,
    Floating,
    Integer,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from colour.models import (
    COLOURSPACE_MODELS_AXIS_LABELS,
    COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE,
    RGB_Colourspace,
    RGB_to_XYZ,
)
from colour.notation import HEX_to_RGB
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    XYZ_to_plotting_colourspace,
    artist,
    colourspace_model_axis_reorder,
    filter_cmfs,
    filter_RGB_colourspaces,
    filter_illuminants,
    override_style,
    render,
)
from colour.volume import solid_RoschMacAdam
from colour.utilities import (
    CaseInsensitiveMapping,
    as_int_array,
    first_item,
    full,
    optional,
    required,
    suppress_warnings,
    tstack,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MAPPING_AXIS_TO_PLANE",
    "plot_hull_section_colours",
    "plot_hull_section_contour",
    "plot_visible_spectrum_section",
    "plot_RGB_colourspace_section",
]

MAPPING_AXIS_TO_PLANE: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {"+x": (1, 2), "+y": (0, 2), "+z": (0, 1)}
)
MAPPING_AXIS_TO_PLANE.__doc__ = """Axis to plane mapping."""


@required("trimesh")
@override_style()
def plot_hull_section_colours(
    hull: trimesh.Trimesh,  # type: ignore[name-defined]  # noqa
    model: Union[
        Literal[
            "CAM02LCD",
            "CAM02SCD",
            "CAM02UCS",
            "CAM16LCD",
            "CAM16SCD",
            "CAM16UCS",
            "CIE XYZ",
            "CIE xyY",
            "CIE Lab",
            "CIE Luv",
            "CIE UCS",
            "CIE UVW",
            "DIN99",
            "Hunter Lab",
            "Hunter Rdab",
            "ICaCb",
            "ICtCp",
            "IPT",
            "IgPgTg",
            "Jzazbz",
            "OSA UCS",
            "Oklab",
            "hdr-CIELAB",
            "hdr-IPT",
        ],
        str,
    ] = "CIE xyY",
    axis: Union[Literal["+z", "+x", "+y"], str] = "+z",
    origin: Floating = 0.5,
    normalise: Boolean = True,
    section_colours: Optional[Union[ArrayLike, str]] = None,
    section_opacity: Floating = 1,
    convert_kwargs: Optional[Dict] = None,
    samples: Integer = 256,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the section colours of given *trimesh* hull along given axis and
    origin.

    Parameters
    ----------
    hull
        *Trimesh* hull.
    model
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    axis
        Axis the hull section will be normal to.
    origin
        Coordinate along ``axis`` at which to plot the hull section.
    normalise
        Whether to normalise ``axis`` to the extent of the hull along it.
    section_colours
        Colours of the hull section, if ``section_colours`` is set to *RGB*,
        the colours will be computed according to the corresponding
        coordinates.
    section_opacity
        Opacity of the hull section colours.
    convert_kwargs
        Keyword arguments for the :func:`colour.convert` definition.
    samples
        Samples count on one axis when computing the hull section colours.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
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

    axis = validate_method(
        axis,
        ["+z", "+x", "+y"],
        '"{0}" axis is invalid, it must be one of {1}!',
    )

    hull = hull.copy()

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    section_colours = cast(
        ArrayLike,
        optional(
            section_colours, HEX_to_RGB(CONSTANTS_COLOUR_STYLE.colour.average)
        ),
    )

    convert_kwargs = optional(convert_kwargs, {})

    # Luminance / Lightness reordered along "z" axis.
    with suppress_warnings(python_warnings=True):
        ijk_vertices = colourspace_model_axis_reorder(
            convert(hull.vertices, "CIE XYZ", model, **convert_kwargs), model
        )
        ijk_vertices = np.nan_to_num(ijk_vertices)
        ijk_vertices *= COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[
            model
        ]

    hull.vertices = ijk_vertices

    if axis == "+x":
        index_origin = 0
    elif axis == "+y":
        index_origin = 1
    elif axis == "+z":
        index_origin = 2
    plane = MAPPING_AXIS_TO_PLANE[axis]

    section = hull_section(hull, axis, origin, normalise)

    padding = 0.1 * np.mean(
        COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model]
    )
    min_x = np.min(ijk_vertices[..., plane[0]]) - padding
    max_x = np.max(ijk_vertices[..., plane[0]]) + padding
    min_y = np.min(ijk_vertices[..., plane[1]]) - padding
    max_y = np.max(ijk_vertices[..., plane[1]]) + padding
    extent = (min_x, max_x, min_y, max_y)

    use_RGB_section_colours = str(section_colours).upper() == "RGB"
    if use_RGB_section_colours:
        ii, jj = np.meshgrid(
            np.linspace(min_x, max_x, samples),
            np.linspace(max_y, min_y, samples),
        )
        ij = tstack([ii, jj])
        ijk_section = full(
            (samples, samples, 3), np.median(section[..., index_origin])
        )
        ijk_section[..., plane] = ij
        ijk_section /= COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[
            model
        ]
        XYZ_section = convert(
            colourspace_model_axis_reorder(ijk_section, model, "Inverse"),
            model,
            "CIE XYZ",
            **convert_kwargs,
        )
        RGB_section = XYZ_to_plotting_colourspace(XYZ_section)
    else:
        section_colours = np.hstack([section_colours, section_opacity])

    facecolor = "none" if use_RGB_section_colours else section_colours
    polygon = Polygon(
        section[..., plane],
        facecolor=facecolor,
        edgecolor="none",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
    )
    axes.add_patch(polygon)
    if use_RGB_section_colours:
        image = axes.imshow(
            np.clip(RGB_section, 0, 1),
            interpolation="bilinear",
            extent=extent,
            clip_path=None,
            alpha=section_opacity,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
        )
        image.set_clip_path(polygon)

    settings = {
        "axes": axes,
        "bounding_box": extent,
    }
    settings.update(kwargs)

    return render(**settings)


@required("trimesh")
@override_style()
def plot_hull_section_contour(
    hull: trimesh.Trimesh,  # type: ignore[name-defined]  # noqa
    model: Union[
        Literal[
            "CAM02LCD",
            "CAM02SCD",
            "CAM02UCS",
            "CAM16LCD",
            "CAM16SCD",
            "CAM16UCS",
            "CIE XYZ",
            "CIE xyY",
            "CIE Lab",
            "CIE Luv",
            "CIE UCS",
            "CIE UVW",
            "DIN99",
            "Hunter Lab",
            "Hunter Rdab",
            "ICaCb",
            "ICtCp",
            "IPT",
            "IgPgTg",
            "Jzazbz",
            "OSA UCS",
            "Oklab",
            "hdr-CIELAB",
            "hdr-IPT",
        ],
        str,
    ] = "CIE xyY",
    axis: Union[Literal["+z", "+x", "+y"], str] = "+z",
    origin: Floating = 0.5,
    normalise: Boolean = True,
    contour_colours: Optional[Union[ArrayLike, str]] = None,
    contour_opacity: Floating = 1,
    convert_kwargs: Optional[Dict] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the section contour of given *trimesh* hull along given axis and
    origin.

    Parameters
    ----------
    hull
        *Trimesh* hull.
    model
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    axis
        Axis the hull section will be normal to.
    origin
        Coordinate along ``axis`` at which to plot the hull section.
    normalise
        Whether to normalise ``axis`` to the extent of the hull along it.
    contour_colours
        Colours of the hull section contour, if ``contour_colours`` is set to
        *RGB*, the colours will be computed according to the corresponding
        coordinates.
    contour_opacity
        Opacity of the hull section contour.
    convert_kwargs
        Keyword arguments for the :func:`colour.convert` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
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

    contour_colours = cast(
        Union[ArrayLike, str],
        optional(contour_colours, CONSTANTS_COLOUR_STYLE.colour.dark),
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    convert_kwargs = optional(convert_kwargs, {})

    # Luminance / Lightness is re-ordered along "z-up" axis.
    with suppress_warnings(python_warnings=True):
        ijk_vertices = colourspace_model_axis_reorder(
            convert(hull.vertices, "CIE XYZ", model, **convert_kwargs), model
        )
        ijk_vertices = np.nan_to_num(ijk_vertices)
        ijk_vertices *= COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[
            model
        ]

    hull.vertices = ijk_vertices

    plane = MAPPING_AXIS_TO_PLANE[axis]

    padding = 0.1 * np.mean(
        COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model]
    )
    min_x = np.min(ijk_vertices[..., plane[0]]) - padding
    max_x = np.max(ijk_vertices[..., plane[0]]) + padding
    min_y = np.min(ijk_vertices[..., plane[1]]) - padding
    max_y = np.max(ijk_vertices[..., plane[1]]) + padding
    extent = (min_x, max_x, min_y, max_y)

    use_RGB_contour_colours = str(contour_colours).upper() == "RGB"
    section = hull_section(hull, axis, origin, normalise)
    if use_RGB_contour_colours:
        ijk_section = section / (
            COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model]
        )
        XYZ_section = convert(
            colourspace_model_axis_reorder(ijk_section, model, "Inverse"),
            model,
            "CIE XYZ",
            **convert_kwargs,
        )
        contour_colours = np.clip(
            XYZ_to_plotting_colourspace(XYZ_section), 0, 1
        )

    section = np.reshape(section[..., plane], (-1, 1, 2))
    line_collection = LineCollection(
        np.concatenate([section[:-1], section[1:]], axis=1),
        colors=contour_colours,
        alpha=contour_opacity,
        zorder=CONSTANTS_COLOUR_STYLE.zorder.background_line,
    )
    axes.add_collection(line_collection)

    settings = {
        "axes": axes,
        "bounding_box": extent,
    }
    settings.update(kwargs)

    return render(**settings)


@required("trimesh")
@override_style()
def plot_visible_spectrum_section(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    illuminant: Union[SpectralDistribution, str] = "D65",
    model: Union[
        Literal[
            "CAM02LCD",
            "CAM02SCD",
            "CAM02UCS",
            "CAM16LCD",
            "CAM16SCD",
            "CAM16UCS",
            "CIE XYZ",
            "CIE xyY",
            "CIE Lab",
            "CIE Luv",
            "CIE UCS",
            "CIE UVW",
            "DIN99",
            "Hunter Lab",
            "Hunter Rdab",
            "ICaCb",
            "ICtCp",
            "IPT",
            "IgPgTg",
            "Jzazbz",
            "OSA UCS",
            "Oklab",
            "hdr-CIELAB",
            "hdr-IPT",
        ],
        str,
    ] = "CIE xyY",
    axis: Union[Literal["+z", "+x", "+y"], str] = "+z",
    origin: Floating = 0.5,
    normalise: Boolean = True,
    show_section_colours: Boolean = True,
    show_section_contour: Boolean = True,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the visible spectrum volume, i.e. *Rösch-MacAdam* colour solid,
    section colours along given axis and origin.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.  ``cmfs`` can be of any type or
        form supported by the :func:`colour.plotting.filter_cmfs` definition.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant D65*.
        ``illuminant`` can be of any type or form supported by the
        :func:`colour.plotting.filter_illuminants` definition.
    model
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    axis
        Axis the hull section will be normal to.
    origin
        Coordinate along ``axis`` at which to plot the hull section.
    normalise
        Whether to normalise ``axis`` to the extent of the hull along it.
    show_section_colours
        Whether to show the hull section colours.
    show_section_contour
        Whether to show the hull section contour.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`,
        :func:`colour.plotting.section.plot_hull_section_colours`
        :func:`colour.plotting.section.plot_hull_section_contour`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
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

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    # pylint: disable=E1102
    cmfs = reshape_msds(
        first_item(filter_cmfs(cmfs).values()), SpectralShape(360, 780, 1)
    )
    illuminant = cast(
        SpectralDistribution,
        first_item(filter_illuminants(illuminant).values()),
    )

    vertices = solid_RoschMacAdam(
        cmfs,
        illuminant,
        point_order="Pulse Wave Width",
        filter_jagged_points=True,
    )
    mesh = trimesh.Trimesh(vertices)
    hull = trimesh.convex.convex_hull(mesh)

    if show_section_colours:
        settings = {"axes": axes}
        settings.update(kwargs)
        settings["standalone"] = False

        plot_hull_section_colours(
            hull, model, axis, origin, normalise, **settings
        )

    if show_section_contour:
        settings = {"axes": axes}
        settings.update(kwargs)
        settings["standalone"] = False

        plot_hull_section_contour(
            hull, model, axis, origin, normalise, **settings
        )

    title = (
        f"Visible Spectrum Section - "
        f"{f'{origin * 100}%' if normalise else origin} - "
        f"{model} - "
        f"{cmfs.strict_name}"
    )

    plane = MAPPING_AXIS_TO_PLANE[axis]

    labels = np.array(COLOURSPACE_MODELS_AXIS_LABELS[model])[
        as_int_array(colourspace_model_axis_reorder([0, 1, 2], model))
    ]
    x_label, y_label = labels[plane[0]], labels[plane[1]]

    settings.update(
        {
            "axes": axes,
            "standalone": True,
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
        }
    )
    settings.update(kwargs)

    return render(**settings)


@required("trimesh")
@override_style()
def plot_RGB_colourspace_section(
    colourspace: Union[
        RGB_Colourspace, str, Sequence[Union[RGB_Colourspace, str]]
    ],
    model: Union[
        Literal[
            "CAM02LCD",
            "CAM02SCD",
            "CAM02UCS",
            "CAM16LCD",
            "CAM16SCD",
            "CAM16UCS",
            "CIE XYZ",
            "CIE xyY",
            "CIE Lab",
            "CIE Luv",
            "CIE UCS",
            "CIE UVW",
            "DIN99",
            "Hunter Lab",
            "Hunter Rdab",
            "ICaCb",
            "ICtCp",
            "IPT",
            "IgPgTg",
            "Jzazbz",
            "OSA UCS",
            "Oklab",
            "hdr-CIELAB",
            "hdr-IPT",
        ],
        str,
    ] = "CIE xyY",
    axis: Union[Literal["+z", "+x", "+y"], str] = "+z",
    origin: Floating = 0.5,
    normalise: Boolean = True,
    show_section_colours: Boolean = True,
    show_section_contour: Boolean = True,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given *RGB* colourspace section colours along given axis and origin.

    Parameters
    ----------
    colourspace
        *RGB* colourspace of the *RGB* array. ``colourspace`` can be of any
        type or form supported by the
        :func:`colour.plotting.filter_RGB_colourspaces` definition.
    model
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    axis
        Axis the hull section will be normal to.
    origin
        Coordinate along ``axis`` at which to plot the hull section.
    normalise
        Whether to normalise ``axis`` to the extent of the hull along it.
    show_section_colours
        Whether to show the hull section colours.
    show_section_contour
        Whether to show the hull section contour.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`,
        :func:`colour.plotting.section.plot_hull_section_colours`
        :func:`colour.plotting.section.plot_hull_section_contour`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
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

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    colourspace = cast(
        RGB_Colourspace,
        first_item(filter_RGB_colourspaces(colourspace).values()),
    )

    vertices, faces, _outline = primitive_cube(1, 1, 1, 64, 64, 64)
    XYZ_vertices = RGB_to_XYZ(
        vertices["position"] + 0.5,
        colourspace.whitepoint,
        colourspace.whitepoint,
        colourspace.matrix_RGB_to_XYZ,
    )
    hull = trimesh.Trimesh(XYZ_vertices, faces, process=False)

    if show_section_colours:
        settings = {"axes": axes}
        settings.update(kwargs)
        settings["standalone"] = False

        plot_hull_section_colours(
            hull, model, axis, origin, normalise, **settings
        )

    if show_section_contour:
        settings = {"axes": axes}
        settings.update(kwargs)
        settings["standalone"] = False

        plot_hull_section_contour(
            hull, model, axis, origin, normalise, **settings
        )

    title = (
        f"{colourspace.name} Section - "
        f"{f'{origin * 100}%' if normalise else origin} - "
        f"{model}"
    )

    plane = MAPPING_AXIS_TO_PLANE[axis]

    labels = np.array(COLOURSPACE_MODELS_AXIS_LABELS[model])[
        as_int_array(colourspace_model_axis_reorder([0, 1, 2], model))
    ]
    x_label, y_label = labels[plane[0]], labels[plane[1]]

    settings.update(
        {
            "axes": axes,
            "standalone": True,
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
        }
    )
    settings.update(kwargs)

    return render(**settings)
